/*
 * gdr_probe: which GPU memory can the NIC (libfabric CXI) still use after a
 * CUDA checkpoint and restore?  Extends gdr_ckpt (phase 0b), which found that
 * no GPU memory of a restored process can be registered.  No NCCL, no DMTCP.
 *
 * The probe forks a helper process that is never checkpointed.  The driver
 * script runs `cuda-checkpoint --toggle --pid PID` twice on the probe only
 * (see gdr_probe.sh).  Before and after, the probe registers:
 *
 *   A    cudaMalloc'd before the checkpoint
 *   B    cudaMalloc'd after the restore
 *   V    cuMemCreate with gpuDirectRDMACapable, after the restore
 *   C    cuMemAlloc in a new (non-primary) context, after the restore
 *   H    allocated by the helper (cuMemCreate, gpuDirectRDMACapable), mapped
 *        here through a POSIX fd handle, registered through the helper's
 *        dma-buf fd
 *   X    A's dma-buf fd, exported here after the restore, registered by the
 *        helper in its own libfabric domain
 *
 * each as iov (libfabric's CUDA support), dma-buf with iface CUDA, and dma-buf
 * with iface SYSTEM (no CUDA in libfabric: what netsplit's lower half could
 * do).  A registration that succeeds is checked with a loopback send from a
 * host buffer into the GPU buffer, read back with cudaMemcpy.
 *
 *   REG  <phase> <buf> <path> ok|<error>
 *   XFER <phase> <buf> <path> ok|bad data|<error>
 *
 *   gdr_probe -d DIR          (the helper is "gdr_probe -H SOCKET")
 */
#define _GNU_SOURCE
#include <cuda.h>
#include <cuda_runtime.h>
#include <rdma/fabric.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/uio.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#define SIZE (8UL << 20)
#define XS (1UL << 20)

static int g_dev;
static const char *g_role = "probe";

struct fab {
    struct fi_info *info;
    struct fid_fabric *fabric;
    struct fid_domain *domain;
    struct fid_av *av;
    struct fid_cq *cq;
    struct fid_ep *ep;
    fi_addr_t self;
    int hmem;            /* FI_HMEM granted */
    char *hsrc, *hdst;   /* host buffers for the loopback test */
    struct fid_mr *hsrc_mr, *hdst_mr;
};

static void die(const char *what, long rc) {
    printf("FAIL [%s] %s: %ld (%s)\n", g_role, what, rc, rc < 0 ? fi_strerror((int)-rc) : "");
    fflush(stdout);
    exit(1);
}

#define CU(call)                                                                   \
    do {                                                                           \
        CUresult r_ = (call);                                                      \
        if (r_ != CUDA_SUCCESS)                                                    \
            die(#call, (long)r_);                                                  \
    } while (0)
#define CR(call)                                                                   \
    do {                                                                           \
        cudaError_t e_ = (call);                                                   \
        if (e_ != cudaSuccess)                                                     \
            die(#call, (long)e_);                                                  \
    } while (0)

/* ---- libfabric ----------------------------------------------------------- */

static int mr_finish(struct fab *f, struct fid_mr *mr) {
    int rc = 0;
    if (f->info->domain_attr->mr_mode & FI_MR_ENDPOINT) {
        rc = fi_mr_bind(mr, &f->ep->fid, 0);
        if (rc == 0)
            rc = fi_mr_enable(mr);
    }
    return rc;
}

static struct fid_mr *reg_host(struct fab *f, void *p, size_t n) {
    struct fid_mr *mr = NULL;
    int rc = fi_mr_reg(f->domain, p, n, FI_SEND | FI_RECV, 0, 0, 0, &mr, NULL);
    if (rc == 0)
        rc = mr_finish(f, mr);
    if (rc != 0)
        die("host fi_mr_reg", rc);
    return mr;
}

static void fab_open(struct fab *f) {
    struct fi_info *hints, *info = NULL, *pick = NULL;
    char want[16];
    int rc = -1;
    memset(f, 0, sizeof(*f));
    /* With FI_HMEM=system the provider offers no FI_HMEM: ask without it. */
    for (int hmem = 1; hmem >= 0 && rc != 0; hmem--) {
        hints = fi_allocinfo();
        hints->caps = FI_MSG | FI_RMA | FI_LOCAL_COMM | FI_REMOTE_COMM | (hmem ? FI_HMEM : 0);
        hints->ep_attr->type = FI_EP_RDM;
        hints->domain_attr->mr_mode = FI_MR_LOCAL | FI_MR_VIRT_ADDR | FI_MR_ALLOCATED |
                                      FI_MR_PROV_KEY | FI_MR_ENDPOINT | (hmem ? FI_MR_HMEM : 0);
        hints->fabric_attr->prov_name = strdup("cxi");
        rc = fi_getinfo(FI_VERSION(1, 18), NULL, NULL, 0, hints, &info);
        fi_freeinfo(hints);
        f->hmem = hmem;
    }
    if (rc != 0)
        die("fi_getinfo", rc);
    snprintf(want, sizeof(want), "cxi%d", g_dev);
    for (struct fi_info *i = info; i != NULL; i = i->next)
        if (pick == NULL && strcmp(i->domain_attr->name, want) == 0)
            pick = i;
    f->info = fi_dupinfo(pick != NULL ? pick : info);
    fi_freeinfo(info);
    if ((rc = fi_fabric(f->info->fabric_attr, &f->fabric, NULL)) != 0)
        die("fi_fabric", rc);
    if ((rc = fi_domain(f->fabric, f->info, &f->domain, NULL)) != 0)
        die("fi_domain", rc);
    struct fi_av_attr av = {.type = FI_AV_TABLE};
    if ((rc = fi_av_open(f->domain, &av, &f->av, NULL)) != 0)
        die("fi_av_open", rc);
    struct fi_cq_attr cq = {.format = FI_CQ_FORMAT_CONTEXT};
    if ((rc = fi_cq_open(f->domain, &cq, &f->cq, NULL)) != 0)
        die("fi_cq_open", rc);
    if ((rc = fi_endpoint(f->domain, f->info, &f->ep, NULL)) != 0)
        die("fi_endpoint", rc);
    if ((rc = fi_ep_bind(f->ep, &f->av->fid, 0)) != 0)
        die("fi_ep_bind av", rc);
    if ((rc = fi_ep_bind(f->ep, &f->cq->fid, FI_TRANSMIT | FI_RECV)) != 0)
        die("fi_ep_bind cq", rc);
    if ((rc = fi_enable(f->ep)) != 0)
        die("fi_enable", rc);
    char name[256];
    size_t len = sizeof(name);
    if ((rc = fi_getname(&f->ep->fid, name, &len)) != 0)
        die("fi_getname", rc);
    if ((rc = fi_av_insert(f->av, name, 1, &f->self, 0, NULL)) != 1)
        die("fi_av_insert", rc);
    f->hsrc = aligned_alloc(4096, XS);
    f->hdst = aligned_alloc(4096, XS);
    f->hsrc_mr = reg_host(f, f->hsrc, XS);
    f->hdst_mr = reg_host(f, f->hdst, XS);
    printf("INFO [%s] domain %s, FI_HMEM %s, mr_mode 0x%x\n", g_role,
           f->info->domain_attr->name, f->hmem ? "yes" : "no", f->info->domain_attr->mr_mode);
}

/* Sends XS bytes from the host buffer to dst (described by desc) on our own
 * address; 0 when both completions arrive. */
static int loopback(struct fab *f, void *dst, void *desc, unsigned char pat) {
    struct fi_context rctx, sctx;
    memset(f->hsrc, pat, XS);
    int rc = (int)fi_recv(f->ep, dst, XS, desc, FI_ADDR_UNSPEC, &rctx);
    if (rc != 0)
        return rc;
    rc = (int)fi_send(f->ep, f->hsrc, XS, fi_mr_desc(f->hsrc_mr), f->self, &sctx);
    if (rc != 0)
        return rc;
    int done = 0;
    time_t t0 = time(NULL);
    while (done < 2 && time(NULL) - t0 < 5) {
        struct fi_cq_entry e[2];
        ssize_t n = fi_cq_read(f->cq, e, 2);
        if (n > 0) {
            done += (int)n;
        } else if (n == -FI_EAVAIL) {
            struct fi_cq_err_entry err = {0};
            fi_cq_readerr(f->cq, &err, 0);
            return -err.err;
        } else if (n != -FI_EAGAIN) {
            return (int)n;
        }
    }
    return done == 2 ? 0 : -FI_ETIMEDOUT;
}

/* Registers [ptr, ptr+SIZE): path 0 iov (iface CUDA), 1 dma-buf iface CUDA,
 * 2 dma-buf iface SYSTEM.  dmabuf_fd < 0: export it here.  When it works and
 * check_va is set, runs the loopback test into ptr and reads it back. */
static const char *PATHS[] = {"iov", "dmabuf", "dmabuf-sys"};

static int reg_test(struct fab *f, const char *phase, const char *buf, void *ptr, int path,
                    int dmabuf_fd, int check_va) {
    struct iovec iov = {ptr, SIZE};
    struct fi_mr_dmabuf db = {0};
    struct fi_mr_attr attr;
    struct fid_mr *mr = NULL;
    uint64_t flags = 0;
    int fd = dmabuf_fd, own_fd = 0, rc;
    memset(&attr, 0, sizeof(attr));
    attr.iov_count = 1;
    attr.access = FI_SEND | FI_RECV | FI_READ | FI_WRITE | FI_REMOTE_READ | FI_REMOTE_WRITE;
    attr.iface = path == 2 ? FI_HMEM_SYSTEM : FI_HMEM_CUDA;
    if (attr.iface == FI_HMEM_CUDA)
        attr.device.cuda = g_dev;
    if (path >= 1) {
        if (fd < 0) {
            CUresult cr = cuMemGetHandleForAddressRange(&fd, (CUdeviceptr)ptr, SIZE,
                                                        CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0);
            if (cr != CUDA_SUCCESS) {
                printf("REG  %-7s %s %-10s no dma-buf fd (cuMemGetHandleForAddressRange %d)\n",
                       phase, buf, PATHS[path], (int)cr);
                return -1;
            }
            own_fd = 1;
        }
        db.fd = fd;
        db.offset = 0;
        db.len = SIZE;
        db.base_addr = ptr;
        attr.dmabuf = &db;
        flags = FI_MR_DMABUF;
    } else {
        attr.mr_iov = &iov;
    }
    rc = fi_mr_regattr(f->domain, &attr, flags, &mr);
    if (rc == 0)
        rc = mr_finish(f, mr);
    printf("REG  %-7s %s %-10s %s\n", phase, buf, PATHS[path], rc == 0 ? "ok" : fi_strerror(-rc));
    if (rc == 0 && check_va) {
        unsigned char pat = (unsigned char)(0x40 + path), *back = malloc(XS);
        int x = loopback(f, ptr, fi_mr_desc(mr), pat);
        const char *what = x != 0 ? fi_strerror(-x) : NULL;
        if (x == 0) {
            if (cudaMemcpy(back, ptr, XS, cudaMemcpyDeviceToHost) != cudaSuccess)
                what = "cudaMemcpy failed";
            else
                what = (back[0] == pat && back[XS - 1] == pat) ? "ok" : "bad data";
        }
        printf("XFER %-7s %s %-10s %s\n", phase, buf, PATHS[path], what);
        free(back);
    }
    fflush(stdout);
    if (mr != NULL)
        fi_close(&mr->fid);
    if (own_fd)
        close(fd);
    return rc;
}

static void reg_paths(struct fab *f, const char *phase, const char *buf, void *ptr) {
    for (int p = 0; p < 3; p++)
        if (p != 0 || f->hmem)
            reg_test(f, phase, buf, ptr, p, -1, 1);
}

/* ---- fd passing ---------------------------------------------------------- */

struct msg {
    int op;     /* 1 alloc, 2 register a dma-buf fd, 0 quit */
    int status; /* reply */
    char text[200];
};

static void send_msg(int s, struct msg *m, int *fds, int nfd) {
    struct iovec iov = {m, sizeof(*m)};
    char cbuf[CMSG_SPACE(2 * sizeof(int))];
    struct msghdr h = {.msg_iov = &iov, .msg_iovlen = 1};
    if (nfd > 0) {
        memset(cbuf, 0, sizeof(cbuf));
        h.msg_control = cbuf;
        h.msg_controllen = CMSG_SPACE(nfd * sizeof(int));
        struct cmsghdr *c = CMSG_FIRSTHDR(&h);
        c->cmsg_level = SOL_SOCKET;
        c->cmsg_type = SCM_RIGHTS;
        c->cmsg_len = CMSG_LEN(nfd * sizeof(int));
        memcpy(CMSG_DATA(c), fds, nfd * sizeof(int));
    }
    if (sendmsg(s, &h, 0) != (ssize_t)sizeof(*m))
        die("sendmsg", -1);
}

static int recv_msg(int s, struct msg *m, int *fds, int maxfd) {
    struct iovec iov = {m, sizeof(*m)};
    char cbuf[CMSG_SPACE(2 * sizeof(int))];
    struct msghdr h = {.msg_iov = &iov, .msg_iovlen = 1, .msg_control = cbuf,
                       .msg_controllen = sizeof(cbuf)};
    if (recvmsg(s, &h, 0) != (ssize_t)sizeof(*m))
        return -1;
    int n = 0;
    for (struct cmsghdr *c = CMSG_FIRSTHDR(&h); c != NULL; c = CMSG_NXTHDR(&h, c))
        if (c->cmsg_type == SCM_RIGHTS) {
            n = (int)((c->cmsg_len - CMSG_LEN(0)) / sizeof(int));
            if (n > maxfd)
                n = maxfd;
            memcpy(fds, CMSG_DATA(c), n * sizeof(int));
        }
    return n;
}

/* ---- VMM allocations ----------------------------------------------------- */

static CUdeviceptr vmm_map(CUmemGenericAllocationHandle h) {
    CUdeviceptr va;
    CU(cuMemAddressReserve(&va, SIZE, 0, 0, 0));
    CU(cuMemMap(va, SIZE, 0, h, 0));
    CUmemAccessDesc ad = {.location = {.type = CU_MEM_LOCATION_TYPE_DEVICE, .id = g_dev},
                          .flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE};
    CU(cuMemSetAccess(va, SIZE, &ad, 1));
    return va;
}

static CUdeviceptr vmm_alloc(int shareable, CUmemGenericAllocationHandle *out) {
    CUmemAllocationProp prop;
    memset(&prop, 0, sizeof(prop));
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = g_dev;
    prop.allocFlags.gpuDirectRDMACapable = 1;
    if (shareable)
        prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    CU(cuMemCreate(out, SIZE, &prop, 0));
    return vmm_map(*out);
}

/* ---- helper -------------------------------------------------------------- */

static int helper(const char *sock_path) {
    g_role = "helper";
    int ls = socket(AF_UNIX, SOCK_STREAM, 0);
    struct sockaddr_un a = {.sun_family = AF_UNIX};
    if (strlen(sock_path) >= sizeof(a.sun_path))
        die("socket path too long", -1);
    memcpy(a.sun_path, sock_path, strlen(sock_path) + 1);
    unlink(sock_path);
    if (bind(ls, (struct sockaddr *)&a, sizeof(a)) != 0 || listen(ls, 1) != 0)
        die("bind/listen", -1);
    int s = accept(ls, NULL, NULL);
    int cuda_up = 0, fab_up = 0;
    struct fab f;
    CUdeviceptr own = 0;
    for (;;) {
        struct msg m;
        int fds[2];
        int n = recv_msg(s, &m, fds, 2);
        if (n < 0 || m.op == 0)
            break;
        if (!cuda_up) {
            CUdevice dev;
            CUcontext ctx;
            CU(cuInit(0));
            CU(cuDeviceGet(&dev, g_dev));
            CU(cuDevicePrimaryCtxRetain(&ctx, dev));
            CU(cuCtxSetCurrent(ctx));
            cuda_up = 1;
        }
        if (m.op == 1) {
            /* Never-checkpointed memory: a POSIX fd handle for the probe's
             * mapping and a dma-buf fd for the NIC. */
            CUmemGenericAllocationHandle h;
            CUdeviceptr va = vmm_alloc(1, &h);
            own = va;
            int out[2];
            CU(cuMemExportToShareableHandle(&out[0], h, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
                                            0));
            CU(cuMemGetHandleForAddressRange(&out[1], va, SIZE,
                                             CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0));
            m.status = 0;
            snprintf(m.text, sizeof(m.text), "allocated at helper VA %p", (void *)va);
            send_msg(s, &m, out, 2);
            close(out[0]);
            close(out[1]);
        } else if (m.op == 2) {
            /* Register the probe's dma-buf in a domain of a process that was
             * never checkpointed.  No mapping here: base address is nominal. */
            if (!fab_up) {
                fab_open(&f);
                fab_up = 1;
            }
            int rc = n == 1 ? reg_test(&f, m.text, "X", (void *)0x100000000UL, 1, fds[0], 0) : -1;
            int rc2 = n == 1 ? reg_test(&f, m.text, "X", (void *)0x100000000UL, 2, fds[0], 0) : -1;
            if (n == 1)
                close(fds[0]);
            m.status = rc == 0 || rc2 == 0 ? 0 : -1;
            send_msg(s, &m, NULL, 0);
        } else if (m.op == 3) {
            /* Control: the helper's own allocation in its own domain. */
            if (!fab_up) {
                fab_open(&f);
                fab_up = 1;
            }
            int ofd;
            reg_paths(&f, m.text, "Hself", (void *)own);
            CU(cuMemGetHandleForAddressRange(&ofd, own, SIZE, CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD,
                                             0));
            reg_test(&f, m.text, "Hnominal", (void *)0x100000000UL, 1, ofd, 0);
            close(ofd);
            send_msg(s, &m, NULL, 0);
        }
    }
    return 0;
}

/* Controls that need no mapping of the other process's memory (so nothing
 * that cuda-checkpoint would refuse is left before the checkpoint):
 * the helper's own memory in its own domain, the helper's memory registered
 * here, and A's dma-buf registered by the helper. */
static void helper_checks(int s, struct fab *f, const char *phase, void *a) {
    struct msg m;
    int fds[2], afd;
    memset(&m, 0, sizeof(m));
    snprintf(m.text, sizeof(m.text), "%s", phase);
    m.op = 1;
    send_msg(s, &m, NULL, 0);
    if (recv_msg(s, &m, fds, 2) != 2)
        die("helper alloc", -1);
    m.op = 3;
    snprintf(m.text, sizeof(m.text), "%s", phase);
    send_msg(s, &m, NULL, 0);
    recv_msg(s, &m, NULL, 0);
    reg_test(f, phase, "Hnominal", (void *)0x100000000UL, 1, fds[1], 0);
    reg_test(f, phase, "Hnominal", (void *)0x100000000UL, 2, fds[1], 0);
    close(fds[0]);
    close(fds[1]);
    if (cuMemGetHandleForAddressRange(&afd, (CUdeviceptr)a, SIZE,
                                      CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0) == CUDA_SUCCESS) {
        m.op = 2;
        snprintf(m.text, sizeof(m.text), "%s", phase);
        send_msg(s, &m, &afd, 1);
        recv_msg(s, &m, fds, 0);
        close(afd);
    } else {
        printf("REG  %-7s X no dma-buf fd\n", phase);
    }
}

/* ---- probe --------------------------------------------------------------- */

int main(int argc, char **argv) {
    const char *dir = NULL, *hsock = NULL;
    int c;
    while ((c = getopt(argc, argv, "d:H:")) != -1) {
        if (c == 'd')
            dir = optarg;
        else if (c == 'H')
            hsock = optarg;
    }
    g_dev = getenv("SLURM_LOCALID") ? atoi(getenv("SLURM_LOCALID")) : 0;
    if (hsock != NULL)
        return helper(hsock);
    if (dir == NULL) {
        fprintf(stderr, "usage: gdr_probe -d DIR\n");
        return 2;
    }
    setvbuf(stdout, NULL, _IOLBF, 0);

    /* The helper starts before any CUDA call here and stays out of the
     * checkpoint (the driver toggles this pid only). */
    char sock_path[4096];
    snprintf(sock_path, sizeof(sock_path), "%s/helper.sock", dir);
    pid_t hp = fork();
    if (hp == 0) {
        execl("/proc/self/exe", argv[0], "-H", sock_path, (char *)NULL);
        _exit(127);
    }
    int s = socket(AF_UNIX, SOCK_STREAM, 0);
    struct sockaddr_un sa = {.sun_family = AF_UNIX};
    if (strlen(sock_path) >= sizeof(sa.sun_path))
        die("socket path too long", -1);
    memcpy(sa.sun_path, sock_path, strlen(sock_path) + 1);
    for (int i = 0; connect(s, (struct sockaddr *)&sa, sizeof(sa)) != 0; i++) {
        if (i > 100)
            die("connect to helper", -1);
        usleep(50000);
    }

    CR(cudaSetDevice(g_dev));
    CR(cudaFree(0));
    void *a, *b;
    CR(cudaMalloc(&a, SIZE));
    CR(cudaMemset(a, 1, SIZE));
    CR(cudaDeviceSynchronize());

    struct fab f;
    fab_open(&f);
    /* Control: host to host. */
    int x = loopback(&f, f.hdst, fi_mr_desc(f.hdst_mr), 0x5a);
    printf("XFER control host->host %s\n",
           x != 0 ? fi_strerror(-x) : (f.hdst[0] == 0x5a ? "ok" : "bad data"));
    reg_paths(&f, "before", "A", a);
    helper_checks(s, &f, "before", a);

    char path[4096], tmp[4200];
    snprintf(path, sizeof(path), "%s/ready", dir);
    snprintf(tmp, sizeof(tmp), "%s.tmp", path);
    FILE *fp = fopen(tmp, "w");
    fprintf(fp, "%d\n", (int)getpid());
    fclose(fp);
    rename(tmp, path);
    printf("INFO pid %d waiting for the checkpoint and restore\n", (int)getpid());
    snprintf(path, sizeof(path), "%s/restored", dir);
    while (access(path, F_OK) != 0)
        usleep(100000);

    /* The CUDA state was checkpointed and restored in place. */
    CR(cudaMemset(a, 2, SIZE));
    CR(cudaDeviceSynchronize());
    reg_paths(&f, "after", "A", a);
    CR(cudaMalloc(&b, SIZE));
    reg_paths(&f, "after", "B", b);

    CUmemGenericAllocationHandle vh;
    void *v = (void *)vmm_alloc(0, &vh);
    reg_paths(&f, "after", "V", v);

    CUcontext primary, ctx;
    CUdevice dev;
    CU(cuCtxGetCurrent(&primary));
    CU(cuDeviceGet(&dev, g_dev));
    CUresult cr = cuCtxCreate(&ctx, NULL, 0, dev);
    if (cr == CUDA_SUCCESS) {
        CUdeviceptr cp;
        CU(cuMemAlloc(&cp, SIZE));
        reg_paths(&f, "after", "C", (void *)cp); /* the readback runs in this context */
        CU(cuCtxSetCurrent(primary));
    } else {
        printf("REG  after   C cuCtxCreate failed: %d\n", (int)cr);
    }

    helper_checks(s, &f, "after", a);

    /* Memory owned by the helper, mapped here. */
    struct msg m = {.op = 1};
    int fds[2];
    send_msg(s, &m, NULL, 0);
    if (recv_msg(s, &m, fds, 2) != 2)
        die("helper alloc", -1);
    printf("INFO helper %s\n", m.text);
    CUmemGenericAllocationHandle hh;
    CU(cuMemImportFromShareableHandle(&hh, (void *)(uintptr_t)fds[0],
                                      CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
    void *hv = (void *)vmm_map(hh);
    for (int p = 0; p < 3; p++)
        if (p != 0 || f.hmem)
            reg_test(&f, "after", "H", hv, p, p == 0 ? -1 : fds[1], 1);

    m.op = 0;
    send_msg(s, &m, NULL, 0);
    waitpid(hp, NULL, 0);
    printf("RESULT gdr_probe: done\n");
    return 0;
}
