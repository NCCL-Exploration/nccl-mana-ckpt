/*
 * gdr_ckpt: netsplit phase 0b.  Can GPU memory be registered with the NIC
 * (libfabric CXI) after a CUDA checkpoint and restore?  No NCCL, no DMTCP.
 *
 * After both NCCL-in-MANA's-lower-half and the single-process plugin, NCCL's
 * AWS-OFI plugin fails to register GPU memory after a checkpoint
 * ("NET/OFI Unable to register memory (type = 2) ... RC: 22").  This test
 * removes everything but CUDA and libfabric:
 *
 *   1. cudaMalloc buffer A; register it (iov path and explicit dma-buf),
 *      deregister.  Host memory as a control.
 *   2. Write DIR/ready (pid), wait for DIR/restored.  The driver meanwhile
 *      runs `cuda-checkpoint --toggle --pid PID` twice (checkpoint, restore).
 *   3. Register A again, a new buffer B, and both again in a freshly opened
 *      fabric/domain/endpoint (what a new netsplit lower half would do).
 *
 * Each registration prints one "REG <phase> <buffer> <path> ok|rc=..." line.
 *
 *   gdr_ckpt -d DIR
 */
#define _GNU_SOURCE
#include <cuda.h>
#include <cuda_runtime.h>
#include <rdma/fabric.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/uio.h>
#include <time.h>
#include <unistd.h>

#define SIZE (8UL << 20)

static int g_dev;

struct fab {
    struct fi_info *info;
    struct fid_fabric *fabric;
    struct fid_domain *domain;
    struct fid_av *av;
    struct fid_cq *cq;
    struct fid_ep *ep;
};

static void die(const char *what, int rc) {
    printf("FAIL %s: %d (%s)\n", what, rc, rc < 0 ? fi_strerror(-rc) : "");
    fflush(stdout);
    exit(1);
}

#define CU(call)                                                                   \
    do {                                                                           \
        cudaError_t e_ = (call);                                                   \
        if (e_ != cudaSuccess) {                                                   \
            printf("FAIL %s: %s\n", #call, cudaGetErrorString(e_));                \
            exit(1);                                                               \
        }                                                                          \
    } while (0)

static void fab_open(struct fab *f) {
    struct fi_info *hints = fi_allocinfo(), *info, *pick = NULL;
    char want[16];
    int rc;
    hints->caps = FI_MSG | FI_RMA | FI_HMEM | FI_LOCAL_COMM | FI_REMOTE_COMM;
    hints->ep_attr->type = FI_EP_RDM;
    hints->domain_attr->mr_mode = FI_MR_LOCAL | FI_MR_HMEM | FI_MR_VIRT_ADDR | FI_MR_ALLOCATED |
                                  FI_MR_PROV_KEY | FI_MR_ENDPOINT;
    hints->fabric_attr->prov_name = strdup("cxi");
    rc = fi_getinfo(FI_VERSION(1, 18), NULL, NULL, 0, hints, &info);
    if (rc != 0)
        die("fi_getinfo (FI_HMEM)", rc);
    snprintf(want, sizeof(want), "cxi%d", g_dev);
    for (struct fi_info *i = info; i != NULL; i = i->next)
        if (pick == NULL && strcmp(i->domain_attr->name, want) == 0)
            pick = i;
    if (pick == NULL)
        pick = info;
    f->info = fi_dupinfo(pick);
    fi_freeinfo(info);
    fi_freeinfo(hints);
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
    printf("INFO domain %s, mr_mode 0x%x\n", f->info->domain_attr->name,
           f->info->domain_attr->mr_mode);
}

static void fab_close(struct fab *f) {
    fi_close(&f->ep->fid);
    fi_close(&f->cq->fid);
    fi_close(&f->av->fid);
    fi_close(&f->domain->fid);
    fi_close(&f->fabric->fid);
    fi_freeinfo(f->info);
    memset(f, 0, sizeof(*f));
}

/* Registers [ptr, ptr+size) and deregisters it; prints one REG line. */
static int reg(struct fab *f, const char *phase, const char *buf, void *ptr, size_t size,
               enum fi_hmem_iface iface, int dmabuf) {
    struct iovec iov = {ptr, size};
    struct fi_mr_dmabuf db = {0};
    struct fi_mr_attr attr;
    struct fid_mr *mr = NULL;
    uint64_t flags = 0;
    int fd = -1, rc;
    memset(&attr, 0, sizeof(attr));
    attr.iov_count = 1;
    attr.access = FI_SEND | FI_RECV | FI_READ | FI_WRITE | FI_REMOTE_READ | FI_REMOTE_WRITE;
    attr.iface = iface;
    if (iface == FI_HMEM_CUDA)
        attr.device.cuda = g_dev;
    if (dmabuf) {
        CUresult cr = cuMemGetHandleForAddressRange(&fd, (CUdeviceptr)ptr, size,
                                                    CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0);
        if (cr != CUDA_SUCCESS) {
            printf("REG %-7s %s dmabuf  no fd: cuMemGetHandleForAddressRange=%d\n", phase, buf,
                   (int)cr);
            return -1;
        }
        db.fd = fd;
        db.offset = 0;
        db.len = size;
        db.base_addr = ptr;
        attr.dmabuf = &db;
        flags = FI_MR_DMABUF;
    } else {
        attr.mr_iov = &iov;
    }
    rc = fi_mr_regattr(f->domain, &attr, flags, &mr);
    if (rc == 0 && (f->info->domain_attr->mr_mode & FI_MR_ENDPOINT)) {
        rc = fi_mr_bind(mr, &f->ep->fid, 0);
        if (rc == 0)
            rc = fi_mr_enable(mr);
    }
    printf("REG %-7s %s %-7s %s\n", phase, buf, dmabuf ? "dmabuf" : "iov",
           rc == 0 ? "ok" : fi_strerror(-rc));
    if (mr != NULL)
        fi_close(&mr->fid);
    if (fd >= 0)
        close(fd);
    return rc;
}

static void reg_all(struct fab *f, const char *phase, const char *name, void *gpu, void *host) {
    reg(f, phase, name, gpu, SIZE, FI_HMEM_CUDA, 0);
    reg(f, phase, name, gpu, SIZE, FI_HMEM_CUDA, 1);
    if (host != NULL)
        reg(f, phase, "host", host, SIZE, FI_HMEM_SYSTEM, 0);
}

int main(int argc, char **argv) {
    const char *dir = NULL;
    int c;
    while ((c = getopt(argc, argv, "d:")) != -1)
        if (c == 'd')
            dir = optarg;
    if (dir == NULL) {
        fprintf(stderr, "usage: gdr_ckpt -d DIR\n");
        return 2;
    }
    g_dev = getenv("SLURM_LOCALID") ? atoi(getenv("SLURM_LOCALID")) : 0;
    CU(cudaSetDevice(g_dev));
    CU(cudaFree(0));
    void *a, *b, *host = aligned_alloc(4096, SIZE);
    CU(cudaMalloc(&a, SIZE));
    CU(cudaMemset(a, 1, SIZE));
    CU(cudaDeviceSynchronize());

    struct fab f;
    memset(&f, 0, sizeof(f));
    fab_open(&f);
    reg_all(&f, "before", "A", a, host);

    char path[4096], tmp[4200];
    snprintf(path, sizeof(path), "%s/ready", dir);
    snprintf(tmp, sizeof(tmp), "%s.tmp", path);
    FILE *fp = fopen(tmp, "w");
    fprintf(fp, "%d\n", (int)getpid());
    fclose(fp);
    rename(tmp, path);
    printf("INFO pid %d waiting for the checkpoint and restore\n", (int)getpid());
    fflush(stdout);
    snprintf(path, sizeof(path), "%s/restored", dir);
    while (access(path, F_OK) != 0)
        usleep(100000);

    /* The CUDA state was checkpointed and restored in place. */
    CU(cudaMemset(a, 2, SIZE));
    CU(cudaDeviceSynchronize());
    reg_all(&f, "after", "A", a, host);
    CU(cudaMalloc(&b, SIZE));
    reg_all(&f, "after", "B", b, NULL);

    /* A new lower half would open everything again. */
    fab_close(&f);
    fab_open(&f);
    reg_all(&f, "fresh", "A", a, host);
    reg_all(&f, "fresh", "B", b, NULL);
    fab_close(&f);
    printf("RESULT gdr_ckpt: done\n");
    return 0;
}
