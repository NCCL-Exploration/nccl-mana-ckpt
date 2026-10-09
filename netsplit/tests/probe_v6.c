/*
 * probe_v6: netsplit phase 0a.  Drives aws-ofi-nccl through ncclNet v6 the way
 * NCCL's network proxy would, in a process with no CUDA: does the back end
 * work with only the netsplit CUDA stub?
 *
 * Rank 0 (SLURM_PROCID) listens and receives, rank 1 connects and sends; the
 * listen handle goes through a file in -d DIR.  Host memory only.  At the end
 * each rank lists the CUDA libraries mapped in the process: with the stub
 * there must be none but the stub.
 *
 *   probe_v6 -p PLUGIN.so [-c STUB.so] -d DIR [-s BYTES] [-n ITERS]
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "net_v6.h"

static int g_rank, g_verbose;

static void logger(ncclDebugLogLevel level, unsigned long flags, const char *file, int line,
                   const char *fmt, ...) {
    (void)flags;
    if (level > (g_verbose ? NCCL_LOG_INFO : NCCL_LOG_WARN) && level != NCCL_LOG_ABORT)
        return;
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "[rank %d plugin %s:%d] ", g_rank, file, line);
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    va_end(ap);
}

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

#define CHECK(call)                                                                       \
    do {                                                                                  \
        ncclResult_t r_ = (call);                                                         \
        if (r_ != ncclSuccess) {                                                          \
            fprintf(stdout, "FAIL rank %d: %s returned %d\n", g_rank, #call, (int)r_);    \
            exit(1);                                                                      \
        }                                                                                 \
    } while (0)

static void die(const char *msg) {
    fprintf(stdout, "FAIL rank %d: %s\n", g_rank, msg);
    exit(1);
}

/* Prints every mapped library whose name says CUDA, NVML or GDRCopy. */
static void report_cuda_mappings(void) {
    FILE *f = fopen("/proc/self/maps", "r");
    char line[4096], seen[16][512];
    int nseen = 0;
    if (f == NULL)
        return;
    while (fgets(line, sizeof(line), f)) {
        char *path = strchr(line, '/');
        if (path == NULL)
            continue;
        path[strcspn(path, "\n")] = '\0';
        if (!strstr(path, "libcuda") && !strstr(path, "libnvidia") && !strstr(path, "libgdrapi") &&
            !strstr(path, "cudart_stub"))
            continue;
        int dup = 0;
        for (int i = 0; i < nseen; i++)
            dup |= strcmp(seen[i], path) == 0;
        if (!dup && nseen < 16)
            snprintf(seen[nseen++], sizeof(seen[0]), "%s", path);
    }
    fclose(f);
    printf("MAPS rank %d: %d CUDA-related libraries mapped\n", g_rank, nseen);
    for (int i = 0; i < nseen; i++)
        printf("MAPS rank %d:   %s\n", g_rank, seen[i]);
}

int main(int argc, char **argv) {
    const char *plugin = NULL, *stub = NULL, *dir = NULL;
    size_t size = 4 << 20;
    int iters = 200, c;
    while ((c = getopt(argc, argv, "p:c:d:s:n:v")) != -1) {
        switch (c) {
        case 'p': plugin = optarg; break;
        case 'c': stub = optarg; break;
        case 'd': dir = optarg; break;
        case 's': size = strtoul(optarg, NULL, 0); break;
        case 'n': iters = atoi(optarg); break;
        case 'v': g_verbose = 1; break;
        default: return 2;
        }
    }
    if (plugin == NULL || dir == NULL) {
        fprintf(stderr, "usage: probe_v6 -p PLUGIN.so [-c STUB.so] -d DIR [-s BYTES] [-n ITERS]\n");
        return 2;
    }
    g_rank = getenv("SLURM_PROCID") ? atoi(getenv("SLURM_PROCID")) : 0;
    int local = getenv("SLURM_LOCALID") ? atoi(getenv("SLURM_LOCALID")) : 0;

    /* The stub first, RTLD_GLOBAL: the plugin's DT_NEEDED libcudart.so.13 then
     * resolves to it by SONAME. */
    if (stub != NULL && dlopen(stub, RTLD_NOW | RTLD_GLOBAL) == NULL)
        die(dlerror());
    void *h = dlopen(plugin, RTLD_NOW | RTLD_LOCAL);
    if (h == NULL)
        die(dlerror());
    ncclNet_v6_t *net = (ncclNet_v6_t *)dlsym(h, "ncclNetPlugin_v6");
    if (net == NULL)
        die("no ncclNetPlugin_v6 in the plugin");

    CHECK(net->init(logger));
    int ndev = 0;
    CHECK(net->devices(&ndev));
    printf("INFO rank %d: plugin '%s', %d devices\n", g_rank, net->name, ndev);
    for (int d = 0; d < ndev; d++) {
        ncclNetProperties_v6_t p;
        memset(&p, 0, sizeof(p));
        CHECK(net->getProperties(d, &p));
        printf("INFO rank %d: dev %d name=%s pci=%s ptrSupport=0x%x speed=%d maxComms=%d maxRecvs=%d\n",
               g_rank, d, p.name ? p.name : "?", p.pciPath ? p.pciPath : "?", p.ptrSupport, p.speed,
               p.maxComms, p.maxRecvs);
    }
    if (ndev <= 0)
        die("no network device");
    int dev = local % ndev;

    char path[4096], tmp[4200];
    snprintf(path, sizeof(path), "%s/handle.bin", dir);
    char handle[NCCL_NET_HANDLE_MAXSIZE];
    void *lcomm = NULL, *scomm = NULL, *rcomm = NULL;
    double t0 = now();
    if (g_rank == 0) {
        memset(handle, 0, sizeof(handle));
        CHECK(net->listen(dev, handle, &lcomm));
        snprintf(tmp, sizeof(tmp), "%s.tmp", path);
        FILE *f = fopen(tmp, "w");
        if (f == NULL || fwrite(handle, sizeof(handle), 1, f) != 1)
            die("cannot write the handle file");
        fclose(f);
        rename(tmp, path);
        while (rcomm == NULL) {
            CHECK(net->accept(lcomm, &rcomm));
            if (now() - t0 > 120)
                die("accept timed out");
        }
    } else {
        FILE *f;
        while ((f = fopen(path, "r")) == NULL) {
            if (now() - t0 > 120)
                die("no handle file");
            usleep(10000);
        }
        if (fread(handle, sizeof(handle), 1, f) != 1)
            die("short handle file");
        fclose(f);
        while (scomm == NULL) {
            CHECK(net->connect(dev, handle, &scomm));
            if (now() - t0 > 120)
                die("connect timed out");
        }
    }
    printf("INFO rank %d: connected on dev %d in %.3f s\n", g_rank, dev, now() - t0);

    unsigned char *buf = aligned_alloc(4096, size);
    if (buf == NULL)
        die("out of memory");
    void *mh = NULL;
    CHECK(net->regMr(g_rank == 0 ? rcomm : scomm, buf, (int)size, NCCL_PTR_HOST, &mh));

    double t1 = now();
    for (int it = 0; it < iters; it++) {
        void *req = NULL;
        int done = 0, got = 0, tag = 1, sz = (int)size;
        if (g_rank == 1) {
            for (size_t i = 0; i < size; i += 4096)
                buf[i] = (unsigned char)(it * 31 + i / 4096);
            while (req == NULL)
                CHECK(net->isend(scomm, buf, (int)size, tag, mh, &req));
        } else {
            void *data = buf;
            while (req == NULL)
                CHECK(net->irecv(rcomm, 1, &data, &sz, &tag, &mh, &req));
        }
        while (!done) {
            CHECK(net->test(req, &done, &got));
            if (now() - t1 > 300)
                die("transfer timed out");
        }
        if (g_rank == 0) {
            for (size_t i = 0; i < size; i += 4096)
                if (buf[i] != (unsigned char)(it * 31 + i / 4096)) {
                    fprintf(stdout, "FAIL rank 0: iteration %d byte %zu = %u\n", it, i, buf[i]);
                    return 1;
                }
        }
    }
    double dt = now() - t1;
    printf("INFO rank %d: %d x %zu bytes in %.3f s (%.1f MB/s)\n", g_rank, iters, size, dt,
           iters * (double)size / dt / 1e6);

    CHECK(net->deregMr(g_rank == 0 ? rcomm : scomm, mh));
    if (scomm)
        CHECK(net->closeSend(scomm));
    if (rcomm)
        CHECK(net->closeRecv(rcomm));
    if (lcomm)
        CHECK(net->closeListen(lcomm));
    report_cuda_mappings();
    printf("RESULT probe_v6 rank %d: PASS\n", g_rank);
    fflush(stdout);
    return 0;
}
