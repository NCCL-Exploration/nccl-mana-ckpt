/*
 * ckpt_race: does a thread that keeps running while its own process is
 * checkpointed with the CUDA checkpoint API get CUDA error 719 or SIGSEGV?
 * No NCCL, no DMTCP.
 *
 * NAS IS over NCCL fails at 2x4 in about half of its checkpoints: on some
 * ranks the main thread's cudaDeviceSynchronize() returns 719
 * (CUDA_ERROR_LAUNCH_FAILED) during the GPU checkpoint, and the context
 * keeps that error through the image; on others the main thread segfaults.
 * IS launches no kernels of its own.  It sorts on the CPU over cudaMallocHost
 * buffers and calls cudaDeviceSynchronize() before each NCCL call, and
 * nothing stops it during DMTCP_EVENT_PRESUSPEND, where DMTCP's CUDA plugin
 * (dmtcp/plugin/cuda/cuda-ckpt.cpp) locks and checkpoints the GPU and then
 * maps PROT_NONE over every /dev/nvidia* mapping.  This test does the same
 * to itself while a worker thread does what IS does:
 *
 *   ckpt_race [-m sync|touch|both|idle] [-r] [-n N] [-s MiB]
 *     -m  the worker's loop: cudaDeviceSynchronize(), writes to the pinned
 *         buffer, both alternately (IS), or nothing (control)
 *     -r  also map PROT_NONE over the /dev/nvidia* mappings after the
 *         checkpoint and unmap them before the restore, as the plugin does
 *     -n  checkpoint/restore cycles (default 10)
 *     -s  size of the pinned and the device buffer in MiB (default 1024)
 *
 * Prints the pinned buffer's /proc/self/maps line before and after the
 * first checkpoint, one "CYCLE" line per cycle, and "RESULT ok" or
 * "RESULT fail ..." at the end.  A SIGSEGV prints the faulting address and
 * its maps line, then exits with status 3.
 */
#define _GNU_SOURCE
#include <cuda.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#define MAX_AREAS 4096

enum mode { M_SYNC, M_TOUCH, M_BOTH, M_IDLE };

static enum mode g_mode = M_BOTH;
static size_t g_size = 1024UL << 20;
static unsigned char *g_pinned;
static void *g_dev;
static atomic_int g_stop;
static atomic_long g_passes;
static atomic_int g_err;          /* first error the worker saw */
static atomic_long g_err_pass;    /* the pass it saw it in */
static atomic_long g_in_cuda;     /* passes in which a cycle found it in CUDA */
static atomic_int g_worker_in_cuda;

struct area {
    void *addr;
    size_t size;
};
static struct area g_areas[MAX_AREAS];
static int g_nareas;

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

#define CU(call)                                                               \
    do {                                                                       \
        cudaError_t e_ = (call);                                               \
        if (e_ != cudaSuccess) {                                               \
            printf("FAIL %s: %d %s\n", #call, e_, cudaGetErrorString(e_));     \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

/* Async-signal-safe: writes the /proc/self/maps line that holds addr. */
static void write_maps_line(uintptr_t addr) {
    static char buf[1 << 20];
    int fd = open("/proc/self/maps", O_RDONLY);
    if (fd < 0)
        return;
    ssize_t n = 0, r;
    while (n < (ssize_t)sizeof(buf) - 1 && (r = read(fd, buf + n, sizeof(buf) - 1 - n)) > 0)
        n += r;
    close(fd);
    buf[n < 0 ? 0 : n] = '\0';
    for (char *line = buf; *line;) {
        char *end = strchr(line, '\n');
        if (end == NULL)
            end = line + strlen(line);
        uintptr_t lo = strtoul(line, NULL, 16);
        char *dash = strchr(line, '-');
        uintptr_t hi = dash != NULL ? strtoul(dash + 1, NULL, 16) : 0;
        if (addr >= lo && addr < hi) {
            if (write(1, line, end - line + (*end == '\n')) < 0)
                return;
            return;
        }
        line = *end ? end + 1 : end;
    }
    const char none[] = "(no mapping)\n";
    if (write(1, none, sizeof(none) - 1) < 0)
        return;
}

static void on_segv(int sig, siginfo_t *si, void *uc) {
    char msg[128];
    int n = snprintf(msg, sizeof(msg), "SIGSEGV at %p (worker in CUDA: %d); maps: ",
                     si->si_addr, atomic_load(&g_worker_in_cuda));
    if (write(1, msg, n) < 0)
        _exit(3);
    write_maps_line((uintptr_t)si->si_addr);
    _exit(3);
}

static void note_err(cudaError_t e, long pass) {
    int zero = 0;
    if (e != cudaSuccess && atomic_compare_exchange_strong(&g_err, &zero, (int)e))
        atomic_store(&g_err_pass, pass);
}

static void *worker(void *arg) {
    CU(cudaSetDevice(0));
    for (long pass = 0; !atomic_load(&g_stop); pass++) {
        if (g_mode == M_IDLE) {
            usleep(1000);
            continue;
        }
        if (g_mode == M_TOUCH || g_mode == M_BOTH) {
            /* One write per page, like IS's bucket loops over its keys. */
            for (size_t off = 0; off < g_size; off += 4096)
                g_pinned[off]++;
        }
        if (g_mode == M_SYNC || g_mode == M_BOTH) {
            atomic_store(&g_worker_in_cuda, 1);
            note_err(cudaDeviceSynchronize(), pass);
            atomic_store(&g_worker_in_cuda, 0);
        }
        atomic_fetch_add(&g_passes, 1);
    }
    return NULL;
}

static void print_maps_of(const char *tag, void *p) {
    printf("MAPS %s: ", tag);
    fflush(stdout);
    write_maps_line((uintptr_t)p);
}

/* The /dev/nvidia* mappings, as the plugin lists them between the lock and
 * the checkpoint. */
static void list_nvidia_areas(void) {
    FILE *f = fopen("/proc/self/maps", "r");
    char line[4096];
    g_nareas = 0;
    while (f != NULL && fgets(line, sizeof(line), f) != NULL && g_nareas < MAX_AREAS) {
        if (strstr(line, "/dev/nvidia") == NULL)
            continue;
        uintptr_t lo, hi;
        if (sscanf(line, "%lx-%lx", &lo, &hi) == 2) {
            g_areas[g_nareas].addr = (void *)lo;
            g_areas[g_nareas].size = hi - lo;
            g_nareas++;
        }
    }
    if (f != NULL)
        fclose(f);
}

static int covers_pinned(void) {
    for (int i = 0; i < g_nareas; i++)
        if ((unsigned char *)g_areas[i].addr <= g_pinned &&
            g_pinned < (unsigned char *)g_areas[i].addr + g_areas[i].size)
            return 1;
    return 0;
}

int main(int argc, char **argv) {
    int reserve = 0, cycles = 10, opt;
    while ((opt = getopt(argc, argv, "m:rn:s:")) != -1) {
        switch (opt) {
        case 'm':
            g_mode = !strcmp(optarg, "sync")    ? M_SYNC
                     : !strcmp(optarg, "touch") ? M_TOUCH
                     : !strcmp(optarg, "idle")  ? M_IDLE
                                                : M_BOTH;
            break;
        case 'r':
            reserve = 1;
            break;
        case 'n':
            cycles = atoi(optarg);
            break;
        case 's':
            g_size = strtoul(optarg, NULL, 10) << 20;
            break;
        default:
            fprintf(stderr, "usage: %s [-m sync|touch|both|idle] [-r] [-n N] [-s MiB]\n", argv[0]);
            return 2;
        }
    }
    struct sigaction sa = {0};
    sa.sa_sigaction = on_segv;
    sa.sa_flags = SA_SIGINFO;
    sigaction(SIGSEGV, &sa, NULL);
    sigaction(SIGBUS, &sa, NULL);

    CU(cudaSetDevice(0));
    CU(cudaMallocHost((void **)&g_pinned, g_size));
    memset(g_pinned, 0, g_size);
    CU(cudaMalloc(&g_dev, g_size));
    CU(cudaMemset(g_dev, 0x5a, g_size));
    CU(cudaDeviceSynchronize());
    printf("mode=%d reserve=%d cycles=%d size=%zu MiB pid=%d\n", g_mode, reserve, cycles,
           g_size >> 20, getpid());
    print_maps_of("pinned before", g_pinned);

    pthread_t t;
    pthread_create(&t, NULL, worker, NULL);
    srand(getpid());
    cuInit(0);
    int pid = getpid(), failed = 0;
    for (int c = 0; c < cycles && !failed; c++) {
        usleep(100000 + rand() % 200000);
        CUcheckpointLockArgs lock_args = {0};
        CUcheckpointCheckpointArgs ckpt_args = {0};
        CUcheckpointRestoreArgs restore_args = {0};
        CUcheckpointUnlockArgs unlock_args = {0};
        double t0 = now();
        CUresult r_lock = cuCheckpointProcessLock(pid, &lock_args);
        double t1 = now();
        list_nvidia_areas();
        int in_cuda = atomic_load(&g_worker_in_cuda);
        CUresult r_ckpt = cuCheckpointProcessCheckpoint(pid, &ckpt_args);
        double t2 = now();
        if (c == 0)
            print_maps_of("pinned after checkpoint", g_pinned);
        if (reserve) {
            for (int i = 0; i < g_nareas; i++)
                if (mmap(g_areas[i].addr, g_areas[i].size, PROT_NONE,
                         MAP_FIXED | MAP_ANONYMOUS | MAP_PRIVATE, -1, 0) == MAP_FAILED)
                    printf("mmap PROT_NONE failed at %p\n", g_areas[i].addr);
        }
        usleep(200000);
        if (reserve) {
            for (int i = 0; i < g_nareas; i++)
                munmap(g_areas[i].addr, g_areas[i].size);
        }
        double t3 = now();
        CUresult r_restore = cuCheckpointProcessRestore(pid, &restore_args);
        CUresult r_unlock = cuCheckpointProcessUnlock(pid, &unlock_args);
        double t4 = now();
        if (in_cuda)
            atomic_fetch_add(&g_in_cuda, 1);
        cudaError_t e_main = cudaDeviceSynchronize();
        unsigned char probe = 0;
        cudaError_t e_copy = cudaMemcpy(&probe, (char *)g_dev + g_size / 2, 1, cudaMemcpyDeviceToHost);
        printf("CYCLE %d lock=%d (%.3fs) ckpt=%d (%.3fs) restore=%d unlock=%d (%.3fs) "
               "nvidia_areas=%d covers_pinned=%d worker_in_cuda=%d worker_err=%d main_sync=%d "
               "copy=%d probe=0x%02x passes=%ld\n",
               c, r_lock, t1 - t0, r_ckpt, t2 - t1, r_restore, r_unlock, t4 - t3, g_nareas,
               covers_pinned(), in_cuda, atomic_load(&g_err), e_main, e_copy, probe,
               atomic_load(&g_passes));
        fflush(stdout);
        if (r_lock || r_ckpt || r_restore || r_unlock || atomic_load(&g_err) || e_main || e_copy ||
            probe != 0x5a)
            failed = 1;
    }
    atomic_store(&g_stop, 1);
    pthread_join(t, NULL);
    if (failed)
        printf("RESULT fail worker_err=%d (pass %ld) cycles_with_worker_in_cuda=%ld\n",
               atomic_load(&g_err), atomic_load(&g_err_pass), atomic_load(&g_in_cuda));
    else
        printf("RESULT ok cycles_with_worker_in_cuda=%ld\n", atomic_load(&g_in_cuda));
    return failed;
}
