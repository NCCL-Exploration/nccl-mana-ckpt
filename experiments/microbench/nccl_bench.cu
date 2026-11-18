/*
 * nccl_bench -- MPI-free NCCL microbenchmarks for the checkpoint report.
 *
 * Methodology follows OSU 7.x osu_xccl_*: warm-up iterations, host wall-clock
 * timing around stream-synchronized operations, ping-pong latency, windowed
 * bandwidth, per-iteration collective latency reported as avg/min/max over
 * ranks. Unlike OSU there is no MPI anywhere in the process: ranks come from
 * Slurm, the ncclUniqueId is exchanged through a file on a shared filesystem,
 * and barriers / timing reductions run on the benchmark's own communicator
 * (outside the timed region). Every message size is validated once, untimed,
 * so a run that crossed a checkpoint/restart proves its data is still right.
 *
 * Launch (one task per GPU):
 *   srun -N 2 --ntasks-per-node=4 --gpus-per-node=4 \
 *       nccl_bench -b allreduce -f $SCRATCH/run/id.bin
 *
 * Rank/size/local rank: SLURM_PROCID/SLURM_NTASKS/SLURM_LOCALID, falling back
 * to RANK/WORLD_SIZE/LOCAL_RANK for non-Slurm launchers.
 *
 * -DWITH_MPI_INIT (build with an MPI compiler wrapper): call MPI_Init and
 * MPI_Finalize and nothing else from MPI. For checkpointers such as MANA that
 * need MPI initialized; all communication is still NCCL.
 */

#include <cuda_runtime.h>
#ifdef WITH_MPI_INIT
#include <mpi.h>
#endif
#include <errno.h>
#include <getopt.h>
#include <math.h>
#include <nccl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define CUDA_CHECK(x)                                                                              \
    do {                                                                                           \
        cudaError_t e_ = (x);                                                                      \
        if (e_ != cudaSuccess) {                                                                   \
            fprintf(stderr, "[rank %d] CUDA error %s at %s:%d\n", g_rank, cudaGetErrorString(e_),  \
                    __FILE__, __LINE__);                                                           \
            exit(2);                                                                               \
        }                                                                                          \
    } while (0)
#define NCCL_CHECK(x)                                                                              \
    do {                                                                                           \
        ncclResult_t r_ = (x);                                                                     \
        if (r_ != ncclSuccess) {                                                                   \
            fprintf(stderr, "[rank %d] NCCL error %s at %s:%d\n", g_rank, ncclGetErrorString(r_),  \
                    __FILE__, __LINE__);                                                           \
            exit(2);                                                                               \
        }                                                                                          \
    } while (0)

enum bench_t { LATENCY, BW, BIBW, ALLREDUCE, ALLGATHER, BCAST, REDUCE, REDUCE_SCATTER, ALLTOALL };
static const char *bench_names[] = {"latency", "bw",     "bibw",           "allreduce", "allgather",
                                    "bcast",   "reduce", "reduce_scatter", "alltoall"};
#define NBENCH (int)(sizeof(bench_names) / sizeof(bench_names[0]))

static int g_rank, g_nranks, g_local;
static ncclComm_t g_comm;
static cudaStream_t g_stream;
static float *g_scratch; /* device scratch for barrier / reductions */

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static int env_int(const char *a, const char *b, int dflt) {
    const char *s = getenv(a);
    if (!s && b)
        s = getenv(b);
    return s ? atoi(s) : dflt;
}

/* ---- bootstrap: exchange the ncclUniqueId through a shared file ---------- */

/* The file carries the Slurm job and step id, so a stale file from an earlier
 * step (same path) is never mistaken for this step's id. */
typedef struct {
    char magic[8];
    char tag[64];
    ncclUniqueId id;
} id_file_t;

static void make_tag(char *tag, size_t n) {
    snprintf(tag, n, "%s.%s.%d", getenv("SLURM_JOB_ID") ? getenv("SLURM_JOB_ID") : "nojob",
             getenv("SLURM_STEP_ID") ? getenv("SLURM_STEP_ID") : "nostep", g_nranks);
}

static void bootstrap(const char *path, ncclUniqueId *id) {
    id_file_t rec;
    char tag[64];
    make_tag(tag, sizeof(tag));
    if (g_rank == 0) {
        char tmp[4096];
        memset(&rec, 0, sizeof(rec));
        memcpy(rec.magic, "NCCLBID", 8);
        snprintf(rec.tag, sizeof(rec.tag), "%s", tag);
        NCCL_CHECK(ncclGetUniqueId(&rec.id));
        snprintf(tmp, sizeof(tmp), "%s.tmp.%d", path, (int)getpid());
        FILE *f = fopen(tmp, "wb");
        if (!f || fwrite(&rec, sizeof(rec), 1, f) != 1 || fclose(f) != 0) {
            fprintf(stderr, "cannot write %s: %s\n", tmp, strerror(errno));
            exit(2);
        }
        if (rename(tmp, path) != 0) { /* atomic: readers never see a partial file */
            fprintf(stderr, "cannot rename to %s: %s\n", path, strerror(errno));
            exit(2);
        }
        *id = rec.id;
        return;
    }
    double t0 = now_s();
    for (;;) {
        FILE *f = fopen(path, "rb");
        if (f) {
            size_t got = fread(&rec, sizeof(rec), 1, f);
            fclose(f);
            if (got == 1 && memcmp(rec.magic, "NCCLBID", 8) == 0 && strcmp(rec.tag, tag) == 0) {
                *id = rec.id;
                return;
            }
        }
        if (now_s() - t0 > 600) {
            fprintf(stderr, "[rank %d] timed out waiting for %s (tag %s)\n", g_rank, path, tag);
            exit(2);
        }
        usleep(50000);
    }
}

/* ---- helpers on the benchmark communicator (never inside timed regions) -- */

static void sync_stream(void) {
    CUDA_CHECK(cudaStreamSynchronize(g_stream));
}

static void barrier(void) {
    NCCL_CHECK(ncclAllReduce(g_scratch, g_scratch, 1, ncclFloat, ncclSum, g_comm, g_stream));
    sync_stream();
}

/* Reduce one double across ranks (op = ncclSum/ncclMin/ncclMax); result on all ranks. */
static double reduce_double(double v, ncclRedOp_t op) {
    double *d = (double *)g_scratch;
    CUDA_CHECK(cudaMemcpyAsync(d, &v, sizeof(v), cudaMemcpyHostToDevice, g_stream));
    NCCL_CHECK(ncclAllReduce(d, d, 1, ncclDouble, op, g_comm, g_stream));
    CUDA_CHECK(cudaMemcpyAsync(&v, d, sizeof(v), cudaMemcpyDeviceToHost, g_stream));
    sync_stream();
    return v;
}

/* ---- device fill / check -------------------------------------------------- */

__global__ void fill_kernel(float *p, size_t n, float v) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x)
        p[i] = v;
}
__global__ void fill_blocks_kernel(float *p, size_t block, int nblocks, float base, float step) {
    size_t n = block * nblocks;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x)
        p[i] = base + step * (float)(i / block);
}
__global__ void count_bad_kernel(const float *p, size_t block, int nblocks, float base, float step,
                                 unsigned long long *bad) {
    size_t n = block * nblocks;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x)
        if (p[i] != base + step * (float)(i / block))
            atomicAdd(bad, 1ULL);
}

static void fill(float *p, size_t n, float v) {
    if (n)
        fill_kernel<<<256, 256, 0, g_stream>>>(p, n, v);
}
static void fill_blocks(float *p, size_t block, int nblocks, float base, float step) {
    if (block)
        fill_blocks_kernel<<<256, 256, 0, g_stream>>>(p, block, nblocks, base, step);
}
/* Number of elements of p (nblocks blocks of `block` floats) != base + step*blockidx. */
static unsigned long long count_bad(const float *p, size_t block, int nblocks, float base,
                                    float step) {
    unsigned long long *d = (unsigned long long *)g_scratch, h = 0;
    if (!block)
        return 0;
    CUDA_CHECK(cudaMemsetAsync(d, 0, sizeof(h), g_stream));
    count_bad_kernel<<<256, 256, 0, g_stream>>>(p, block, nblocks, base, step, d);
    CUDA_CHECK(cudaMemcpyAsync(&h, d, sizeof(h), cudaMemcpyDeviceToHost, g_stream));
    sync_stream();
    return h;
}

/* ---- one operation of each benchmark ------------------------------------- */

/* count = floats per rank-buffer (see usage text for per-benchmark meaning). */
static void run_op(int b, float *sbuf, float *rbuf, size_t count, int window, int root) {
    int peer = 1 - g_rank;
    switch (b) {
    case LATENCY: /* ping-pong, rank 0 <-> rank 1 */
        if (g_rank == 0) {
            NCCL_CHECK(ncclSend(sbuf, count, ncclFloat, peer, g_comm, g_stream));
            sync_stream();
            NCCL_CHECK(ncclRecv(rbuf, count, ncclFloat, peer, g_comm, g_stream));
            sync_stream();
        } else {
            NCCL_CHECK(ncclRecv(rbuf, count, ncclFloat, peer, g_comm, g_stream));
            sync_stream();
            NCCL_CHECK(ncclSend(sbuf, count, ncclFloat, peer, g_comm, g_stream));
            sync_stream();
        }
        break;
    case BW: /* window of sends 0 -> 1, then a 1-float ack 1 -> 0 */
        NCCL_CHECK(ncclGroupStart());
        for (int w = 0; w < window; w++) {
            if (g_rank == 0)
                NCCL_CHECK(ncclSend(sbuf, count, ncclFloat, peer, g_comm, g_stream));
            else
                NCCL_CHECK(ncclRecv(rbuf, count, ncclFloat, peer, g_comm, g_stream));
        }
        NCCL_CHECK(ncclGroupEnd());
        sync_stream();
        if (g_rank == 0)
            NCCL_CHECK(ncclRecv(g_scratch, 1, ncclFloat, peer, g_comm, g_stream));
        else
            NCCL_CHECK(ncclSend(g_scratch, 1, ncclFloat, peer, g_comm, g_stream));
        sync_stream();
        break;
    case BIBW:
        NCCL_CHECK(ncclGroupStart());
        for (int w = 0; w < window; w++) {
            NCCL_CHECK(ncclSend(sbuf, count, ncclFloat, peer, g_comm, g_stream));
            NCCL_CHECK(ncclRecv(rbuf, count, ncclFloat, peer, g_comm, g_stream));
        }
        NCCL_CHECK(ncclGroupEnd());
        sync_stream();
        break;
    case ALLREDUCE:
        NCCL_CHECK(ncclAllReduce(sbuf, rbuf, count, ncclFloat, ncclSum, g_comm, g_stream));
        sync_stream();
        break;
    case ALLGATHER:
        NCCL_CHECK(ncclAllGather(sbuf, rbuf, count, ncclFloat, g_comm, g_stream));
        sync_stream();
        break;
    case BCAST:
        NCCL_CHECK(ncclBroadcast(sbuf, rbuf, count, ncclFloat, root, g_comm, g_stream));
        sync_stream();
        break;
    case REDUCE:
        NCCL_CHECK(ncclReduce(sbuf, rbuf, count, ncclFloat, ncclSum, root, g_comm, g_stream));
        sync_stream();
        break;
    case REDUCE_SCATTER:
        NCCL_CHECK(
            ncclReduceScatter(sbuf, rbuf, count / g_nranks, ncclFloat, ncclSum, g_comm, g_stream));
        sync_stream();
        break;
    case ALLTOALL: /* grouped send/recv, as NCCL's docs and OSU's xccl shim do */
        NCCL_CHECK(ncclGroupStart());
        for (int p = 0; p < g_nranks; p++) {
            NCCL_CHECK(ncclSend(sbuf + p * count, count, ncclFloat, p, g_comm, g_stream));
            NCCL_CHECK(ncclRecv(rbuf + p * count, count, ncclFloat, p, g_comm, g_stream));
        }
        NCCL_CHECK(ncclGroupEnd());
        sync_stream();
        break;
    }
}

/* Fill inputs so the expected output is exact in fp32, run once, check. */
static unsigned long long validate(int b, float *sbuf, float *rbuf, size_t count, int root) {
    int n = g_nranks, r = g_rank;
    float sum = (float)n * (n + 1) / 2; /* sum over ranks of (rank + 1) */
    unsigned long long bad = 0;
    switch (b) {
    case LATENCY:
    case BW:
    case BIBW:
        fill(sbuf, count, (float)(r + 1));
        fill(rbuf, count, -1.f);
        run_op(b, sbuf, rbuf, count, 1, root);
        if (!(b == BW && r == 0)) /* in bw only rank 1 receives the payload */
            bad = count_bad(rbuf, count, 1, (float)(2 - r), 0.f);
        break;
    case ALLREDUCE:
        fill(sbuf, count, (float)(r + 1));
        run_op(b, sbuf, rbuf, count, 1, root);
        bad = count_bad(rbuf, count, 1, sum, 0.f);
        break;
    case ALLGATHER:
        fill(sbuf, count, (float)(r + 1));
        run_op(b, sbuf, rbuf, count, 1, root);
        bad = count_bad(rbuf, count, n, 1.f, 1.f); /* block p == p + 1 */
        break;
    case BCAST:
        fill(sbuf, count, r == root ? 7.f : -1.f);
        fill(rbuf, count, -1.f);
        run_op(b, sbuf, rbuf, count, 1, root);
        bad = count_bad(rbuf, count, 1, 7.f, 0.f);
        break;
    case REDUCE:
        fill(sbuf, count, (float)(r + 1));
        fill(rbuf, count, -1.f);
        run_op(b, sbuf, rbuf, count, 1, root);
        if (r == root)
            bad = count_bad(rbuf, count, 1, sum, 0.f);
        break;
    case REDUCE_SCATTER:
        fill(sbuf, count, (float)(r + 1));
        run_op(b, sbuf, rbuf, count, 1, root);
        bad = count_bad(rbuf, count / n, 1, sum, 0.f);
        break;
    case ALLTOALL:
        /* block p sent by rank r = r*n + p; so block q received by r = q*n + r */
        fill_blocks(sbuf, count, n, (float)(r * n), 1.f);
        run_op(b, sbuf, rbuf, count, 1, root);
        bad = count_bad(rbuf, count, n, (float)r, (float)n);
        break;
    }
    return (unsigned long long)reduce_double((double)bad, ncclSum);
}

static void usage(void) {
    fprintf(stderr,
            "usage: nccl_bench -b BENCH [-m MIN:MAX] [-i ITERS] [-x WARMUP] [-w WINDOW]\n"
            "                  [-r ROOT] [-p EVERY] -f ID_FILE\n"
            "  BENCH: latency bw bibw (exactly 2 ranks) | allreduce allgather bcast\n"
            "         reduce reduce_scatter alltoall\n"
            "  sizes are bytes (multiples of 4, fp32 data):\n"
            "    allgather: per-rank contribution     alltoall: per-peer block\n"
            "    reduce_scatter: total input per rank (output = size / nranks)\n"
            "  -i ITERS   timed iterations per size (default 1000; 100 above 1 MiB)\n"
            "  -x WARMUP  untimed iterations per size (default 100; 10 above 1 MiB)\n"
            "  -w WINDOW  messages per bw/bibw iteration (default 64, as OSU)\n"
            "  -T SECONDS time budget per message size: the warm-up is timed, the per-\n"
            "             iteration time agreed across ranks (max), and the timed\n"
            "             iterations chosen to fill SECONDS (overrides -i; at least 10)\n"
            "  -p EVERY   print '# progress' every EVERY iterations (for checkpoint tests)\n"
            "  -f FILE    uniqueId exchange file on a filesystem all ranks see\n"
            "             (default $NCCL_BENCH_ID_FILE)\n");
    exit(2);
}

int main(int argc, char **argv) {
    int b = -1, iters = -1, warmup = -1, window = 64, root = 0, progress = 0, opt;
    double budget = 0;
    size_t min_size = 4, max_size = 1 << 22;
    const char *id_file = getenv("NCCL_BENCH_ID_FILE");
#ifdef WITH_MPI_INIT
    MPI_Init(&argc, &argv);
#endif

    while ((opt = getopt(argc, argv, "b:m:i:x:w:r:p:f:T:h")) != -1) {
        switch (opt) {
        case 'b':
            for (int k = 0; k < NBENCH; k++)
                if (strcmp(optarg, bench_names[k]) == 0)
                    b = k;
            break;
        case 'm':
            if (strchr(optarg, ':'))
                sscanf(optarg, "%zu:%zu", &min_size, &max_size);
            else
                max_size = strtoull(optarg, NULL, 10);
            break;
        case 'i':
            iters = atoi(optarg);
            break;
        case 'x':
            warmup = atoi(optarg);
            break;
        case 'w':
            window = atoi(optarg);
            break;
        case 'r':
            root = atoi(optarg);
            break;
        case 'p':
            progress = atoi(optarg);
            break;
        case 'f':
            id_file = optarg;
            break;
        case 'T':
            budget = atof(optarg);
            break;
        default:
            usage();
        }
    }
    if (b < 0 || !id_file || !*id_file)
        usage();
    if (min_size < 4)
        min_size = 4;

    g_rank = env_int("SLURM_PROCID", "RANK", 0);
    g_nranks = env_int("SLURM_NTASKS", "WORLD_SIZE", 1);
    g_local = env_int("SLURM_LOCALID", "LOCAL_RANK", 0);
    if ((b == LATENCY || b == BW || b == BIBW) && g_nranks != 2) {
        if (g_rank == 0)
            fprintf(stderr, "%s needs exactly 2 ranks (have %d)\n", bench_names[b], g_nranks);
        return 2;
    }
    if (root < 0 || root >= g_nranks)
        root = 0;

    int ndev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    CUDA_CHECK(cudaSetDevice(g_local % ndev));
    CUDA_CHECK(cudaStreamCreateWithFlags(&g_stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaMalloc(&g_scratch, 64));
    CUDA_CHECK(cudaMemset(g_scratch, 0, 64));

    ncclUniqueId id;
    bootstrap(id_file, &id);
    double t_init = now_s();
    NCCL_CHECK(ncclCommInitRank(&g_comm, g_nranks, id, g_rank));
    t_init = now_s() - t_init;

    /* Buffers sized for the largest message of this benchmark. */
    size_t maxc = max_size / 4;
    size_t sfl = maxc, rfl = maxc;
    if (b == ALLGATHER)
        rfl = maxc * g_nranks;
    if (b == ALLTOALL)
        sfl = rfl = maxc * g_nranks;
    float *sbuf, *rbuf;
    CUDA_CHECK(cudaMalloc(&sbuf, sfl * 4));
    CUDA_CHECK(cudaMalloc(&rbuf, rfl * 4));
    CUDA_CHECK(cudaMemset(sbuf, 0, sfl * 4));
    CUDA_CHECK(cudaMemset(rbuf, 0, rfl * 4));

    int pt2pt = (b == LATENCY || b == BW || b == BIBW);
    if (g_rank == 0) {
        int v;
        ncclGetVersion(&v);
        printf("# nccl_bench %s: %d ranks, NCCL %d, comm init %.3f s\n", bench_names[b], g_nranks,
               v, t_init);
        if (b == BW || b == BIBW)
            printf("%-10s%18s%12s\n", "# Size", "Bandwidth (MB/s)", "Iterations");
        else
            printf("%-10s%18s%18s%18s%12s\n", "# Size", "Avg Latency(us)", "Min Latency(us)",
                   "Max Latency(us)", "Iterations");
        fflush(stdout);
    }

    int failed = 0;
    long done = 0;
    for (size_t size = min_size; size <= max_size; size *= 2) {
        size_t count = size / 4;
        if (b == REDUCE_SCATTER && count < (size_t)g_nranks)
            continue;
        if (b == REDUCE_SCATTER)
            count -= count % g_nranks;
        int it = iters > 0 ? iters : (size > (1 << 20) ? 100 : 1000);
        int wu = warmup >= 0 ? warmup : (size > (1 << 20) ? 10 : 100);

        unsigned long long bad = validate(b, sbuf, rbuf, count, root);
        if (bad) {
            if (g_rank == 0)
                printf("# VALIDATION FAILED size %zu: %llu bad elements\n", size, bad);
            failed = 1;
        }

        barrier();
        double t = 0, t_wu = 0;
        if (budget > 0 && wu < 3)
            wu = 3; /* the first warm-up iteration (connection setup) is not timed */
        for (int i = 0; i < wu + it; i++) {
            double t0 = now_s();
            run_op(b, sbuf, rbuf, count, window, root);
            double dt = now_s() - t0;
            if (i >= wu)
                t += dt;
            else if (i > 0)
                t_wu += dt;
            if (!pt2pt)
                barrier(); /* untimed, as OSU's per-iteration MPI_Barrier */
            if (progress > 0 && ++done % progress == 0 && g_rank == 0) {
                printf("# progress %ld\n", done);
                fflush(stdout);
            }
            if (budget > 0 && i == wu - 1) {
                /* Every rank must pick the same count: agree on the slowest. */
                double per = reduce_double(t_wu / (wu - 1), ncclMax);
                long n = per > 0 ? (long)(budget / per) : 10;
                it = (int)(n < 10 ? 10 : (n > 100000000 ? 100000000 : n));
            }
        }

        if (b == LATENCY) {
            /* rank 0's round trips, halved; both ranks reduce so neither blocks */
            double lat = reduce_double(g_rank == 0 ? t : 0.0, ncclSum) * 1e6 / (2.0 * it);
            if (g_rank == 0)
                printf("%-10zu%18.2f%18.2f%18.2f%12d\n", size, lat, lat, lat, it);
        } else if (b == BW || b == BIBW) {
            double tt = reduce_double(t, ncclMax);
            double bytes = (double)size * window * it * (b == BIBW ? 2 : 1);
            if (g_rank == 0)
                printf("%-10zu%18.2f%12d\n", size, bytes / 1e6 / tt, it);
        } else {
            double lat = t * 1e6 / it;
            double avg = reduce_double(lat, ncclSum) / g_nranks;
            double mn = reduce_double(lat, ncclMin), mx = reduce_double(lat, ncclMax);
            if (g_rank == 0)
                printf("%-10zu%18.2f%18.2f%18.2f%12d\n", size, avg, mn, mx, it);
        }
        if (g_rank == 0)
            fflush(stdout);
    }

    if (g_rank == 0)
        printf("# nccl_bench %s: validation %s\n", bench_names[b], failed ? "FAILED" : "PASSED");
    CUDA_CHECK(cudaFree(sbuf));
    CUDA_CHECK(cudaFree(rbuf));
    NCCL_CHECK(ncclCommDestroy(g_comm));
    if (g_rank == 0)
        unlink(id_file);
#ifdef WITH_MPI_INIT
    MPI_Finalize();
#endif
    return failed;
}
