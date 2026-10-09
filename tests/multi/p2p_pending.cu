/*
 * p2p_pending.cu -- a checkpoint that lands while a point-to-point transfer
 * has only one side posted.
 *
 *   srun -n N p2p_pending SCENARIO ID_FILE [WINDOW_S]
 *
 * Ranks come from Slurm (SLURM_PROCID, SLURM_NTASKS, SLURM_LOCALID), and the
 * ncclUniqueId goes through ID_FILE on a filesystem every rank sees, so no MPI
 * is needed. tests/report/run_bench.sh drives it (benchmarks p2p_recv_first,
 * p2p_send_first, p2p_chain and p2p_relay; see tests/multi/run_p2p_pending.sh).
 *
 * Every rank prints "[p2p] rank R in window" once its side is set up. The
 * driver checkpoints right after that, while rank 0 (rank 1 for relay) still
 * sleeps for WINDOW_S seconds (default 20) before its own operation. So the
 * checkpoint always finds an operation whose counterpart is not posted yet.
 *
 * NCCL connects ranks at their first send or receive, and at a communicator's
 * first collective, and that call blocks until the peers join. So every
 * scenario except recv-first-cold first exchanges one message with both ring
 * neighbours and runs one allreduce: afterwards ncclSend, ncclRecv and
 * ncclAllReduce only enqueue work and return, and the pending operation sits
 * on the GPU with no thread inside NCCL. (Without the allreduce, multi hangs
 * at 2x4 even with no checkpoint: rank 0's first allreduce waits for rank 1,
 * which waits for rank 0's send.)
 *
 * Scenarios:
 *   recv-first  Rank 1 posts ncclRecv from rank 0; rank 0 sends after the
 *               window. The checkpoint finds a receive without its send.
 *   recv-first-cold
 *               recv-first without the warm-up: rank 1's thread stays inside
 *               ncclRecv, connecting to rank 0, until rank 0 sends.
 *   send-first  Rank 0 posts ncclSend to rank 1; rank 1 receives after the
 *               window. The checkpoint finds a send without its receive.
 *   chain       Rank 0 sends to rank 1 after the window. Every rank i > 0
 *               receives from i-1, waits for the data (cudaStreamSynchronize),
 *               checks it and forwards it to i+1. The checkpoint finds a
 *               receive on every rank > 0, and rank 1 blocked in CUDA until
 *               rank 0 sends. Needs 3 or more ranks.
 *   relay       Rank 0 posts ncclRecv from rank 1. Rank 1 sleeps through the
 *               window, then receives from rank 2 and waits for it, then sends
 *               to rank 0. Rank 2 sends to rank 1 a quarter of the way into the
 *               window, which is after the checkpoint has started: its send
 *               is only needed once rank 1 has posted the receive, so the
 *               checkpoint has to learn about that receive while it is
 *               already waiting for rank 1. Needs 3 or more ranks.
 *   multi       Two communicators: the world and a split of it in which
 *               ranks 0 and 1 form one group. Rank 1 posts ncclRecv from rank
 *               0 on the split; the other ranks post an ncclAllReduce on the
 *               world. Rank 0 sleeps through the window, then joins the
 *               allreduce and sends on the split, on another stream. So the
 *               checkpoint must let rank 0 catch up on a collective of one
 *               communicator and a send of the other, and rank 1, blocked in
 *               CUDA until that send arrives, catch up on the allreduce after.
 *               Needs 4 or more ranks.
 *
 * Every received buffer must hold its sender's values (element i of a message
 * from rank s is 1000 * (s + 1) + i). Then an ncclAllReduce checks that the
 * communicator (rebuilt, after a checkpoint) still works. Each rank prints
 * "PASS p2p_pending: rank R" on success and exits 1 on a mismatch.
 */

#include <cuda_runtime.h>
#include <errno.h>
#include <nccl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

#define NELEMS (1 << 20) /* 4 MiB per message */

#define CUDA_CHECK(cmd)                                                                            \
    do {                                                                                           \
        cudaError_t e_ = (cmd);                                                                    \
        if (e_ != cudaSuccess) {                                                                   \
            fprintf(stderr, "[p2p] rank %d: CUDA error %s at %s:%d\n", g_rank,                     \
                    cudaGetErrorString(e_), __FILE__, __LINE__);                                   \
            exit(2);                                                                               \
        }                                                                                          \
    } while (0)

#define NCCL_CHECK(cmd)                                                                            \
    do {                                                                                           \
        ncclResult_t r_ = (cmd);                                                                   \
        if (r_ != ncclSuccess) {                                                                   \
            fprintf(stderr, "[p2p] rank %d: NCCL error %s at %s:%d\n", g_rank,                     \
                    ncclGetErrorString(r_), __FILE__, __LINE__);                                   \
            exit(2);                                                                               \
        }                                                                                          \
    } while (0)

static int g_rank, g_nranks;

static int env_int(const char *a, const char *b, int dflt) {
    const char *s = getenv(a);
    if (s == NULL || *s == '\0')
        s = getenv(b);
    return (s != NULL && *s != '\0') ? atoi(s) : dflt;
}

static double now_s(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

/* ---- bootstrap: the ncclUniqueId through a shared file (as nccl_bench) ---- */

/* The tag names the Slurm job and step, so a file left by an earlier step at
 * the same path is never taken for this step's id. */
typedef struct {
    char magic[8];
    char tag[64];
    ncclUniqueId id;
} id_file_t;

static void bootstrap(const char *path, ncclUniqueId *id) {
    id_file_t rec;
    char tag[64];
    snprintf(tag, sizeof(tag), "%s.%s.%d", getenv("SLURM_JOB_ID") ? getenv("SLURM_JOB_ID") : "-",
             getenv("SLURM_STEP_ID") ? getenv("SLURM_STEP_ID") : "-", g_nranks);
    if (g_rank == 0) {
        char tmp[4096];
        FILE *f;
        memset(&rec, 0, sizeof(rec));
        memcpy(rec.magic, "P2PPEND", 8);
        snprintf(rec.tag, sizeof(rec.tag), "%s", tag);
        NCCL_CHECK(ncclGetUniqueId(&rec.id));
        snprintf(tmp, sizeof(tmp), "%s.tmp.%d", path, (int)getpid());
        f = fopen(tmp, "wb");
        if (f == NULL || fwrite(&rec, sizeof(rec), 1, f) != 1 || fclose(f) != 0 ||
            rename(tmp, path) != 0) { /* rename: readers never see a partial file */
            fprintf(stderr, "[p2p] cannot write %s: %s\n", path, strerror(errno));
            exit(2);
        }
        *id = rec.id;
        return;
    }
    double t0 = now_s();
    for (;;) {
        FILE *f = fopen(path, "rb");
        if (f != NULL) {
            size_t got = fread(&rec, sizeof(rec), 1, f);
            fclose(f);
            if (got == 1 && memcmp(rec.magic, "P2PPEND", 8) == 0 && strcmp(rec.tag, tag) == 0) {
                *id = rec.id;
                return;
            }
        }
        if (now_s() - t0 > 600) {
            fprintf(stderr, "[p2p] rank %d: timed out waiting for %s\n", g_rank, path);
            exit(2);
        }
        usleep(50000);
    }
}

/* ---- payload ----------------------------------------------------------------- */

static float *g_host;

static void fill(float *dev, int src) {
    for (int i = 0; i < NELEMS; i++)
        g_host[i] = (float)(1000 * (src + 1) + (i & 0xffff));
    CUDA_CHECK(cudaMemcpy(dev, g_host, NELEMS * sizeof(float), cudaMemcpyHostToDevice));
}

/* Returns 1 if dev holds the message rank src sent. */
static int check(const float *dev, int src, const char *what) {
    CUDA_CHECK(cudaMemcpy(g_host, dev, NELEMS * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < NELEMS; i++) {
        float want = (float)(1000 * (src + 1) + (i & 0xffff));
        if (g_host[i] != want) {
            printf("[p2p] rank %d: FAIL %s from rank %d: element %d is %g, want %g\n", g_rank, what,
                   src, i, g_host[i], want);
            return 0;
        }
    }
    printf("[p2p] rank %d: %s from rank %d ok\n", g_rank, what, src);
    return 1;
}

static void in_window(void) { printf("[p2p] rank %d in window\n", g_rank); }

/* Connect every rank with both ring neighbours, in both directions, with
 * messages of the size the scenarios use. */
static void warm_up(ncclComm_t comm, int rank, int nranks, cudaStream_t s, float *buf,
                    float *tmp) {
    int next = (rank + 1) % nranks, prev = (rank + nranks - 1) % nranks;
    NCCL_CHECK(ncclGroupStart());
    NCCL_CHECK(ncclSend(buf, NELEMS, ncclFloat, next, comm, s));
    NCCL_CHECK(ncclRecv(tmp, NELEMS, ncclFloat, prev, comm, s));
    NCCL_CHECK(ncclGroupEnd());
    NCCL_CHECK(ncclGroupStart());
    NCCL_CHECK(ncclSend(buf, NELEMS, ncclFloat, prev, comm, s));
    NCCL_CHECK(ncclRecv(tmp, NELEMS, ncclFloat, next, comm, s));
    NCCL_CHECK(ncclGroupEnd());
    CUDA_CHECK(cudaStreamSynchronize(s));
}

/* Sum of rank + 1 over the communicator; returns 1 if the result is right. */
static int allreduce_check(ncclComm_t comm, int rank, int nranks, cudaStream_t s, float *red,
                           const char *what) {
    float mine = (float)(rank + 1), sum = 0;
    float want = (float)nranks * (nranks + 1) / 2;
    CUDA_CHECK(cudaMemcpy(red, &mine, sizeof(float), cudaMemcpyHostToDevice));
    NCCL_CHECK(ncclAllReduce(red, red, 1, ncclFloat, ncclSum, comm, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
    CUDA_CHECK(cudaMemcpy(&sum, red, sizeof(float), cudaMemcpyDeviceToHost));
    if (sum != want) {
        printf("[p2p] rank %d: FAIL %s gave %g, want %g\n", g_rank, what, sum, want);
        return 0;
    }
    return 1;
}

int main(int argc, char **argv) {
    const char *scenario, *id_path;
    int window, ok = 1, ndev = 0;
    ncclUniqueId id;
    ncclComm_t comm, sub = NULL;
    int sub_rank = 0, sub_n = 1;
    cudaStream_t s, s2 = NULL;
    float *buf, *red, *red2 = NULL;

    setvbuf(stdout, NULL, _IOLBF, 0);
    if (argc < 3) {
        fprintf(stderr,
                "usage: %s recv-first|recv-first-cold|send-first|chain|relay|multi ID_FILE"
                " [WINDOW_S]\n",
                argv[0]);
        return 2;
    }
    scenario = argv[1];
    id_path = argv[2];
    window = argc > 3 ? atoi(argv[3]) : 20;

    g_rank = env_int("RANK", "SLURM_PROCID", 0);
    g_nranks = env_int("WORLD_SIZE", "SLURM_NTASKS", 1);
    {
        int need = !strcmp(scenario, "multi")                                     ? 4
                   : (!strcmp(scenario, "chain") || !strcmp(scenario, "relay")) ? 3
                                                                                : 2;
        if (g_nranks < need) {
            fprintf(stderr, "[p2p] %s needs %d or more ranks\n", scenario, need);
            return 2;
        }
    }
    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    CUDA_CHECK(cudaSetDevice(env_int("LOCAL_RANK", "SLURM_LOCALID", 0) % ndev));

    bootstrap(id_path, &id);
    NCCL_CHECK(ncclCommInitRank(&comm, g_nranks, id, g_rank));
    if (g_rank == 0)
        printf("# p2p_pending %s: %d ranks, window %d s\n", scenario, g_nranks, window);
    CUDA_CHECK(cudaStreamCreate(&s));
    CUDA_CHECK(cudaMalloc((void **)&buf, NELEMS * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&red, sizeof(float)));
    g_host = (float *)malloc(NELEMS * sizeof(float));

    if (strcmp(scenario, "recv-first-cold") != 0) {
        float *tmp;
        CUDA_CHECK(cudaMalloc((void **)&tmp, NELEMS * sizeof(float)));
        warm_up(comm, g_rank, g_nranks, s, buf, tmp);
        if (!allreduce_check(comm, g_rank, g_nranks, s, red, "warm-up allreduce"))
            return 1;
        if (!strcmp(scenario, "multi")) {
            NCCL_CHECK(ncclCommSplit(comm, g_rank < 2 ? 0 : 1, g_rank, &sub, NULL));
            NCCL_CHECK(ncclCommUserRank(sub, &sub_rank));
            NCCL_CHECK(ncclCommCount(sub, &sub_n));
            CUDA_CHECK(cudaStreamCreate(&s2));
            CUDA_CHECK(cudaMalloc((void **)&red2, sizeof(float)));
            warm_up(sub, sub_rank, sub_n, s2, buf, tmp);
            if (!allreduce_check(sub, sub_rank, sub_n, s2, red, "warm-up split allreduce"))
                return 1;
        }
        CUDA_CHECK(cudaFree(tmp));
    }

    if (!strcmp(scenario, "recv-first-cold")) {
        if (g_rank == 1) {
            in_window(); /* before the call: it blocks until rank 0 joins */
            NCCL_CHECK(ncclRecv(buf, NELEMS, ncclFloat, 0, comm, s));
            CUDA_CHECK(cudaStreamSynchronize(s));
            ok = check(buf, 0, "recv");
        } else {
            in_window();
            if (g_rank == 0) {
                fill(buf, 0);
                sleep(window);
                NCCL_CHECK(ncclSend(buf, NELEMS, ncclFloat, 1, comm, s));
                CUDA_CHECK(cudaStreamSynchronize(s));
            }
        }
    } else if (!strcmp(scenario, "recv-first")) {
        if (g_rank == 1) {
            NCCL_CHECK(ncclRecv(buf, NELEMS, ncclFloat, 0, comm, s));
            in_window();
            CUDA_CHECK(cudaStreamSynchronize(s));
            ok = check(buf, 0, "recv");
        } else {
            in_window();
            if (g_rank == 0) {
                fill(buf, 0);
                sleep(window);
                NCCL_CHECK(ncclSend(buf, NELEMS, ncclFloat, 1, comm, s));
                CUDA_CHECK(cudaStreamSynchronize(s));
            }
        }
    } else if (!strcmp(scenario, "send-first")) {
        if (g_rank == 0) {
            fill(buf, 0);
            NCCL_CHECK(ncclSend(buf, NELEMS, ncclFloat, 1, comm, s));
            in_window();
            CUDA_CHECK(cudaStreamSynchronize(s));
        } else {
            in_window();
            if (g_rank == 1) {
                sleep(window);
                NCCL_CHECK(ncclRecv(buf, NELEMS, ncclFloat, 0, comm, s));
                CUDA_CHECK(cudaStreamSynchronize(s));
                ok = check(buf, 0, "recv");
            }
        }
    } else if (!strcmp(scenario, "chain")) {
        if (g_rank == 0) {
            fill(buf, 0);
            in_window();
            sleep(window);
            NCCL_CHECK(ncclSend(buf, NELEMS, ncclFloat, 1, comm, s));
            CUDA_CHECK(cudaStreamSynchronize(s));
        } else {
            NCCL_CHECK(ncclRecv(buf, NELEMS, ncclFloat, g_rank - 1, comm, s));
            in_window();
            CUDA_CHECK(cudaStreamSynchronize(s));
            /* Every hop forwards rank 0's message unchanged. */
            ok = check(buf, 0, "chain recv");
            if (g_rank + 1 < g_nranks) {
                NCCL_CHECK(ncclSend(buf, NELEMS, ncclFloat, g_rank + 1, comm, s));
                CUDA_CHECK(cudaStreamSynchronize(s));
            }
        }
    } else if (!strcmp(scenario, "relay")) {
        if (g_rank == 0) {
            NCCL_CHECK(ncclRecv(buf, NELEMS, ncclFloat, 1, comm, s));
            in_window();
            CUDA_CHECK(cudaStreamSynchronize(s));
            ok = check(buf, 1, "relay recv");
        } else if (g_rank == 1) {
            float *mid;
            CUDA_CHECK(cudaMalloc((void **)&mid, NELEMS * sizeof(float)));
            in_window();
            sleep(window);
            NCCL_CHECK(ncclRecv(mid, NELEMS, ncclFloat, 2, comm, s));
            CUDA_CHECK(cudaStreamSynchronize(s));
            ok = check(mid, 2, "relay recv");
            CUDA_CHECK(cudaFree(mid));
            fill(buf, 1);
            NCCL_CHECK(ncclSend(buf, NELEMS, ncclFloat, 0, comm, s));
            CUDA_CHECK(cudaStreamSynchronize(s));
        } else {
            in_window();
            if (g_rank == 2) {
                fill(buf, 2);
                sleep(window > 4 ? window / 4 : 1);
                NCCL_CHECK(ncclSend(buf, NELEMS, ncclFloat, 1, comm, s));
                CUDA_CHECK(cudaStreamSynchronize(s));
            }
        }
    } else if (!strcmp(scenario, "multi")) {
        /* Sub-communicator ranks: rank 0 and rank 1 are 0 and 1 in theirs. */
        if (g_rank == 0) {
            fill(buf, 0);
            in_window();
            sleep(window);
            /* Separate streams: the send must not queue behind the allreduce,
             * which cannot finish before rank 1 has received. */
            NCCL_CHECK(ncclAllReduce(red2, red2, 1, ncclFloat, ncclSum, comm, s));
            NCCL_CHECK(ncclSend(buf, NELEMS, ncclFloat, 1, sub, s2));
            CUDA_CHECK(cudaStreamSynchronize(s2));
            CUDA_CHECK(cudaStreamSynchronize(s));
        } else if (g_rank == 1) {
            NCCL_CHECK(ncclRecv(buf, NELEMS, ncclFloat, 0, sub, s2));
            in_window();
            CUDA_CHECK(cudaStreamSynchronize(s2));
            ok = check(buf, 0, "multi recv");
            NCCL_CHECK(ncclAllReduce(red2, red2, 1, ncclFloat, ncclSum, comm, s));
            CUDA_CHECK(cudaStreamSynchronize(s));
        } else {
            in_window();
            NCCL_CHECK(ncclAllReduce(red2, red2, 1, ncclFloat, ncclSum, comm, s));
            CUDA_CHECK(cudaStreamSynchronize(s));
        }
    } else {
        fprintf(stderr, "[p2p] unknown scenario %s\n", scenario);
        return 2;
    }

    /* The communicator must still work: after a checkpoint it is a rebuilt one.
     * The second allreduce keeps every rank in the job until all have finished
     * the scenario: a sender whose message NCCL buffered eagerly runs ahead, and
     * it must not exit while the checkpoint is still waiting for its peer (the
     * gate holds it at its next NCCL call, but not at ncclCommDestroy). */
    if (sub != NULL && !allreduce_check(sub, sub_rank, sub_n, s2, red, "split allreduce"))
        ok = 0;
    for (int pass = 0; pass < 2; pass++)
        if (!allreduce_check(comm, g_rank, g_nranks, s, red, "allreduce"))
            ok = 0;

    CUDA_CHECK(cudaFree(buf));
    CUDA_CHECK(cudaFree(red));
    CUDA_CHECK(cudaStreamDestroy(s));
    if (sub != NULL) {
        CUDA_CHECK(cudaFree(red2));
        CUDA_CHECK(cudaStreamDestroy(s2));
        NCCL_CHECK(ncclCommDestroy(sub));
    }
    NCCL_CHECK(ncclCommDestroy(comm));
    free(g_host);
    if (!ok)
        return 1;
    printf("PASS p2p_pending: rank %d\n", g_rank);
    return 0;
}
