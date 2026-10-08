/*
 * MPI + NCCL program for checking that MANA (MPI checkpointing) and the NCCL
 * plugin work together, including NCCL's CUDA IPC transports.
 *
 * MPI is used the way real MPI+NCCL codes use it: to bootstrap NCCL
 * (MPI_Bcast of the ncclUniqueId) and for host-side work (MPI_Allreduce).
 * NCCL carries the device traffic: an allreduce and a ring of send/recv, so
 * with 2+ GPUs in a node NCCL uses its P2P (CUDA IPC) or SHM transport.
 *
 * Every iteration checks all results exactly, and a device-resident counter
 * proves GPU memory survived each checkpoint/restart.
 *
 *   srun -n 2 mana_launch --cuda --with-plugin libncclplugin.so ./mana_nccl_test [-n ITERS] [-s MS]
 * [-c]
 *
 * -c: CUDA only, no NCCL (device counter + MPI). The control that separates
 *     MANA + the CUDA plugin from anything the NCCL plugin does.
 */

#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

static int g_rank;

#define CUDA_CHECK(x)                                                                              \
    do {                                                                                           \
        cudaError_t e_ = (x);                                                                      \
        if (e_ != cudaSuccess) {                                                                   \
            fprintf(stderr, "[rank %d] FAIL CUDA %s at %s:%d\n", g_rank, cudaGetErrorString(e_),   \
                    __FILE__, __LINE__);                                                           \
            MPI_Abort(MPI_COMM_WORLD, 2);                                                          \
        }                                                                                          \
    } while (0)
#define NCCL_CHECK(x)                                                                              \
    do {                                                                                           \
        ncclResult_t r_ = (x);                                                                     \
        if (r_ != ncclSuccess) {                                                                   \
            fprintf(stderr, "[rank %d] FAIL NCCL %s at %s:%d\n", g_rank, ncclGetErrorString(r_),   \
                    __FILE__, __LINE__);                                                           \
            MPI_Abort(MPI_COMM_WORLD, 2);                                                          \
        }                                                                                          \
    } while (0)

#define N (1 << 20) /* floats per buffer: 4 MiB, above NCCL's small-message paths */

__global__ void fill(float *p, int n, float v) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        p[i] = v;
}
__global__ void bump(long *counter) {
    *counter += 1;
}
__global__ void count_bad(const float *p, int n, float v, int *bad) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && p[i] != v)
        atomicAdd(bad, 1);
}

static int check(const float *p, float v, int *d_bad, cudaStream_t s) {
    int bad;
    CUDA_CHECK(cudaMemsetAsync(d_bad, 0, sizeof(int), s));
    count_bad<<<(N + 255) / 256, 256, 0, s>>>(p, N, v, d_bad);
    CUDA_CHECK(cudaMemcpyAsync(&bad, d_bad, sizeof(int), cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
    return bad;
}

int main(int argc, char **argv) {
    int iters = 600, sleep_ms = 100, opt, size, ndev, use_nccl = 1;
    while ((opt = getopt(argc, argv, "n:s:c")) != -1) {
        if (opt == 'n')
            iters = atoi(optarg);
        else if (opt == 's')
            sleep_ms = atoi(optarg);
        else if (opt == 'c')
            use_nccl = 0;
    }

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const char *lr = getenv("SLURM_LOCALID");
    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    CUDA_CHECK(cudaSetDevice((lr ? atoi(lr) : g_rank) % ndev));

    ncclUniqueId id;
    ncclComm_t comm = NULL;
    if (use_nccl) {
        if (g_rank == 0)
            NCCL_CHECK(ncclGetUniqueId(&id));
        MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
        NCCL_CHECK(ncclCommInitRank(&comm, size, id, g_rank));
    }

    cudaStream_t s;
    float *a_in, *a_out, *p_send, *p_recv;
    long *d_counter, counter;
    int *d_bad;
    CUDA_CHECK(cudaStreamCreate(&s));
    CUDA_CHECK(cudaMalloc(&a_in, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&a_out, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&p_send, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&p_recv, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_counter, sizeof(long)));
    CUDA_CHECK(cudaMalloc(&d_bad, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_counter, 0, sizeof(long)));

    int next = (g_rank + 1) % size, prev = (g_rank + size - 1) % size;
    if (g_rank == 0)
        printf("[mana-nccl] %d ranks, %d iterations, %d ms sleep\n", size, iters, sleep_ms);
    fflush(stdout);

    for (int it = 0; it < iters; it++) {
        /* NCCL allreduce: sum over ranks of (rank + 1 + it). Exact in fp32. */
        float mine = (float)(g_rank + 1 + it % 1000);
        float want = (float)size * (size + 1) / 2 + (float)size * (it % 1000);
        fill<<<(N + 255) / 256, 256, 0, s>>>(a_in, N, mine);
        fill<<<(N + 255) / 256, 256, 0, s>>>(p_send, N, (float)(g_rank * 1000 + it % 1000));
        if (use_nccl) {
            NCCL_CHECK(ncclAllReduce(a_in, a_out, N, ncclFloat, ncclSum, comm, s));
            /* NCCL ring: send to next, receive from prev (P2P / IPC within a node). */
            NCCL_CHECK(ncclGroupStart());
            NCCL_CHECK(ncclSend(p_send, N, ncclFloat, next, comm, s));
            NCCL_CHECK(ncclRecv(p_recv, N, ncclFloat, prev, comm, s));
            NCCL_CHECK(ncclGroupEnd());
        }

        bump<<<1, 1, 0, s>>>(d_counter);
        CUDA_CHECK(cudaStreamSynchronize(s));

        int bad_ar = use_nccl ? check(a_out, want, d_bad, s) : 0;
        int bad_p2p = use_nccl ? check(p_recv, (float)(prev * 1000 + it % 1000), d_bad, s) : 0;
        CUDA_CHECK(cudaMemcpy(&counter, d_counter, sizeof(long), cudaMemcpyDeviceToHost));

        /* MPI on host data, through MANA's lower half. */
        long hsum = 0, hmine = g_rank + it;
        MPI_Allreduce(&hmine, &hsum, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
        long hwant = (long)size * (size - 1) / 2 + (long)size * it;

        if (bad_ar || bad_p2p || counter != it + 1 || hsum != hwant) {
            fprintf(stderr,
                    "[mana-nccl] FAIL rank %d iter %d: allreduce bad=%d p2p bad=%d "
                    "counter=%ld (want %d) mpi=%ld (want %ld)\n",
                    g_rank, it, bad_ar, bad_p2p, counter, it + 1, hsum, hwant);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (g_rank == 0 && it % 10 == 0) {
            printf("[mana-nccl] iter %d ok\n", it);
            fflush(stdout);
        }
        usleep(sleep_ms * 1000);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    printf("[mana-nccl] PASS rank %d, %d iterations\n", g_rank, iters);
    fflush(stdout);
    if (use_nccl)
        NCCL_CHECK(ncclCommDestroy(comm));
    MPI_Finalize();
    return 0;
}
