/*
 * cc_complex.c
 *
 * 3 MPI processes on 3 GPUs.
 * Tests the CC algorithm with multiple communicators where rank 1
 * is a member of two groups:
 *
 *   comm_world: ranks 0, 1, 2
 *   comm_A:     ranks 0, 1     (runs 2 AllReduces per outer iteration)
 *   comm_B:     ranks 1, 2     (runs 1 AllReduce per outer iteration)
 *
 * This creates a scenario where rank 1 has different seq numbers
 * across groups, requiring target propagation to checkpoint safely.
 *
 * Build & Run:
 *   Run `make cc_complex_check` in parent directory
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <assert.h>
#include <mpi.h>
#include <nccl.h>
#include <cuda_runtime.h>

#define NELEMS 1024
#define NITERS 100

#define CUDACHECK(cmd) do {                                 \
    cudaError_t e = cmd;                                    \
    if (e != cudaSuccess) {                                 \
        fprintf(stderr, "CUDA error %d: %s (%s:%d)\n",     \
                e, cudaGetErrorString(e),__FILE__,__LINE__);\
        exit(1);                                            \
    }                                                       \
} while(0)

#define NCCLCHECK(cmd) do {                                 \
    ncclResult_t r = cmd;                                   \
    if (r != ncclSuccess) {                                 \
        fprintf(stderr, "NCCL error %d: %s (%s:%d)\n",     \
                r, ncclGetErrorString(r),__FILE__,__LINE__);\
        exit(1);                                            \
    }                                                       \
} while(0)

/* Create a sub-communicator for the given subset of ranks */
static ncclComm_t init_sub_comm(int my_rank, int *members, int nmembers,
                                 MPI_Comm mpi_world) {
    int in_group = 0;
    for (int i = 0; i < nmembers; i++)
        if (members[i] == my_rank) { in_group = 1; break; }

    MPI_Comm mpi_sub;
    MPI_Comm_split(mpi_world, in_group ? 0 : MPI_UNDEFINED, my_rank, &mpi_sub);

    if (!in_group) return NULL;

    ncclUniqueId id;
    int sub_rank;
    MPI_Comm_rank(mpi_sub, &sub_rank);
    if (sub_rank == 0) NCCLCHECK(ncclGetUniqueId(&id));
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, mpi_sub);

    ncclComm_t comm;
    NCCLCHECK(ncclCommInitRank(&comm, nmembers, id, sub_rank));
    MPI_Comm_free(&mpi_sub);
    return comm;
}

static void allreduce_and_verify(float *d_buf, float *h_buf, int nelems,
                                 float fill_val, float expected,
                                 ncclComm_t comm, cudaStream_t stream,
                                 int rank, int iter, const char *comm_name) {
    for (int j = 0; j < nelems; j++) h_buf[j] = fill_val;
    CUDACHECK(cudaMemcpy(d_buf, h_buf, nelems * sizeof(float),
                         cudaMemcpyHostToDevice));
    NCCLCHECK(ncclAllReduce(d_buf, d_buf, nelems, ncclFloat, ncclSum,
                            comm, stream));
    CUDACHECK(cudaStreamSynchronize(stream));
    CUDACHECK(cudaMemcpy(h_buf, d_buf, nelems * sizeof(float),
                         cudaMemcpyDeviceToHost));
    for (int j = 0; j < nelems; j++) {
        if (h_buf[j] != expected) {
            fprintf(stderr, "[rank %d] MISMATCH %s iter=%d elem=%d: "
                    "%.0f != %.0f\n", rank, comm_name, iter, j,
                    h_buf[j], expected);
            exit(1);
        }
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);

    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    assert(nranks == 3);

    CUDACHECK(cudaSetDevice(rank));

    /* init comm_world (all 3 ranks) */
    ncclUniqueId id_world;
    if (rank == 0) NCCLCHECK(ncclGetUniqueId(&id_world));
    MPI_Bcast(&id_world, sizeof(id_world), MPI_BYTE, 0, MPI_COMM_WORLD);
    ncclComm_t comm_world;
    NCCLCHECK(ncclCommInitRank(&comm_world, 3, id_world, rank));

    /* init comm_A (ranks 0 and 1) and comm_B (ranks 1 and 2) */
    int members_A[] = {0, 1};
    int members_B[] = {1, 2};
    ncclComm_t comm_A = init_sub_comm(rank, members_A, 2, MPI_COMM_WORLD);
    ncclComm_t comm_B = init_sub_comm(rank, members_B, 2, MPI_COMM_WORLD);

    cudaStream_t stream;
    CUDACHECK(cudaStreamCreate(&stream));

    float *d_buf;
    CUDACHECK(cudaMalloc((void**)&d_buf, NELEMS * sizeof(float)));
    float *h_buf = (float *)malloc(NELEMS * sizeof(float));

    /* expected sums for each communicator */
    float expected_world = 0.0f + 1.0f + 2.0f;  /* = 3 */
    float expected_A     = 0.0f + 1.0f;          /* = 1 */
    float expected_B     = 1.0f + 2.0f;          /* = 3 */

    printf("[rank %d] Starting %d iterations. "
           "CHECKPOINT NOW with: dmtcp_command -c\n", rank, NITERS);

    for (int i = 0; i < NITERS; i++) {

        /* all ranks: AllReduce on comm_world */
        allreduce_and_verify(d_buf, h_buf, NELEMS, (float)rank,
                             expected_world, comm_world, stream,
                             rank, i, "comm_world");

        /* ranks 0 and 1: 2 AllReduces on comm_A (faster than comm_B) */
        if (comm_A != NULL) {
            allreduce_and_verify(d_buf, h_buf, NELEMS, (float)rank,
                                 expected_A, comm_A, stream,
                                 rank, i, "comm_A iter 1");
            allreduce_and_verify(d_buf, h_buf, NELEMS, (float)rank,
                                 expected_A, comm_A, stream,
                                 rank, i, "comm_A iter 2");
        }

        /* ranks 1 and 2: 1 AllReduce on comm_B (slower than comm_A) */
        if (comm_B != NULL) {
            allreduce_and_verify(d_buf, h_buf, NELEMS, (float)rank,
                                 expected_B, comm_B, stream,
                                 rank, i, "comm_B");
        }

        if (i == 10)
            printf("[rank %d] iter 10 — CHECKPOINT NOW: dmtcp_command -c\n",
                   rank);
    }

    printf("[rank %d] PASS\n", rank);

    free(h_buf);
    CUDACHECK(cudaFree(d_buf));
    CUDACHECK(cudaStreamDestroy(stream));
    NCCLCHECK(ncclCommDestroy(comm_world));
    if (comm_A != NULL) NCCLCHECK(ncclCommDestroy(comm_A));
    if (comm_B != NULL) NCCLCHECK(ncclCommDestroy(comm_B));
    MPI_Finalize();
    return 0;
}
