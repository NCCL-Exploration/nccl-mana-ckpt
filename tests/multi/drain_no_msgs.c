/*
 * drain_no_msgs.c
 *
 * 2 MPI processes on 2 GPUs.
 * Both ranks initialize NCCL, post no NCCL send/recv/collective work, then
 * sleep through the checkpoint window. This proves drain can handle a
 * zero-message communicator without hanging in consensus.
 *
 * Build & Run:
 *   Run `make drain_no_msgs_check` in parent directory
 */

#include <assert.h>
#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define CUDACHECK(cmd)                                                                             \
    do {                                                                                           \
        cudaError_t e = cmd;                                                                       \
        if (e != cudaSuccess) {                                                                    \
            fprintf(stderr, "CUDA error %d: %s (%s:%d)\n", e, cudaGetErrorString(e), __FILE__,     \
                    __LINE__);                                                                     \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)

#define NCCLCHECK(cmd)                                                                             \
    do {                                                                                           \
        ncclResult_t r = cmd;                                                                      \
        if (r != ncclSuccess) {                                                                    \
            fprintf(stderr, "NCCL error %d: %s (%s:%d)\n", r, ncclGetErrorString(r), __FILE__,     \
                    __LINE__);                                                                     \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);

    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    assert(nranks == 2);

    CUDACHECK(cudaSetDevice(rank));

    ncclUniqueId id;
    if (rank == 0)
        NCCLCHECK(ncclGetUniqueId(&id));
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

    ncclComm_t comm;
    NCCLCHECK(ncclCommInitRank(&comm, 2, id, rank));

    cudaStream_t stream;
    CUDACHECK(cudaStreamCreate(&stream));

    printf("[rank %d] NCCL initialized; posting no messages. [CHECKPOINT WINDOW]\n", rank);
    sleep(12);

    MPI_Barrier(MPI_COMM_WORLD);
    printf("[rank %d] PASS - drain completed with no messages\n", rank);

    CUDACHECK(cudaStreamDestroy(stream));
    NCCLCHECK(ncclCommDestroy(comm));
    MPI_Finalize();
    return 0;
}
