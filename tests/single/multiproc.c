#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define CUDA_CHECK(cmd)                                                                            \
    do {                                                                                           \
        cudaError_t e = cmd;                                                                       \
        if (e != cudaSuccess) {                                                                    \
            printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e));           \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)

#define NCCL_CHECK(cmd)                                                                            \
    do {                                                                                           \
        ncclResult_t r = cmd;                                                                      \
        if (r != ncclSuccess) {                                                                    \
            printf("NCCL error %s:%d: %s\n", __FILE__, __LINE__, ncclGetErrorString(r));           \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)

#define MPI_CHECK(cmd)                                                                             \
    do {                                                                                           \
        int e = cmd;                                                                               \
        if (e != MPI_SUCCESS) {                                                                    \
            printf("MPI error %s:%d: %d\n", __FILE__, __LINE__, e);                                \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)

int main(int argc, char *argv[]) {
    int size = 32;
    int rank, nranks;

    MPI_CHECK(MPI_Init(&argc, &argv));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &nranks));

    // Stagger initialization
    sleep(rank * 3);
    printf("[Rank %d] Waking up, initializing...\n", rank);

    CUDA_CHECK(cudaSetDevice(0));
    printf("[Rank %d] Set device 0\n", rank);

    ncclUniqueId ncclId;
    if (rank == 0) {
        NCCL_CHECK(ncclGetUniqueId(&ncclId));
    }
    MPI_CHECK(MPI_Bcast(&ncclId, sizeof(ncclId), MPI_BYTE, 0, MPI_COMM_WORLD));

    printf("[Rank %d] Creating communicator...\n", rank);
    ncclComm_t comm;
    NCCL_CHECK(ncclCommInitRank(&comm, nranks, ncclId, rank));
    printf("[Rank %d] Communicator created!\n", rank);

    // Allocate and init
    float *sendbuf, *recvbuf;
    CUDA_CHECK(cudaMalloc((void **)&sendbuf, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&recvbuf, size * sizeof(float)));

    float *hostbuf = (float *)malloc(size * sizeof(float));
    for (int i = 0; i < size; i++)
        hostbuf[i] = rank + 1.0f;
    CUDA_CHECK(cudaMemcpy(sendbuf, hostbuf, size * sizeof(float), cudaMemcpyHostToDevice));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // AllReduce
    NCCL_CHECK(ncclAllReduce(sendbuf, recvbuf, size, ncclFloat, ncclSum, comm, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    float *result = (float *)malloc(size * sizeof(float));
    CUDA_CHECK(cudaMemcpy(result, recvbuf, size * sizeof(float), cudaMemcpyDeviceToHost));

    float expected = nranks * (nranks + 1) / 2.0f;
    printf("[Rank %d] Input: %.1f, Output: %.1f (expected: %.1f)\n", rank, hostbuf[0], result[0],
           expected);

    free(hostbuf);
    free(result);
    CUDA_CHECK(cudaFree(sendbuf));
    CUDA_CHECK(cudaFree(recvbuf));
    CUDA_CHECK(cudaStreamDestroy(stream));
    NCCL_CHECK(ncclCommDestroy(comm));

    MPI_CHECK(MPI_Finalize());
    if (rank == 0)
        printf("Success!\n");
    return 0;
}
