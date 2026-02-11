#include <cuda_runtime.h>
#include <nccl.h>
#include <stdio.h>

#define CUDA_CHECK(cmd)                                                        \
    do {                                                                       \
        cudaError_t e = cmd;                                                   \
        if (e != cudaSuccess) {                                                \
            printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__,               \
                   cudaGetErrorString(e));                                     \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

#define NCCL_CHECK(cmd)                                                        \
    do {                                                                       \
        ncclResult_t r = cmd;                                                  \
        if (r != ncclSuccess) {                                                \
            printf("NCCL error %s:%d: %s\n", __FILE__, __LINE__,               \
                   ncclGetErrorString(r));                                     \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

int main()
{
    int size = 32;
    int devId = 0;

    // Set device
    CUDA_CHECK(cudaSetDevice(devId));

    // Allocate device buffers
    float *sendbuf, *recvbuf;
    CUDA_CHECK(cudaMalloc(&sendbuf, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&recvbuf, size * sizeof(float)));

    // Initialize send buffer with some data
    float hostbuf[32];
    for (int i = 0; i < size; i++) hostbuf[i] = i + 1.0f;
    CUDA_CHECK(cudaMemcpy(sendbuf, hostbuf, size * sizeof(float),
                          cudaMemcpyHostToDevice));

    // Create NCCL communicator for single device
    ncclComm_t comm;
    NCCL_CHECK(ncclCommInitAll(&comm, 1, &devId));

    // Create CUDA stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Perform AllReduce (sum) - with single device, output equals input
    NCCL_CHECK(ncclAllReduce(sendbuf, recvbuf, size, ncclFloat, ncclSum, comm,
                             stream));

    // Wait for operation to complete
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Verify result
    float result[32];
    CUDA_CHECK(cudaMemcpy(result, recvbuf, size * sizeof(float),
                          cudaMemcpyDeviceToHost));

    printf("NCCL Hello World!\n");
    printf("Input:  [%.1f, %.1f, %.1f, ...]\n", hostbuf[0], hostbuf[1],
           hostbuf[2]);
    printf("Output: [%.1f, %.1f, %.1f, ...]\n", result[0], result[1],
           result[2]);

    // Cleanup
    CUDA_CHECK(cudaFree(sendbuf));
    CUDA_CHECK(cudaFree(recvbuf));
    NCCL_CHECK(ncclCommDestroy(comm));
    CUDA_CHECK(cudaStreamDestroy(stream));

    printf("Success!\n");
    return 0;
}
