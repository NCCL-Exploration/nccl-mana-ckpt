/**
 * send_recv_test.c — Single-rank NCCL self send/recv.
 *
 * Rank 0 sends to itself and receives. Grouped to avoid deadlock.
 * Compile: nvcc -o send_recv_test send_recv_test.c -lnccl
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <nccl.h>

#define CUDA_CHECK(cmd) do { \
    cudaError_t e = cmd; \
    if (e != cudaSuccess) { \
        printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(1); \
    } \
} while(0)

#define NCCL_CHECK(cmd) do { \
    ncclResult_t r = cmd; \
    if (r != ncclSuccess) { \
        printf("NCCL error %s:%d: %s\n", __FILE__, __LINE__, ncclGetErrorString(r)); \
        exit(1); \
    } \
} while(0)

#define COUNT 1024

int main(void)
{
    CUDA_CHECK(cudaSetDevice(0));

    int rank = 0;
    ncclComm_t comm;
    NCCL_CHECK(ncclCommInitAll(&comm, 1, &rank));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    float *send_buf, *recv_buf;
    CUDA_CHECK(cudaMalloc((void **)&send_buf, COUNT * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&recv_buf, COUNT * sizeof(float)));

    float *h_send = (float *)malloc(COUNT * sizeof(float));
    for (int i = 0; i < COUNT; i++) h_send[i] = (float)i;
    CUDA_CHECK(cudaMemcpy(send_buf, h_send, COUNT * sizeof(float), cudaMemcpyHostToDevice));

    NCCL_CHECK(ncclGroupStart());
    NCCL_CHECK(ncclSend(send_buf, COUNT, ncclFloat, 0, comm, stream));
    NCCL_CHECK(ncclRecv(recv_buf, COUNT, ncclFloat, 0, comm, stream));
    NCCL_CHECK(ncclGroupEnd());

    CUDA_CHECK(cudaStreamSynchronize(stream));

    float *h_recv = (float *)malloc(COUNT * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_recv, recv_buf, COUNT * sizeof(float), cudaMemcpyDeviceToHost));

    int errors = 0;
    for (int i = 0; i < COUNT; i++) {
        if (h_recv[i] != h_send[i]) {
            if (errors < 5)
                printf("Error at %d: expected %f, got %f\n", i, h_send[i], h_recv[i]);
            errors++;
        }
    }

    if (errors == 0)
        printf("[TEST] SUCCESS: First: %.0f, Last: %.0f\n", h_recv[0], h_recv[COUNT - 1]);
    else
        printf("[TEST] FAILURE: %d errors found.\n", errors);

    CUDA_CHECK(cudaFree(send_buf));
    CUDA_CHECK(cudaFree(recv_buf));
    free(h_send);
    free(h_recv);
    NCCL_CHECK(ncclCommDestroy(comm));

    return (errors == 0) ? 0 : 1;
}
