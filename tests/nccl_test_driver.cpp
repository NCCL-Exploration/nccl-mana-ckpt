#include <stdio.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <stdlib.h>
#include <string.h>

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

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s <mode> [count] [size_kb]\n", argv[0]);
        printf("Modes:\n");
        printf("  1: Basic P2P Complete\n");
        printf("  2: Basic P2P with post-ckpt sleep\n");
        printf("  3: Basic P2P with pre-init sleep\n");
        printf("  4: Single message in-transit (Mode 4.1/4.2)\n");
        printf("  5: Multi message in-transit (Mode 5.1/5.2)\n");
        return 1;
    }

    int mode = atoi(argv[1]);
    int num_msgs = (argc >= 3) ? atoi(argv[2]) : 1;
    int size_kb = (argc >= 4) ? atoi(argv[3]) : 4; // 4KB default
    size_t count = (size_kb * 1024) / sizeof(float);

    int rank = 0, nranks = 1;
    ncclComm_t comm;
    cudaStream_t stream;

    CUDA_CHECK(cudaSetDevice(0));
    
    if (mode == 3) {
        printf("[TEST] Mode 3: Sleeping 10s before NCCL init...\n");
        sleep(10);
    }

    NCCL_CHECK(ncclCommInitAll(&comm, 1, &rank));
    CUDA_CHECK(cudaStreamCreate(&stream));

    float *send_buff, *recv_buff;
    CUDA_CHECK(cudaMalloc(&send_buff, count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&recv_buff, count * sizeof(float)));

    // Initialize data
    float *h_send = (float*)malloc(count * sizeof(float));
    for(size_t i=0; i<count; i++) h_send[i] = (float)(mode + i);
    CUDA_CHECK(cudaMemcpy(send_buff, h_send, count * sizeof(float), cudaMemcpyHostToDevice));

    printf("[TEST] Mode %d starting. Msg Count: %d, Size: %d KB\n", mode, num_msgs, size_kb);

    if (mode == 1) {
        // Just run and verify
        for (int i = 0; i < num_msgs; i++) {
            NCCL_CHECK(ncclSend(send_buff, count, ncclFloat, 0, comm, stream));
            NCCL_CHECK(ncclRecv(recv_buff, count, ncclFloat, 0, comm, stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        printf("[TEST] Mode 1: Communication complete.\n");
    } 
    else if (mode == 2) {
        // Complete then sleep
        for (int i = 0; i < num_msgs; i++) {
            NCCL_CHECK(ncclSend(send_buff, count, ncclFloat, 0, comm, stream));
            NCCL_CHECK(ncclRecv(recv_buff, count, ncclFloat, 0, comm, stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        printf("[TEST] Mode 2: Messaging finished. Sleeping 20s for post-ckpt...\n");
        sleep(20);
    }
    else if (mode == 4 || mode == 5) {
        // In-transit logic
        // We send all messages FIRST, then sleep, then recv
        printf("[TEST] Mode %d: Sending %d messages...\n", mode, num_msgs);
        for (int i = 0; i < num_msgs; i++) {
            NCCL_CHECK(ncclSend(send_buff, count, ncclFloat, 0, comm, stream));
        }
        
        printf("[TEST] Mode %d: Messages SENT. Sleeping 20s to allow checkpoint while in-transit...\n", mode);
        sleep(20);

        printf("[TEST] Mode %d: Waking up. Starting receives...\n", mode);
        for (int i = 0; i < num_msgs; i++) {
             NCCL_CHECK(ncclRecv(recv_buff, count, ncclFloat, 0, comm, stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        printf("[TEST] Mode %d: All receives complete.\n", mode);
    }

    // Verification
    float *h_recv = (float*)malloc(count * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_recv, recv_buff, count * sizeof(float), cudaMemcpyDeviceToHost));
    
    int errors = 0;
    for (size_t i = 0; i < count; i++) {
        if (h_recv[i] != h_send[i]) {
            if (errors < 5) printf("Error at %zu: expected %f, got %f\n", i, h_send[i], h_recv[i]);
            errors++;
        }
    }

    if (errors == 0) {
        printf("[TEST] SUCCESS: Data integrity verified.\n");
    } else {
        printf("[TEST] FAILURE: %d data errors found.\n", errors);
    }

    CUDA_CHECK(cudaFree(send_buff));
    CUDA_CHECK(cudaFree(recv_buff));
    free(h_send);
    free(h_recv);
    NCCL_CHECK(ncclCommDestroy(comm));

    return (errors == 0) ? 0 : 1;
}
