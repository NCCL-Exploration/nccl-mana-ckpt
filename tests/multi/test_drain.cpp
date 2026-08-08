#include "nccl.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <unistd.h>

int main(int argc, char *argv[]) {
    int rank, nranks;
    ncclCommt_t comm;
    cudaStream_t stream;

    ncclCommInitRank(&comm, nranks, ncclId, rank);
    cudaStreamCreate(&stream);

    size_t count = 1024 * 1024; // 1M elements
    float *buff;
    cudaMalloc((void **)&buff, count * sizeof(float));

    if (rank == 0) {
        printf("[Rank 0] Sending data to rank 1...\n");
        ncclSend(buff, count, ncclFloat, 1, comm, stream);
        printf("[Rank 0] Send complete, waiting for checkpoint...\n");

        while (1) {
            sleep(1);
        } // wait for ckpt
    } else if (rank == 1) {
        printf("[Rank 1] Delaying receive to simulate in-flight data...\n");
        sleep(15); // wait for sufficient amount of time for checkpoint

        printf("[Rank 1] Calling ncclRecv...\n");
        ncclRecv(buff, count, ncclFloat, 0, comm, stream);
        printf("[Rank 1] Data received...\n");
    }

    ncclCommDestroy(comm);
    return 0;
}
