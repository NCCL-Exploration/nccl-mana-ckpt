#include <cuda_runtime.h>
#include <nccl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

#define CUDA_CHECK(cmd)                                                                            \
    do {                                                                                           \
        cudaError_t err = cmd;                                                                     \
        if (err != cudaSuccess) {                                                                  \
            printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__,                          \
                   cudaGetErrorString(err));                                                       \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    } while (0)

#define NCCL_CHECK(cmd)                                                                            \
    do {                                                                                           \
        ncclResult_t res = cmd;                                                                    \
        if (res != ncclSuccess) {                                                                  \
            printf("Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__,                          \
                   ncclGetErrorString(res));                                                       \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    } while (0)

int main(int argc, char *argv[]) {
    int nRanks = 2;
    int devs;
    CUDA_CHECK(cudaGetDeviceCount(&devs));
    if (devs < 2) {
        printf("Need at least 2 GPUs to run this test. Found %d\n", devs);
        return 0; // exit gracefully for testing environments
    }

    ncclUniqueId id;
    NCCL_CHECK(ncclGetUniqueId(&id));

    pid_t pid = fork();
    if (pid < 0) {
        perror("fork");
        exit(EXIT_FAILURE);
    }

    int rank = (pid == 0) ? 1 : 0;

    // Each process needs to set its device
    CUDA_CHECK(cudaSetDevice(rank));

    ncclComm_t comm;
    printf("[Rank %d] Initializing NCCL communicator...\n", rank);
    NCCL_CHECK(ncclCommInitRank(&comm, nRanks, id, rank));

    printf("[Rank %d] NCCL initialized successfully. Sleeping for 15 seconds to allow DMTCP "
           "checkpoint...\n",
           rank);
    sleep(15);

    printf("[Rank %d] Woke up. Destroying communicator...\n", rank);
    NCCL_CHECK(ncclCommDestroy(comm));

    if (pid > 0) {
        int status;
        wait(&status);
        if (WIFEXITED(status) && WEXITSTATUS(status) == 0) {
            printf("[Parent] Test completed successfully.\n");
        } else {
            printf("[Parent] Child test failed.\n");
            return EXIT_FAILURE;
        }
    }

    return 0;
}
