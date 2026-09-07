/*
 * Non-blocking communicator through checkpoint/restart, on a SINGLE GPU.
 *
 *   ./nonblocking_test [iterations]
 *
 * PyTorch sets ncclConfig_t.blocking = 0 under TORCH_NCCL_USE_COMM_NONBLOCKING.
 * Creation then succeeds by returning ncclInProgress and the caller polls
 * ncclCommGetAsyncError, which is what this test does.
 *
 * The rebuild is the part that matters: the stored config is replayed verbatim,
 * so the comm is reconstructed non-blocking too and ncclCommInitRank returns
 * ncclInProgress there as well. Reading that as failure killed the process at
 * resume-in-place and again at restart, and publishing the handle without
 * waiting would hand application threads a comm NCCL still rejects.
 */

#include <cuda_runtime.h>
#include <nccl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define COUNT 1024

#define CUDACHECK(cmd)                                                                             \
    do {                                                                                           \
        cudaError_t e = (cmd);                                                                     \
        if (e != cudaSuccess) {                                                                    \
            printf("FAIL %s:%d CUDA: %s\n", __FILE__, __LINE__, cudaGetErrorString(e));            \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)

/* ncclInProgress is success for a non-blocking comm, not an error. */
#define NCCLCHECK(cmd)                                                                             \
    do {                                                                                           \
        ncclResult_t r = (cmd);                                                                    \
        if (r != ncclSuccess && r != ncclInProgress) {                                             \
            printf("FAIL %s:%d NCCL: %s\n", __FILE__, __LINE__, ncclGetErrorString(r));            \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)

static void wait_ready(ncclComm_t comm, const char *what) {
    long spins = 0;
    for (;;) {
        ncclResult_t state = ncclSuccess;
        NCCLCHECK(ncclCommGetAsyncError(comm, &state));
        if (state == ncclSuccess)
            return;
        if (state != ncclInProgress) {
            printf("FAIL %s: async error %s\n", what, ncclGetErrorString(state));
            exit(1);
        }
        if (++spins > 2000000) {
            printf("FAIL %s: never became ready\n", what);
            exit(1);
        }
        usleep(50);
    }
}

int main(int argc, char **argv) {
    int iters = (argc > 1) ? atoi(argv[1]) : 12;
    ncclUniqueId id;
    ncclComm_t comm = NULL;
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    cudaStream_t stream;
    float *dsend = NULL, *drecv = NULL;
    float *host = NULL;
    int it, i;

    config.blocking = 0;

    CUDACHECK(cudaSetDevice(0));
    CUDACHECK(cudaStreamCreate(&stream));
    CUDACHECK(cudaMalloc((void **)&dsend, COUNT * sizeof(float)));
    CUDACHECK(cudaMalloc((void **)&drecv, COUNT * sizeof(float)));
    host = (float *)malloc(COUNT * sizeof(float));
    if (host == NULL)
        return 1;

    /* Creation in a group region, as ProcessGroupNCCL does it. */
    NCCLCHECK(ncclGetUniqueId(&id));
    NCCLCHECK(ncclGroupStart());
    NCCLCHECK(ncclCommInitRankConfig(&comm, 1, id, 0, &config));
    NCCLCHECK(ncclGroupEnd());
    wait_ready(comm, "init");
    printf("[app] nonblocking comm handle=%p\n", (void *)comm);
    fflush(stdout);

    for (it = 0; it < iters; it++) {
        int rank = -1, nranks = -1, dev = -1;
        float val = (float)(it + 1);

        NCCLCHECK(ncclCommUserRank(comm, &rank));
        NCCLCHECK(ncclCommCount(comm, &nranks));
        NCCLCHECK(ncclCommCuDevice(comm, &dev));
        if (rank != 0 || nranks != 1 || dev != 0) {
            printf("FAIL iter %d reports rank=%d nranks=%d dev=%d, want 0/1/0\n", it, rank, nranks,
                   dev);
            return 1;
        }

        for (i = 0; i < COUNT; i++)
            host[i] = val;
        CUDACHECK(cudaMemcpyAsync(dsend, host, COUNT * sizeof(float), cudaMemcpyHostToDevice,
                                  stream));
        NCCLCHECK(ncclGroupStart());
        NCCLCHECK(ncclAllReduce(dsend, drecv, COUNT, ncclFloat32, ncclSum, comm, stream));
        NCCLCHECK(ncclGroupEnd());
        wait_ready(comm, "allreduce");
        CUDACHECK(cudaStreamSynchronize(stream));
        CUDACHECK(cudaMemcpy(host, drecv, COUNT * sizeof(float), cudaMemcpyDeviceToHost));
        for (i = 0; i < COUNT; i++) {
            if (host[i] != val) {
                printf("FAIL iter %d: element %d is %f, want %f\n", it, i, host[i], val);
                return 1;
            }
        }
        printf("[app] iter %2d ok\n", it);
        fflush(stdout);
        sleep(1);
    }

    printf("[app] handle=%p (unchanged across restart)\n", (void *)comm);
    NCCLCHECK(ncclCommDestroy(comm));
    free(host);
    printf("PASS nonblocking: %d iterations\n", iters);
    fflush(stdout);
    return 0;
}
