/*
 * The NCCL-using half of nccl_dlopen_test: a shared library that is DT_NEEDED on
 * libnccl and is dlopen'd RTLD_LOCAL by a driver with no NCCL linkage, mirroring
 * how CPython loads libtorch_cuda.so. Replays
 * ProcessGroupNCCL::initNCCLComm's sequence with one rank.
 */
#include <cuda_runtime.h>
#include <nccl.h>
#include <stdio.h>

#define STEP(label, expr)                                                                          \
    do {                                                                                           \
        ncclResult_t r_ = (expr);                                                                  \
        printf("  %-24s -> %d (%s)\n", label, (int)r_, ncclGetErrorString(r_));                    \
        fflush(stdout);                                                                            \
        if (r_ != ncclSuccess && r_ != ncclInProgress) {                                           \
            printf("FAIL %s returned %d\n", label, (int)r_);                                       \
            return 1;                                                                              \
        }                                                                                          \
    } while (0)

int run_pytorch_sequence(void) {
    ncclUniqueId id;
    ncclComm_t comm = NULL;
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    cudaStream_t stream;
    float *buf = NULL;

    if (cudaSetDevice(0) != cudaSuccess) {
        printf("FAIL cudaSetDevice\n");
        return 1;
    }
    cudaStreamCreate(&stream);
    cudaMalloc((void **)&buf, 4 * sizeof(float));
    cudaMemset(buf, 0, 4 * sizeof(float));

    STEP("ncclGetUniqueId", ncclGetUniqueId(&id));
    STEP("ncclGroupStart", ncclGroupStart());
    STEP("ncclCommInitRankConfig", ncclCommInitRankConfig(&comm, 1, id, 0, &config));
    STEP("ncclGroupEnd", ncclGroupEnd());
    printf("  comm handle              = %p\n", (void *)comm);
    fflush(stdout);
    STEP("ncclAllReduce", ncclAllReduce(buf, buf, 4, ncclFloat, ncclSum, comm, stream));
    cudaStreamSynchronize(stream);
    STEP("ncclCommDestroy", ncclCommDestroy(comm));
    return 0;
}
