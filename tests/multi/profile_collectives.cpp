#include <cuda_runtime.h>
#include <nccl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

extern "C" ncclResult_t ncclAlltoAll(const void* sendbuff, void* recvbuff, size_t count,
                                     ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) __attribute__((weak));
extern "C" ncclResult_t ncclGather(const void* sendbuff, void* recvbuff, size_t count,
                                   ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream) __attribute__((weak));
extern "C" ncclResult_t ncclScatter(const void* sendbuff, void* recvbuff, size_t count,
                                    ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream) __attribute__((weak));

#define CUDA_CHECK(cmd) \
    do { \
        cudaError_t e = cmd; \
        if (e != cudaSuccess) { \
            printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
            exit(1); \
        } \
    } while (0)

#define NCCL_CHECK(cmd) \
    do { \
        ncclResult_t r = cmd; \
        if (r != ncclSuccess) { \
            printf("NCCL error %s:%d: %s\n", __FILE__, __LINE__, ncclGetErrorString(r)); \
            exit(1); \
        } \
    } while (0)

#define ITERS 10
#define WARMUP 2
#define COUNT (1024 * 1024)

struct GpuCtx {
    ncclComm_t comm;
    cudaStream_t stream;
    float *send_buf;
    float *recv_buf;
};

double get_time_ms() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec * 1000.0) + (tv.tv_usec / 1000.0);
}

#define MEASURE_COLLECTIVE(NAME, FUNC_CALL) do { \
    for (i = 0; i < WARMUP; ++i) { \
        NCCL_CHECK(ncclGroupStart()); \
        for (g = 0; g < num_gpus; ++g) { \
            CUDA_CHECK(cudaSetDevice(g)); \
            FUNC_CALL; \
        } \
        NCCL_CHECK(ncclGroupEnd()); \
        for (g = 0; g < num_gpus; ++g) { \
            CUDA_CHECK(cudaSetDevice(g)); \
            CUDA_CHECK(cudaStreamSynchronize(ctxs[g].stream)); \
        } \
    } \
    double t0 = get_time_ms(); \
    for (i = 0; i < ITERS; ++i) { \
        NCCL_CHECK(ncclGroupStart()); \
        for (g = 0; g < num_gpus; ++g) { \
            CUDA_CHECK(cudaSetDevice(g)); \
            FUNC_CALL; \
        } \
        NCCL_CHECK(ncclGroupEnd()); \
        for (g = 0; g < num_gpus; ++g) { \
            CUDA_CHECK(cudaSetDevice(g)); \
            CUDA_CHECK(cudaStreamSynchronize(ctxs[g].stream)); \
        } \
    } \
    printf("%-20s : %.4f ms\n", NAME, (get_time_ms() - t0) / ITERS); \
} while(0)

int main(int argc, char* argv[]) {
    int num_gpus = 4;
    int dev_count = 0;
    int g, i;
    struct GpuCtx ctxs[8];
    
    CUDA_CHECK(cudaGetDeviceCount(&dev_count));
    if (dev_count < num_gpus) num_gpus = dev_count;
    
    printf("Profiling NCCL Collectives on %d GPUs (Iterations: %d, Payload: %d floats/GPU)\n", num_gpus, ITERS, COUNT);
    
    ncclUniqueId id;
    NCCL_CHECK(ncclGetUniqueId(&id));
    
    NCCL_CHECK(ncclGroupStart());
    for (g = 0; g < num_gpus; ++g) {
        CUDA_CHECK(cudaSetDevice(g));
        NCCL_CHECK(ncclCommInitRank(&ctxs[g].comm, num_gpus, id, g));
    }
    NCCL_CHECK(ncclGroupEnd());
    
    for (g = 0; g < num_gpus; ++g) {
        CUDA_CHECK(cudaSetDevice(g));
        CUDA_CHECK(cudaStreamCreate(&ctxs[g].stream));
        CUDA_CHECK(cudaMalloc(&ctxs[g].send_buf, COUNT * sizeof(float) * num_gpus)); 
        CUDA_CHECK(cudaMalloc(&ctxs[g].recv_buf, COUNT * sizeof(float) * num_gpus));
        CUDA_CHECK(cudaMemset(ctxs[g].send_buf, 0, COUNT * sizeof(float) * num_gpus));
        CUDA_CHECK(cudaMemset(ctxs[g].recv_buf, 0, COUNT * sizeof(float) * num_gpus));
    }
    
    printf("----------------------------------------\n");
    printf("%-20s   %s\n", "Collective", "Avg Runtime (ms)");
    printf("----------------------------------------\n");
    fflush(stdout);
    
    MEASURE_COLLECTIVE("AllReduce", NCCL_CHECK(ncclAllReduce(ctxs[g].send_buf, ctxs[g].recv_buf, COUNT, ncclFloat32, ncclSum, ctxs[g].comm, ctxs[g].stream)));
    MEASURE_COLLECTIVE("Broadcast", NCCL_CHECK(ncclBroadcast(ctxs[g].send_buf, ctxs[g].recv_buf, COUNT, ncclFloat32, 0, ctxs[g].comm, ctxs[g].stream)));
    MEASURE_COLLECTIVE("Reduce", NCCL_CHECK(ncclReduce(ctxs[g].send_buf, ctxs[g].recv_buf, COUNT, ncclFloat32, ncclSum, 0, ctxs[g].comm, ctxs[g].stream)));
    MEASURE_COLLECTIVE("AllGather", NCCL_CHECK(ncclAllGather(ctxs[g].send_buf, ctxs[g].recv_buf, COUNT, ncclFloat32, ctxs[g].comm, ctxs[g].stream)));
    MEASURE_COLLECTIVE("ReduceScatter", NCCL_CHECK(ncclReduceScatter(ctxs[g].send_buf, ctxs[g].recv_buf, COUNT, ncclFloat32, ncclSum, ctxs[g].comm, ctxs[g].stream)));
    
    if (ncclAlltoAll != NULL) {
        MEASURE_COLLECTIVE("AlltoAll", NCCL_CHECK(ncclAlltoAll(ctxs[g].send_buf, ctxs[g].recv_buf, COUNT, ncclFloat32, ctxs[g].comm, ctxs[g].stream)));
    }
    if (ncclGather != NULL) {
        MEASURE_COLLECTIVE("Gather", NCCL_CHECK(ncclGather(ctxs[g].send_buf, ctxs[g].recv_buf, COUNT, ncclFloat32, 0, ctxs[g].comm, ctxs[g].stream)));
    }
    if (ncclScatter != NULL) {
        MEASURE_COLLECTIVE("Scatter", NCCL_CHECK(ncclScatter(ctxs[g].send_buf, ctxs[g].recv_buf, COUNT, ncclFloat32, 0, ctxs[g].comm, ctxs[g].stream)));
    }
    
    for (g = 0; g < num_gpus; ++g) {
        CUDA_CHECK(cudaSetDevice(g));
        CUDA_CHECK(cudaFree(ctxs[g].send_buf));
        CUDA_CHECK(cudaFree(ctxs[g].recv_buf));
        CUDA_CHECK(cudaStreamDestroy(ctxs[g].stream));
        NCCL_CHECK(ncclCommDestroy(ctxs[g].comm));
    }
    return 0;
}
