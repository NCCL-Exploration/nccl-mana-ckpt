#include <cuda_runtime.h>
#include <nccl.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <string>
#include <vector>

// Forward-declare advanced collectives if they are missing from older nccl.h versions
// but exist in the Perlmutter NCCL 2.x installation or custom forks.
#if !defined(NCCL_MAJOR) || (NCCL_MAJOR == 2 && NCCL_MINOR < 20)
extern "C" ncclResult_t ncclAlltoAll(const void* sendbuff, void* recvbuff, size_t count,
                                     ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) __attribute__((weak));
extern "C" ncclResult_t ncclGather(const void* sendbuff, void* recvbuff, size_t count,
                                   ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream) __attribute__((weak));
extern "C" ncclResult_t ncclScatter(const void* sendbuff, void* recvbuff, size_t count,
                                    ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream) __attribute__((weak));
#endif

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

const int ITERS = 100;
const int WARMUP = 2;
const size_t COUNT = 1024 * 1024; // 1M floats = 4MB per GPU

struct GpuCtx {
    ncclComm_t comm;
    cudaStream_t stream;
    float *send_buf;
    float *recv_buf;
};

template <typename Func>
void measure_collective(const std::string& name, std::vector<GpuCtx>& ctxs, int num_gpus, Func func) {
    // Warmup
    for (int i = 0; i < WARMUP; ++i) {
        NCCL_CHECK(ncclGroupStart());
        for (int g = 0; g < num_gpus; ++g) {
            CUDA_CHECK(cudaSetDevice(g));
            func(g, ctxs[g]);
        }
        NCCL_CHECK(ncclGroupEnd());
        for (int g = 0; g < num_gpus; ++g) {
            CUDA_CHECK(cudaSetDevice(g));
            CUDA_CHECK(cudaStreamSynchronize(ctxs[g].stream));
        }
    }

    // Timed run
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERS; ++i) {
        NCCL_CHECK(ncclGroupStart());
        for (int g = 0; g < num_gpus; ++g) {
            CUDA_CHECK(cudaSetDevice(g));
            func(g, ctxs[g]);
        }
        NCCL_CHECK(ncclGroupEnd());
        for (int g = 0; g < num_gpus; ++g) {
            CUDA_CHECK(cudaSetDevice(g));
            CUDA_CHECK(cudaStreamSynchronize(ctxs[g].stream));
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> diff = end - start;
    double avg_time_ms = (diff.count() * 1000.0) / ITERS;
    
    printf("%-20s : %.4f ms\n", name.c_str(), avg_time_ms);
}

int main(int argc, char* argv[]) {
    int num_gpus = 4;
    int devs[4] = {0, 1, 2, 3};
    
    int dev_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&dev_count));
    if (dev_count < num_gpus) num_gpus = dev_count;
    
    printf("Profiling NCCL Collectives on %d GPUs (Iterations: %d, Payload: %zu floats/GPU)\n", num_gpus, ITERS, COUNT);
    
    std::vector<GpuCtx> ctxs(num_gpus);
    std::vector<ncclComm_t> comms(num_gpus);
    NCCL_CHECK(ncclCommInitAll(comms.data(), num_gpus, devs));
    
    for (int g = 0; g < num_gpus; ++g) {
        CUDA_CHECK(cudaSetDevice(g));
        ctxs[g].comm = comms[g];
        CUDA_CHECK(cudaStreamCreate(&ctxs[g].stream));
        // Allocate enough space for allgather/alltoall (num_gpus * COUNT)
        CUDA_CHECK(cudaMalloc(&ctxs[g].send_buf, COUNT * sizeof(float) * num_gpus)); 
        CUDA_CHECK(cudaMalloc(&ctxs[g].recv_buf, COUNT * sizeof(float) * num_gpus));
    }
    
    printf("----------------------------------------\n");
    printf("%-20s   %s\n", "Collective", "Avg Runtime (ms)");
    printf("----------------------------------------\n");
    
    measure_collective("AllReduce", ctxs, num_gpus, [&](int g, GpuCtx& ctx) {
        NCCL_CHECK(ncclAllReduce(ctx.send_buf, ctx.recv_buf, COUNT, ncclFloat32, ncclSum, ctx.comm, ctx.stream));
    });
    
    measure_collective("Broadcast", ctxs, num_gpus, [&](int g, GpuCtx& ctx) {
        NCCL_CHECK(ncclBroadcast(ctx.send_buf, ctx.recv_buf, COUNT, ncclFloat32, 0, ctx.comm, ctx.stream));
    });
    
    measure_collective("Reduce", ctxs, num_gpus, [&](int g, GpuCtx& ctx) {
        NCCL_CHECK(ncclReduce(ctx.send_buf, ctx.recv_buf, COUNT, ncclFloat32, ncclSum, 0, ctx.comm, ctx.stream));
    });
    
    measure_collective("AllGather", ctxs, num_gpus, [&](int g, GpuCtx& ctx) {
        NCCL_CHECK(ncclAllGather(ctx.send_buf, ctx.recv_buf, COUNT, ncclFloat32, ctx.comm, ctx.stream));
    });
    
    measure_collective("ReduceScatter", ctxs, num_gpus, [&](int g, GpuCtx& ctx) {
        NCCL_CHECK(ncclReduceScatter(ctx.send_buf, ctx.recv_buf, COUNT, ncclFloat32, ncclSum, ctx.comm, ctx.stream));
    });

    if (ncclAlltoAll != nullptr) {
        measure_collective("AlltoAll", ctxs, num_gpus, [&](int g, GpuCtx& ctx) {
            NCCL_CHECK(ncclAlltoAll(ctx.send_buf, ctx.recv_buf, COUNT, ncclFloat32, ctx.comm, ctx.stream));
        });
    } else {
        printf("%-20s : (Not Supported by native NCCL build)\n", "AlltoAll");
    }

    if (ncclGather != nullptr) {
        measure_collective("Gather", ctxs, num_gpus, [&](int g, GpuCtx& ctx) {
            NCCL_CHECK(ncclGather(ctx.send_buf, ctx.recv_buf, COUNT, ncclFloat32, 0, ctx.comm, ctx.stream));
        });
    } else {
        printf("%-20s : (Not Supported by native NCCL build)\n", "Gather");
    }

    if (ncclScatter != nullptr) {
        measure_collective("Scatter", ctxs, num_gpus, [&](int g, GpuCtx& ctx) {
            NCCL_CHECK(ncclScatter(ctx.send_buf, ctx.recv_buf, COUNT, ncclFloat32, 0, ctx.comm, ctx.stream));
        });
    } else {
        printf("%-20s : (Not Supported by native NCCL build)\n", "Scatter");
    }
    
    // Cleanup
    for (int g = 0; g < num_gpus; ++g) {
        CUDA_CHECK(cudaSetDevice(g));
        CUDA_CHECK(cudaFree(ctxs[g].send_buf));
        CUDA_CHECK(cudaFree(ctxs[g].recv_buf));
        CUDA_CHECK(cudaStreamDestroy(ctxs[g].stream));
        NCCL_CHECK(ncclCommDestroy(ctxs[g].comm));
    }
    
    return 0;
}
