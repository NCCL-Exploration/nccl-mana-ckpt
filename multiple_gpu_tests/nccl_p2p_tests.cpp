#include <cuda_runtime.h>
#include <nccl.h>
#include <stdio.h>
#include <stdlib.h>

#include <vector>

#include "nccl_wrappers.h"

#define CUDA_CHECK(cmd)                                                        \
    do {                                                                       \
        cudaError_t e = (cmd);                                                 \
        if (e != cudaSuccess) {                                                \
            printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__,              \
                   cudaGetErrorString(e));                                     \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

#define NCCL_CHECK(cmd)                                                        \
    do {                                                                       \
        ncclResult_t r = (cmd);                                                \
        if (r != ncclSuccess) {                                                \
            printf("NCCL error %s:%d: %s\n", __FILE__, __LINE__,              \
                   ncclGetErrorString(r));                                     \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

struct GpuCtx {
    ncclComm_t comm;
    cudaStream_t stream;
    float *send_buf;
    float *recv_buf;
};

static void usage(const char *prog)
{
    printf("Usage: %s [count] [root] [num_gpus]\n", prog);
    printf("  count    Number of floats sent by each GPU (default: 16)\n");
    printf("  root     Gather root rank/GPU (default: 0)\n");
    printf("  num_gpus Number of GPUs to use (default: all available)\n");
}

static void setup(std::vector<GpuCtx> *gpu, size_t count, int num_gpus, int root)
{
    std::vector<ncclComm_t> comms(num_gpus);
    std::vector<int> dev_list(num_gpus);

    for (int i = 0; i < num_gpus; i++)
        dev_list[i] = i;

    NCCL_CHECK(ncclCommInitAll(comms.data(), num_gpus, dev_list.data()));

    gpu->resize(num_gpus);
    for (int i = 0; i < num_gpus; i++) {
        GpuCtx *ctx = &(*gpu)[i];
        ctx->comm = comms[i];
        ctx->send_buf = NULL;
        ctx->recv_buf = NULL;

        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaStreamCreate(&ctx->stream));
        CUDA_CHECK(cudaMalloc((void **)&ctx->send_buf, count * sizeof(float)));

        std::vector<float> h_send(count);
        for (size_t j = 0; j < count; j++)
            h_send[j] = (float)(i * 1000 + j);

        CUDA_CHECK(cudaMemcpy(ctx->send_buf, h_send.data(),
                              count * sizeof(float), cudaMemcpyHostToDevice));
    }

    CUDA_CHECK(cudaSetDevice(root));
    CUDA_CHECK(cudaMalloc((void **)&(*gpu)[root].recv_buf,
                          num_gpus * count * sizeof(float)));

    std::vector<float> sentinel(num_gpus * count, -1.0f);
    CUDA_CHECK(cudaMemcpy((*gpu)[root].recv_buf, sentinel.data(),
                          sentinel.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
}

static void run_gather(std::vector<GpuCtx> *gpu, size_t count, int num_gpus,
                       int root)
{
    /* Batch every rank's gather call for the single-thread multi-GPU launch. */
    NCCL_CHECK(ncclGroupStart());
    for (int i = 0; i < num_gpus; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        void *recv_buf = (i == root) ? (*gpu)[root].recv_buf : NULL;

        NCCL_CHECK(ncclGather_p2p((*gpu)[i].send_buf, recv_buf, count,
                                  ncclFloat, root, (*gpu)[i].comm,
                                  (*gpu)[i].stream));
    }
    NCCL_CHECK(ncclGroupEnd());

    for (int i = 0; i < num_gpus; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaStreamSynchronize((*gpu)[i].stream));
    }
}

static int verify(const std::vector<GpuCtx> &gpu, size_t count, int num_gpus,
                  int root)
{
    std::vector<float> h_recv(num_gpus * count);

    CUDA_CHECK(cudaSetDevice(root));
    CUDA_CHECK(cudaMemcpy(h_recv.data(), gpu[root].recv_buf,
                          h_recv.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));

    int errors = 0;
    for (int r = 0; r < num_gpus; r++) {
        for (size_t j = 0; j < count; j++) {
            float expected = (float)(r * 1000 + j);
            float got = h_recv[(size_t)r * count + j];

            if (got != expected) {
                if (errors < 10) {
                    printf("[ROOT %d] Error at slot rank=%d elem=%zu: "
                           "expected %f, got %f\n",
                           root, r, j, expected, got);
                }
                errors++;
            }
        }
    }

    if (errors == 0) {
        printf("[GATHER_P2P] SUCCESS: root %d received %d rank slots of %zu "
               "floats in expected order.\n",
               root, num_gpus, count);
    } else {
        printf("[GATHER_P2P] FAILURE: %d data errors found.\n", errors);
    }

    return errors;
}

static void teardown(std::vector<GpuCtx> *gpu)
{
    for (size_t i = 0; i < gpu->size(); i++) {
        CUDA_CHECK(cudaSetDevice((int)i));

        if ((*gpu)[i].send_buf)
            CUDA_CHECK(cudaFree((*gpu)[i].send_buf));
        if ((*gpu)[i].recv_buf)
            CUDA_CHECK(cudaFree((*gpu)[i].recv_buf));

        CUDA_CHECK(cudaStreamDestroy((*gpu)[i].stream));
        NCCL_CHECK(ncclCommDestroy((*gpu)[i].comm));
    }
}

int main(int argc, char **argv)
{
    if (argc > 4) {
        usage(argv[0]);
        return 1;
    }

    size_t count = (argc >= 2) ? strtoull(argv[1], NULL, 10) : 16;
    int root = (argc >= 3) ? atoi(argv[2]) : 0;

    int available_gpus = 0;
    CUDA_CHECK(cudaGetDeviceCount(&available_gpus));

    int num_gpus = (argc >= 4) ? atoi(argv[3]) : available_gpus;

    if (count == 0) {
        fprintf(stderr, "Error: count must be greater than 0\n");
        return 1;
    }
    if (num_gpus < 2) {
        fprintf(stderr, "Error: need at least 2 GPUs, found/requested %d\n",
                num_gpus);
        return 1;
    }
    if (num_gpus > available_gpus) {
        fprintf(stderr, "Error: requested %d GPUs but only %d available\n",
                num_gpus, available_gpus);
        return 1;
    }
    if (root < 0 || root >= num_gpus) {
        fprintf(stderr, "Error: root must be in [0, %d], got %d\n",
                num_gpus - 1, root);
        return 1;
    }

    printf("[GATHER_P2P] GPUs=%d root=%d count=%zu floats\n",
           num_gpus, root, count);

    std::vector<GpuCtx> gpu;
    setup(&gpu, count, num_gpus, root);
    run_gather(&gpu, count, num_gpus, root);
    int errors = verify(gpu, count, num_gpus, root);
    teardown(&gpu);

    return errors == 0 ? 0 : 1;
}
