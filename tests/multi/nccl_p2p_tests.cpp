#include <cuda_runtime.h>
#include <dlfcn.h>
#include <nccl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <vector>

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
    float *p2p_recv_buf;
    float *nccl_recv_buf;
    float *scatter_send_buf;
    float *scatter_recv_buf;
};

typedef ncclResult_t (*ncclGatherP2pFn)(const void *, void *, size_t,
                                        ncclDataType_t, int, ncclComm_t,
                                        cudaStream_t);
typedef ncclResult_t (*ncclScatterP2pFn)(const void *, void *, size_t,
                                         ncclDataType_t, int, ncclComm_t,
                                         cudaStream_t);
typedef ncclResult_t (*ncclAllGatherFn)(const void *, void *, size_t,
                                        ncclDataType_t, ncclComm_t,
                                        cudaStream_t);

static ncclGatherP2pFn resolve_nccl_gather_p2p(void)
{
    dlerror();
    void *sym = dlsym(RTLD_DEFAULT, "ncclGather_p2p");
    const char *err = dlerror();
    if (err != NULL || sym == NULL) {
        fprintf(stderr,
                "Error: could not resolve ncclGather_p2p. "
                "Run this test with the NCCL DMTCP plugin loaded. %s\n",
                err ? err : "");
        exit(1);
    }

    return (ncclGatherP2pFn)sym;
}

static ncclScatterP2pFn resolve_nccl_scatter_p2p(void)
{
    dlerror();
    void *sym = dlsym(RTLD_DEFAULT, "ncclScatter_p2p");
    const char *err = dlerror();
    if (err != NULL || sym == NULL) {
        fprintf(stderr,
                "Error: could not resolve ncclScatter_p2p. "
                "Run this test with the NCCL DMTCP plugin loaded. %s\n",
                err ? err : "");
        exit(1);
    }

    return (ncclScatterP2pFn)sym;
}

static ncclAllGatherFn resolve_nccl_all_gather(void)
{
    const char *libnccl_names[] = {"libnccl.so.2", "libnccl.so"};
    void *handle = NULL;

    for (size_t i = 0; i < sizeof(libnccl_names) / sizeof(libnccl_names[0]);
         i++) {
        dlerror();
        handle = dlopen(libnccl_names[i], RTLD_NOW | RTLD_LOCAL);
        if (handle != NULL)
            break;
    }

    if (handle == NULL) {
        const char *err = dlerror();
        fprintf(stderr,
                "Error: could not open libnccl to resolve ncclAllGather. %s\n",
                err ? err : "");
        exit(1);
    }

    dlerror();
    void *sym = dlsym(handle, "ncclAllGather");
    const char *err = dlerror();
    if (err != NULL || sym == NULL) {
        fprintf(stderr,
                "Error: libnccl does not export ncclAllGather. "
                "Cannot use NVIDIA ncclAllGather as the reference oracle. %s\n",
                err ? err : "");
        exit(1);
    }

    return (ncclAllGatherFn)sym;
}

static void usage(const char *prog)
{
    printf("Usage: %s [count] [root] [num_gpus] [mode]\n", prog);
    printf("  count    Number of floats sent by each GPU (default: 16)\n");
    printf("  root     Collective root rank/GPU (default: 0)\n");
    printf("  num_gpus Number of GPUs to use (default: all available)\n");
    printf("  mode     gather, scatter, or all (default: gather)\n");
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
        ctx->p2p_recv_buf = NULL;
        ctx->nccl_recv_buf = NULL;
        ctx->scatter_send_buf = NULL;
        ctx->scatter_recv_buf = NULL;

        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaStreamCreate(&ctx->stream));
        CUDA_CHECK(cudaMalloc((void **)&ctx->send_buf, count * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void **)&ctx->nccl_recv_buf,
                              num_gpus * count * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void **)&ctx->scatter_recv_buf,
                              count * sizeof(float)));

        std::vector<float> h_send(count);
        for (size_t j = 0; j < count; j++)
            h_send[j] = (float)(i * 1000 + j);

        CUDA_CHECK(cudaMemcpy(ctx->send_buf, h_send.data(),
                              count * sizeof(float), cudaMemcpyHostToDevice));
        std::vector<float> recv_sentinel(count, -2.0f);
        CUDA_CHECK(cudaMemcpy(ctx->scatter_recv_buf, recv_sentinel.data(),
                              recv_sentinel.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
    }

    CUDA_CHECK(cudaSetDevice(root));
    CUDA_CHECK(cudaMalloc((void **)&(*gpu)[root].p2p_recv_buf,
                          num_gpus * count * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&(*gpu)[root].scatter_send_buf,
                          num_gpus * count * sizeof(float)));

    std::vector<float> sentinel(num_gpus * count, -1.0f);
    CUDA_CHECK(cudaMemcpy((*gpu)[root].p2p_recv_buf, sentinel.data(),
                          sentinel.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    std::vector<float> h_scatter_send(num_gpus * count);
    for (int r = 0; r < num_gpus; r++) {
        for (size_t j = 0; j < count; j++) {
            h_scatter_send[(size_t)r * count + j] =
                (float)(100000 + r * 1000 + j);
        }
    }
    CUDA_CHECK(cudaMemcpy((*gpu)[root].scatter_send_buf,
                          h_scatter_send.data(),
                          h_scatter_send.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    for (int i = 0; i < num_gpus; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaMemcpy((*gpu)[i].nccl_recv_buf, sentinel.data(),
                              sentinel.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
    }
}

static void run_gather(std::vector<GpuCtx> *gpu, size_t count, int num_gpus,
                       int root)
{
    ncclGatherP2pFn ncclGather_p2p = resolve_nccl_gather_p2p();
    ncclAllGatherFn ncclAllGather = resolve_nccl_all_gather();

    /* Batch every rank's p2p gather call for the single-thread launch. */
    NCCL_CHECK(ncclGroupStart());
    for (int i = 0; i < num_gpus; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        void *recv_buf = (i == root) ? (*gpu)[root].p2p_recv_buf : NULL;

        NCCL_CHECK(ncclGather_p2p((*gpu)[i].send_buf, recv_buf, count,
                                  ncclFloat, root, (*gpu)[i].comm,
                                  (*gpu)[i].stream));
    }
    NCCL_CHECK(ncclGroupEnd());

    for (int i = 0; i < num_gpus; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaStreamSynchronize((*gpu)[i].stream));
    }

    /* Run reference NCCL AllGather; root's buffer has Gather's same layout. */
    NCCL_CHECK(ncclGroupStart());
    for (int i = 0; i < num_gpus; i++) {
        CUDA_CHECK(cudaSetDevice(i));

        NCCL_CHECK(ncclAllGather((*gpu)[i].send_buf,
                                 (*gpu)[i].nccl_recv_buf, count, ncclFloat,
                                 (*gpu)[i].comm, (*gpu)[i].stream));
    }
    NCCL_CHECK(ncclGroupEnd());

    for (int i = 0; i < num_gpus; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaStreamSynchronize((*gpu)[i].stream));
    }
}

static void run_scatter(std::vector<GpuCtx> *gpu, size_t count, int num_gpus,
                        int root)
{
    ncclScatterP2pFn ncclScatter_p2p = resolve_nccl_scatter_p2p();

    /* Batch every rank's p2p scatter call for the single-thread launch. */
    NCCL_CHECK(ncclGroupStart());
    for (int i = 0; i < num_gpus; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        const void *send_buf =
            (i == root) ? (*gpu)[root].scatter_send_buf : NULL;

        NCCL_CHECK(ncclScatter_p2p(send_buf, (*gpu)[i].scatter_recv_buf, count,
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
    std::vector<float> h_p2p_recv(num_gpus * count);
    std::vector<float> h_nccl_recv(num_gpus * count);

    CUDA_CHECK(cudaSetDevice(root));
    CUDA_CHECK(cudaMemcpy(h_p2p_recv.data(), gpu[root].p2p_recv_buf,
                          h_p2p_recv.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_nccl_recv.data(), gpu[root].nccl_recv_buf,
                          h_nccl_recv.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));

    int errors = 0;
    for (int r = 0; r < num_gpus; r++) {
        for (size_t j = 0; j < count; j++) {
            size_t idx = (size_t)r * count + j;
            float p2p = h_p2p_recv[idx];
            float nccl = h_nccl_recv[idx];

            if (p2p != nccl) {
                if (errors < 10) {
                    printf("[ROOT %d] Error at slot rank=%d elem=%zu: "
                           "p2p != ncclAllGather, p2p=%f, "
                           "ncclAllGather=%f\n",
                           root, r, j, p2p, nccl);
                }
                errors++;
            }
        }
    }

    if (errors == 0) {
        printf("[GATHER_P2P] SUCCESS: root %d received %d rank slots of %zu "
               "floats matching ncclAllGather.\n",
               root, num_gpus, count);
    } else {
        printf("[GATHER_P2P] FAILURE: %d p2p != ncclAllGather mismatches "
               "found.\n",
               errors);
    }

    return errors;
}

static int verify_scatter(const std::vector<GpuCtx> &gpu, size_t count,
                          int num_gpus, int root)
{
    int errors = 0;

    for (int r = 0; r < num_gpus; r++) {
        std::vector<float> h_recv(count);

        CUDA_CHECK(cudaSetDevice(r));
        CUDA_CHECK(cudaMemcpy(h_recv.data(), gpu[r].scatter_recv_buf,
                              h_recv.size() * sizeof(float),
                              cudaMemcpyDeviceToHost));

        for (size_t j = 0; j < count; j++) {
            float expected = (float)(100000 + r * 1000 + j);
            float actual = h_recv[j];

            if (actual != expected) {
                if (errors < 10) {
                    printf("[ROOT %d] Scatter error at rank=%d elem=%zu: "
                           "actual=%f expected=%f\n",
                           root, r, j, actual, expected);
                }
                errors++;
            }
        }
    }

    if (errors == 0) {
        printf("[SCATTER_P2P] SUCCESS: root %d scattered %zu floats to each "
               "of %d ranks.\n",
               root, count, num_gpus);
    } else {
        printf("[SCATTER_P2P] FAILURE: %d scatter mismatches found.\n",
               errors);
    }

    return errors;
}

static void teardown(std::vector<GpuCtx> *gpu)
{
    for (size_t i = 0; i < gpu->size(); i++) {
        CUDA_CHECK(cudaSetDevice((int)i));

        if ((*gpu)[i].send_buf)
            CUDA_CHECK(cudaFree((*gpu)[i].send_buf));
        if ((*gpu)[i].p2p_recv_buf)
            CUDA_CHECK(cudaFree((*gpu)[i].p2p_recv_buf));
        if ((*gpu)[i].nccl_recv_buf)
            CUDA_CHECK(cudaFree((*gpu)[i].nccl_recv_buf));
        if ((*gpu)[i].scatter_send_buf)
            CUDA_CHECK(cudaFree((*gpu)[i].scatter_send_buf));
        if ((*gpu)[i].scatter_recv_buf)
            CUDA_CHECK(cudaFree((*gpu)[i].scatter_recv_buf));

        CUDA_CHECK(cudaStreamDestroy((*gpu)[i].stream));
        NCCL_CHECK(ncclCommDestroy((*gpu)[i].comm));
    }
}

int main(int argc, char **argv)
{
    if (argc > 5) {
        usage(argv[0]);
        return 1;
    }

    size_t count = (argc >= 2) ? strtoull(argv[1], NULL, 10) : 16;
    int root = (argc >= 3) ? atoi(argv[2]) : 0;
    const char *mode = (argc >= 5) ? argv[4] : "gather";

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
    bool run_gather_mode =
        strcmp(mode, "gather") == 0 || strcmp(mode, "all") == 0;
    bool run_scatter_mode =
        strcmp(mode, "scatter") == 0 || strcmp(mode, "all") == 0;
    if (!run_gather_mode && !run_scatter_mode) {
        fprintf(stderr, "Error: mode must be gather, scatter, or all; got %s\n",
                mode);
        return 1;
    }

    printf("[P2P] GPUs=%d root=%d count=%zu floats mode=%s\n", num_gpus, root,
           count, mode);

    std::vector<GpuCtx> gpu;
    setup(&gpu, count, num_gpus, root);

    int errors = 0;
    if (run_gather_mode) {
        run_gather(&gpu, count, num_gpus, root);
        errors += verify(gpu, count, num_gpus, root);
    }
    if (run_scatter_mode) {
        run_scatter(&gpu, count, num_gpus, root);
        errors += verify_scatter(gpu, count, num_gpus, root);
    }

    teardown(&gpu);

    return errors == 0 ? 0 : 1;
}
