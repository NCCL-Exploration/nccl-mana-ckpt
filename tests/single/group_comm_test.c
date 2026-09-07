/*
 * Group-region checkpoint/restart test, runnable on a SINGLE GPU.
 *
 *   ./group_comm_test [iterations]
 *
 * Covers the two shapes the rest of the suite never puts inside a group region:
 *
 *   1. COMMUNICATOR CREATION, which is how PyTorch creates every communicator
 *      (ProcessGroupNCCL::initNCCLComm wraps ncclCommInitRankConfig in group
 *      calls). Inside a region NCCL only queues the job, keeps the ncclComm_t*
 *      it was handed and writes through it at ncclGroupEnd, so rank, size and
 *      device cannot be read off the handle until the region closes.
 *
 *   2. COLLECTIVES. The gate is taken once at ncclGroupStart and held for the
 *      batch, so the per-comm sequence counters must advance from inside the
 *      region too.
 *
 * send_recv.c and the NAS IS benchmark do use group regions, but only around
 * ncclSend/ncclRecv with the comm created outside, so neither shape was covered.
 *
 * Like multicomm_test.c this uses independent one-rank communicators so it runs
 * on one GPU; cross-rank consensus and the p2p drain still need real peers.
 */

#include <cuda_runtime.h>
#include <nccl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define MAX_COMMS 3
#define MAX_OPS 3
#define COUNT 1024

static int NCOMMS = MAX_COMMS;

/* Unequal on purpose: a single global counter cannot describe a batch that
 * advances each communicator at its own rate. */
static const int OPS_PER_ITER[MAX_COMMS] = {1, 2, 3};

#define CUDACHECK(cmd)                                                                             \
    do {                                                                                           \
        cudaError_t e = (cmd);                                                                     \
        if (e != cudaSuccess) {                                                                    \
            printf("FAIL %s:%d CUDA: %s\n", __FILE__, __LINE__, cudaGetErrorString(e));            \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)

#define NCCLCHECK(cmd)                                                                             \
    do {                                                                                           \
        ncclResult_t r = (cmd);                                                                    \
        if (r != ncclSuccess) {                                                                    \
            printf("FAIL %s:%d NCCL: %s\n", __FILE__, __LINE__, ncclGetErrorString(r));            \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)

int main(int argc, char **argv) {
    int iters = (argc > 1) ? atoi(argv[1]) : 12;
    ncclComm_t comms[MAX_COMMS];
    cudaStream_t streams[MAX_COMMS];
    float *dsend[MAX_COMMS], *drecv[MAX_COMMS];
    float *host = NULL;
    long ops_done[MAX_COMMS];
    double checksum = 0.0;
    size_t slab = (size_t)COUNT * MAX_OPS * sizeof(float);
    int c, it, k, i;

    if (argc > 2) {
        NCOMMS = atoi(argv[2]);
        if (NCOMMS < 1 || NCOMMS > MAX_COMMS)
            NCOMMS = MAX_COMMS;
    }
    printf("[app] using %d communicator(s), %d iterations, group regions throughout\n", NCOMMS,
           iters);
    fflush(stdout);

    CUDACHECK(cudaSetDevice(0));
    host = (float *)malloc(slab);
    if (host == NULL)
        return 1;

    /* Shape 1: creation inside a region, one region per comm -- exactly
     * ProcessGroupNCCL::initNCCLComm. */
    for (c = 0; c < NCOMMS; c++) {
        ncclUniqueId id;
        NCCLCHECK(ncclGetUniqueId(&id));
        NCCLCHECK(ncclGroupStart());
        NCCLCHECK(ncclCommInitRank(&comms[c], 1, id, 0));
        NCCLCHECK(ncclGroupEnd());
        CUDACHECK(cudaStreamCreate(&streams[c]));
        CUDACHECK(cudaMalloc((void **)&dsend[c], slab));
        CUDACHECK(cudaMalloc((void **)&drecv[c], slab));
        ops_done[c] = 0;
        printf("[app] comm %d handle=%p\n", c, (void *)comms[c]);
    }
    fflush(stdout);

    for (it = 0; it < iters; it++) {
        /* Answered from the registry without touching NCCL, so it must be
         * right even for a comm born inside a region. */
        for (c = 0; c < NCOMMS; c++) {
            int rank = -1, nranks = -1, dev = -1;
            NCCLCHECK(ncclCommUserRank(comms[c], &rank));
            NCCLCHECK(ncclCommCount(comms[c], &nranks));
            NCCLCHECK(ncclCommCuDevice(comms[c], &dev));
            if (rank != 0 || nranks != 1 || dev != 0) {
                printf("FAIL comm %d reports rank=%d nranks=%d dev=%d, want 0/1/0\n", c, rank,
                       nranks, dev);
                return 1;
            }
        }

        /* Nothing runs until ncclGroupEnd, so no buffer may be reused. */
        for (c = 0; c < NCOMMS; c++) {
            for (k = 0; k < OPS_PER_ITER[c]; k++) {
                float val = (float)(it * 1000 + c * 100 + k + 1);
                for (i = 0; i < COUNT; i++)
                    host[i] = val;
                CUDACHECK(cudaMemcpyAsync(dsend[c] + (size_t)k * COUNT, host,
                                          COUNT * sizeof(float), cudaMemcpyHostToDevice,
                                          streams[c]));
            }
            CUDACHECK(cudaStreamSynchronize(streams[c]));
        }

        /* Shape 2: one region batching collectives across every comm. */
        NCCLCHECK(ncclGroupStart());
        for (c = 0; c < NCOMMS; c++) {
            for (k = 0; k < OPS_PER_ITER[c]; k++) {
                NCCLCHECK(ncclAllReduce(dsend[c] + (size_t)k * COUNT,
                                        drecv[c] + (size_t)k * COUNT, COUNT, ncclFloat32, ncclSum,
                                        comms[c], streams[c]));
            }
        }
        NCCLCHECK(ncclGroupEnd());

        for (c = 0; c < NCOMMS; c++) {
            CUDACHECK(cudaStreamSynchronize(streams[c]));
            CUDACHECK(cudaMemcpy(host, drecv[c], (size_t)OPS_PER_ITER[c] * COUNT * sizeof(float),
                                 cudaMemcpyDeviceToHost));
            for (k = 0; k < OPS_PER_ITER[c]; k++) {
                float val = (float)(it * 1000 + c * 100 + k + 1);
                for (i = 0; i < COUNT; i++) {
                    if (host[(size_t)k * COUNT + i] != val) {
                        printf("FAIL comm %d iter %d op %d: element %d is %f, want %f\n", c, it, k,
                               i, host[(size_t)k * COUNT + i], val);
                        return 1;
                    }
                }
                ops_done[c]++;
                checksum += val;
            }
        }

        printf("[app] iter %2d done; ops per comm: %ld %ld %ld; checksum %.0f\n", it, ops_done[0],
               NCOMMS > 1 ? ops_done[1] : 0L, NCOMMS > 2 ? ops_done[2] : 0L, checksum);
        fflush(stdout);
        sleep(1);
    }

    for (c = 0; c < NCOMMS; c++) {
        long want = (long)iters * OPS_PER_ITER[c];
        if (ops_done[c] != want) {
            printf("FAIL comm %d did %ld ops, want %ld\n", c, ops_done[c], want);
            return 1;
        }
    }

    for (c = 0; c < NCOMMS; c++) {
        printf("[app] comm %d handle=%p (unchanged across restart)\n", c, (void *)comms[c]);
        CUDACHECK(cudaFree(dsend[c]));
        CUDACHECK(cudaFree(drecv[c]));
        CUDACHECK(cudaStreamDestroy(streams[c]));
        NCCLCHECK(ncclCommDestroy(comms[c]));
    }
    free(host);

    printf("PASS groupcomm: %d communicators, %d iterations, checksum %.0f\n", NCOMMS, iters,
           checksum);
    fflush(stdout);
    return 0;
}
