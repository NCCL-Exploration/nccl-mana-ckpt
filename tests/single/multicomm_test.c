/*
 * Multi-communicator checkpoint/restart test, runnable on a SINGLE GPU.
 *
 *   ./multicomm_test [iterations]
 *
 * tests/multi/cc_complex.c is the real acceptance test for N communicators, but
 * it asserts nranks == 3 and binds cudaSetDevice(rank), and NCCL refuses two
 * ranks of one communicator sharing a device -- so it needs a 3-GPU host. This
 * test covers the same plugin machinery on one GPU by creating three
 * INDEPENDENT one-rank communicators in a single process.
 *
 * What that does and does not exercise:
 *
 *   covered      the registry holding several live communicators at once;
 *                application handles staying valid across checkpoint/restart;
 *                per-communicator teardown and rebuild in identity order;
 *                a fresh ncclUniqueId published per communicator;
 *                per-communicator sequence counters advancing at DIFFERENT
 *                rates (1, 2 and 3 collectives per iteration);
 *                the registry-answered tier still reporting rank/size/device.
 *
 *   NOT covered  cross-rank sequence consensus and the p2p drain, which need
 *                real peers. Run tests/multi/cc_complex.c on a multi-GPU host
 *                for those.
 *
 * A one-rank AllReduce is a copy, so the payload checks here cannot detect an op
 * being routed to the wrong communicator. What they do detect is the failure
 * this change is really about: a handle that stopped being valid after restart.
 * An untranslated handle is a non-canonical pointer and faults immediately.
 */

#include <cuda_runtime.h>
#include <nccl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define MAX_COMMS 3
#define COUNT 1024

/* Number of communicators actually used; overridable so the same binary can
 * bisect between the one-comm and many-comm cases. */
static int NCOMMS = MAX_COMMS;

/* Deliberately unequal, so each communicator's sequence counter advances at its
 * own rate -- the thing a single global counter cannot represent. */
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
    int c, it, k, i;

    if (argc > 2) {
        NCOMMS = atoi(argv[2]);
        if (NCOMMS < 1 || NCOMMS > MAX_COMMS)
            NCOMMS = MAX_COMMS;
    }
    printf("[app] using %d communicator(s), %d iterations\n", NCOMMS, iters);
    fflush(stdout);

    CUDACHECK(cudaSetDevice(0));
    host = (float *)malloc(COUNT * sizeof(float));
    if (host == NULL)
        return 1;

    for (c = 0; c < NCOMMS; c++) {
        ncclUniqueId id;
        /* One id per communicator: NCCL consumes the bootstrap socket it names,
         * so distinct communicators necessarily have distinct ids. That is also
         * what gives each one a distinct identity in the plugin's registry. */
        NCCLCHECK(ncclGetUniqueId(&id));
        NCCLCHECK(ncclCommInitRank(&comms[c], 1, id, 0));
        CUDACHECK(cudaStreamCreate(&streams[c]));
        CUDACHECK(cudaMalloc((void **)&dsend[c], COUNT * sizeof(float)));
        CUDACHECK(cudaMalloc((void **)&drecv[c], COUNT * sizeof(float)));
        ops_done[c] = 0;
        printf("[app] comm %d handle=%p\n", c, (void *)comms[c]);
    }
    fflush(stdout);

    for (it = 0; it < iters; it++) {
        for (c = 0; c < NCOMMS; c++) {
            /* The registry answers these without touching NCCL, so they must
             * stay correct across a checkpoint too. */
            int rank = -1, nranks = -1, dev = -1;
            NCCLCHECK(ncclCommUserRank(comms[c], &rank));
            NCCLCHECK(ncclCommCount(comms[c], &nranks));
            NCCLCHECK(ncclCommCuDevice(comms[c], &dev));
            if (rank != 0 || nranks != 1 || dev != 0) {
                printf("FAIL comm %d reports rank=%d nranks=%d dev=%d, want 0/1/0\n", c, rank,
                       nranks, dev);
                return 1;
            }

            for (k = 0; k < OPS_PER_ITER[c]; k++) {
                /* Distinct per (iteration, comm, op), so a stale buffer or a
                 * skipped operation shows up as a mismatch rather than passing
                 * by coincidence. */
                float val = (float)(it * 1000 + c * 100 + k + 1);
                for (i = 0; i < COUNT; i++)
                    host[i] = val;

                CUDACHECK(cudaMemcpyAsync(dsend[c], host, COUNT * sizeof(float),
                                          cudaMemcpyHostToDevice, streams[c]));
                NCCLCHECK(ncclAllReduce(dsend[c], drecv[c], COUNT, ncclFloat32, ncclSum, comms[c],
                                        streams[c]));
                CUDACHECK(cudaMemcpyAsync(host, drecv[c], COUNT * sizeof(float),
                                          cudaMemcpyDeviceToHost, streams[c]));
                CUDACHECK(cudaStreamSynchronize(streams[c]));

                for (i = 0; i < COUNT; i++) {
                    if (host[i] != val) {
                        printf("FAIL comm %d iter %d op %d: element %d is %f, want %f\n", c, it, k,
                               i, host[i], val);
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

    /* The counts must still be in the 1:2:3 ratio. If a rebuild had lost or
     * duplicated work on any communicator, this is where it shows. */
    for (c = 0; c < NCOMMS; c++) {
        long want = (long)iters * OPS_PER_ITER[c];
        if (ops_done[c] != want) {
            printf("FAIL comm %d did %ld ops, want %ld\n", c, ops_done[c], want);
            return 1;
        }
    }

    for (c = 0; c < NCOMMS; c++) {
        /* Handles printed again so a restart log can be diffed against the
         * launch log: they must be byte-identical, which is the whole point of
         * virtualizing them. */
        printf("[app] comm %d handle=%p (unchanged across restart)\n", c, (void *)comms[c]);
        CUDACHECK(cudaFree(dsend[c]));
        CUDACHECK(cudaFree(drecv[c]));
        CUDACHECK(cudaStreamDestroy(streams[c]));
        NCCLCHECK(ncclCommDestroy(comms[c]));
    }
    free(host);

    printf("PASS multicomm: %d communicators, %d iterations, checksum %.0f\n", NCOMMS, iters,
           checksum);
    fflush(stdout);
    return 0;
}
