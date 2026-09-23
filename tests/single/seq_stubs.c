/*
 * Stubs letting src/nccl_wrappers.c link into the host-only gate test: no DMTCP,
 * no CUDA runtime, no NCCL runtime. The real NCCL entry points are resolved with
 * dlsym at call time, so they need no link-time stand-in as long as the test
 * never actually issues a collective.
 */

#include "nccl_wrappers.h"
#include "vcomm.h"
#include <stdint.h>

/* Owned by libncclplugin.c in the real build. */
volatile int ckpt_pending = 0;
int g_my_rank = 0;
int g_nranks = 1;

void add_stream(cudaStream_t stream) {
    (void)stream;
}

void init_rank_info(vcomm_t *v) {
    (void)v;
}

/* DMTCP's checkpoint-exclusion lock is a no-op here: the test is single
 * threaded, and nothing is being checkpointed. */
void dmtcp_plugin_disable_ckpt(void) {}
void dmtcp_plugin_enable_ckpt(void) {}

/* Referenced by the drained-message replay path and by comm registration. */
cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind,
                            cudaStream_t stream) {
    (void)dst;
    (void)src;
    (void)count;
    (void)kind;
    (void)stream;
    return cudaSuccess;
}

cudaError_t cudaGetDevice(int *device) {
    if (device)
        *device = 0;
    return cudaSuccess;
}
