#include "nccl_wrappers.h"
#include <stdint.h>

volatile int ckpt_pending = 0;
int g_my_rank = 0;
void *drain_buffer = NULL;

void init_rank_info(ncclComm_t comm) {}
void add_stream(cudaStream_t stream) {}
int dmtcp_kvdb_set(const char *key, int64_t val) {
    return 1;
}
int64_t dmtcp_kvdb_get(const char *key) {
    return 0;
}

const char *cudaGetErrorString(cudaError_t e) {
    return "stub";
}
cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind) {
    return cudaSuccess;
}
