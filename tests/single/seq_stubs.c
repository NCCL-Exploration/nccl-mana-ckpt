#include "nccl_wrappers.h"
#include <dlfcn.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

volatile int ckpt_pending = 0;
int g_my_rank = -1;
int g_nranks = 0;
ncclComm_t g_comm = NULL;
void *drain_buffer = NULL;

struct fake_nccl_comm {
    int rank;
    int nranks;
};

static struct fake_nccl_comm configured_comms[2];
static struct fake_nccl_comm split_comm;
ncclComm_t fake_last_allreduce_comm = NULL;
ncclComm_t fake_last_destroyed_comm = NULL;
int fake_finalize_resolve_count = 0;
ncclResult_t fake_init_result = ncclSuccess;
ncclResult_t fake_split_result = ncclSuccess;
int fake_last_init_nranks;
int fake_last_init_rank;
ncclUniqueId fake_last_init_id;
ncclConfig_t *fake_last_init_config;
ncclComm_t fake_last_split_parent;
int fake_last_split_color;
int fake_last_split_key;
ncclConfig_t *fake_last_split_config;
ncclComm_t fake_last_finalized_comm;
ncclComm_t fake_last_aborted_comm;
int fake_use_alternate_comm;
int fake_split_queries_ready = 1;

static ncclResult_t fake_ncclCommInitRankConfig(ncclComm_t *comm, int nranks,
                                                ncclUniqueId comm_id, int rank,
                                                ncclConfig_t *config) {
    (void)comm_id;
    fake_last_init_nranks = nranks;
    fake_last_init_rank = rank;
    fake_last_init_id = comm_id;
    fake_last_init_config = config;
    configured_comms[fake_use_alternate_comm].rank = rank;
    configured_comms[fake_use_alternate_comm].nranks = nranks;
    *comm = &configured_comms[fake_use_alternate_comm];
    return fake_init_result;
}

static ncclResult_t fake_ncclCommSplit(ncclComm_t comm, int color, int key, ncclComm_t *newcomm,
                                       ncclConfig_t *config) {
    fake_last_split_parent = comm;
    fake_last_split_color = color;
    fake_last_split_key = key;
    fake_last_split_config = config;
    if (color == NCCL_SPLIT_NOCOLOR) {
        *newcomm = NULL;
        return fake_split_result;
    }
    split_comm.rank = comm->rank / 2;
    split_comm.nranks = comm->nranks / 2;
    *newcomm = &split_comm;
    return fake_split_result;
}

static ncclResult_t fake_ncclCommFinalize(ncclComm_t comm) {
    fake_last_finalized_comm = comm;
    return ncclSuccess;
}

static ncclResult_t fake_ncclCommAbort(ncclComm_t comm) {
    fake_last_aborted_comm = comm;
    return ncclSuccess;
}

static ncclResult_t fake_ncclCommDestroy(ncclComm_t comm) {
    fake_last_destroyed_comm = comm;
    return ncclSuccess;
}

static ncclResult_t fake_ncclAllReduce(const void *sendbuff, void *recvbuff, size_t count,
                                       ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm,
                                       cudaStream_t stream) {
    (void)sendbuff;
    (void)recvbuff;
    (void)count;
    (void)datatype;
    (void)op;
    (void)stream;
    fake_last_allreduce_comm = comm;
    return ncclSuccess;
}

void *dlsym(void *handle, const char *symbol) {
    (void)handle;
    if (strcmp(symbol, "ncclCommInitRankConfig") == 0)
        return (void *)fake_ncclCommInitRankConfig;
    if (strcmp(symbol, "ncclCommSplit") == 0)
        return (void *)fake_ncclCommSplit;
    if (strcmp(symbol, "ncclCommFinalize") == 0) {
        fake_finalize_resolve_count++;
        return (void *)fake_ncclCommFinalize;
    }
    if (strcmp(symbol, "ncclCommAbort") == 0)
        return (void *)fake_ncclCommAbort;
    if (strcmp(symbol, "ncclCommDestroy") == 0)
        return (void *)fake_ncclCommDestroy;
    if (strcmp(symbol, "ncclAllReduce") == 0)
        return (void *)fake_ncclAllReduce;
    return NULL;
}

ncclResult_t ncclCommUserRank(ncclComm_t comm, int *rank) {
    if ((uintptr_t)comm <= 2) {
        *rank = (int)(uintptr_t)comm - 1;
        return ncclSuccess;
    }
    if (comm == &split_comm && !fake_split_queries_ready)
        return ncclInvalidUsage;
    *rank = comm->rank;
    return ncclSuccess;
}

ncclResult_t ncclCommCount(ncclComm_t comm, int *nranks) {
    if ((uintptr_t)comm <= 2) {
        *nranks = 2;
        return ncclSuccess;
    }
    if (comm == &split_comm && !fake_split_queries_ready)
        return ncclInvalidUsage;
    *nranks = comm->nranks;
    return ncclSuccess;
}

const char *ncclGetErrorString(ncclResult_t result) {
    (void)result;
    return "stub";
}

void init_rank_info(ncclComm_t comm) {
    if (g_my_rank == -1) {
        g_comm = comm;
        ncclCommUserRank(comm, &g_my_rank);
        ncclCommCount(comm, &g_nranks);
    }
}
void add_stream(cudaStream_t stream) {
    (void)stream;
}
int dmtcp_kvdb_set(const char *key, int64_t val) {
    (void)key;
    (void)val;
    return 1;
}
int64_t dmtcp_kvdb_get(const char *key) {
    (void)key;
    return 0;
}

const char *cudaGetErrorString(cudaError_t e) {
    (void)e;
    return "stub";
}
cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind) {
    (void)kind;
    memcpy(dst, src, count);
    return cudaSuccess;
}

void inflight_mark_send(int src, int dst, size_t bytes) {
    (void)src;
    (void)dst;
    (void)bytes;
}

void inflight_mark_recv(int src, int dst, size_t bytes) {
    (void)src;
    (void)dst;
    (void)bytes;
}

drain_msg_t *drain_queue_peek(int source) {
    (void)source;
    return NULL;
}

void drain_queue_pop(int source) {
    (void)source;
}
