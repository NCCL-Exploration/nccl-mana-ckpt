#define _GNU_SOURCE
#include <stdio.h>
#include <dlfcn.h>
#include <stdlib.h>
#include <nccl.h>
#include "nccl_wrappers.h"
#include "dmtcp_kvdb_c.h"

extern volatile int ckpt_pending; 
uint64_t g_seq_num = 0;
uint64_t g_target_seq = 0;

static ncclResult_t (*real_ncclSend)(const void*, size_t, ncclDataType_t, int, ncclComm_t, cudaStream_t) = NULL;
static ncclResult_t (*real_ncclRecv)(void*, size_t, ncclDataType_t, int, ncclComm_t, cudaStream_t) = NULL;
static ncclResult_t (*real_ncclAllReduce)(const void*, void*, size_t, ncclDataType_t, ncclRedOp_t, ncclComm_t, cudaStream_t) = NULL;
static ncclResult_t (*real_ncclBroadcast)(const void*, void*, size_t, ncclDataType_t, int, ncclComm_t, cudaStream_t) = NULL;
static ncclResult_t (*real_ncclReduce)(const void*, void*, size_t, ncclDataType_t, ncclRedOp_t, int, ncclComm_t, cudaStream_t) = NULL;
static ncclResult_t (*real_ncclAllGather)(const void*, void*, size_t, ncclDataType_t, ncclComm_t, cudaStream_t) = NULL;
static ncclResult_t (*real_ncclReduceScatter)(const void*, void*, size_t, ncclDataType_t, ncclRedOp_t, ncclComm_t, cudaStream_t) = NULL;
static ncclResult_t (*real_ncclCommInitRank)(ncclComm_t*, int, ncclUniqueId, int) = NULL;
static ncclResult_t (*real_ncclCommInitAll)(ncclComm_t*, int, const int*) = NULL;

#define LOAD_SYM(name) if (!real_##name) real_##name = dlsym(RTLD_NEXT, #name)

// Key format: "src|dest"
static void build_msg_key(char *buf, size_t len, int src, int dest) {
    snprintf(buf, len, "%d|%d", src, dest);
}

static void increment_seq_and_wait() {
    // If a checkpoint is pending, we must not start a new collective
    // until the coordinator has decided on the global target sequence.
    // This rank might be fast and reach the next sequence while others are lagging.
    while (ckpt_pending && g_target_seq != 0 && g_seq_num >= g_target_seq) {
        usleep(100);
    }
    g_seq_num++;
}

ncclResult_t ncclCommInitRank(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank) {
    LOAD_SYM(ncclCommInitRank);
    ncclResult_t res = real_ncclCommInitRank(comm, nranks, commId, rank);
    if (res == ncclSuccess) {
        init_rank_info(*comm);
    }
    return res;
}

ncclResult_t ncclCommInitAll(ncclComm_t* comm, int nGpus, const int* devList) {
    LOAD_SYM(ncclCommInitAll);
    ncclResult_t res = real_ncclCommInitAll(comm, nGpus, devList);
    if (res == ncclSuccess) {
        init_rank_info(*comm);
    }
    return res;
}

ncclResult_t ncclSend(
    const void* sendbuff,
    size_t count,
    ncclDataType_t datatype,
    int peer,
    ncclComm_t comm,
    cudaStream_t stream
) {
    // Call plugin helpers to track state
    init_rank_info(comm);
    add_stream(stream);

    if (ckpt_pending) {
        char key[64];
        build_msg_key(key, sizeof(key), g_my_rank, peer);
        
        if (!dmtcp_kvdb_incr(key, 1)) {
             fprintf(stderr, "[NCCL-MANA] Failed to increment pending message count for %s\n", key);
             abort();
        }
    }

    LOAD_SYM(ncclSend);
    return real_ncclSend(sendbuff, count, datatype, peer, comm, stream);
}

ncclResult_t ncclRecv(
    void* recvbuff,
    size_t count,
    ncclDataType_t datatype,
    int peer,
    ncclComm_t comm,
    cudaStream_t stream
) {
    // Call plugin helpers
    init_rank_info(comm);
    add_stream(stream);

    if (ckpt_pending) {
        char key[64];
        build_msg_key(key, sizeof(key), peer, g_my_rank);
        if (!dmtcp_kvdb_incr(key, -1)) {
             fprintf(stderr, "[NCCL-MANA] Failed to decrement pending message count for %s\n", key);
             abort();
        }
    }

    LOAD_SYM(ncclRecv);
    return real_ncclRecv(recvbuff, count, datatype, peer, comm, stream);
}

ncclResult_t ncclAllReduce(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    ncclComm_t comm,
    cudaStream_t stream
) {
    init_rank_info(comm);
    add_stream(stream);
    increment_seq_and_wait();

    LOAD_SYM(ncclAllReduce);
    return real_ncclAllReduce(sendbuff, recvbuff, count, datatype, op, comm, stream);
}

ncclResult_t ncclBroadcast(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    ncclDataType_t datatype,
    int root,
    ncclComm_t comm,
    cudaStream_t stream
) {
    init_rank_info(comm);
    add_stream(stream);
    increment_seq_and_wait();

    LOAD_SYM(ncclBroadcast);
    return real_ncclBroadcast(sendbuff, recvbuff, count, datatype, root, comm, stream);
}

ncclResult_t ncclReduce(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    int root,
    ncclComm_t comm,
    cudaStream_t stream
) {
    init_rank_info(comm);
    add_stream(stream);
    increment_seq_and_wait();

    LOAD_SYM(ncclReduce);
    return real_ncclReduce(sendbuff, recvbuff, count, datatype, op, root, comm, stream);
}

ncclResult_t ncclAllGather(
    const void* sendbuff,
    void* recvbuff,
    size_t sendcount,
    ncclDataType_t datatype,
    ncclComm_t comm,
    cudaStream_t stream
) {
    init_rank_info(comm);
    add_stream(stream);
    increment_seq_and_wait();

    LOAD_SYM(ncclAllGather);
    return real_ncclAllGather(sendbuff, recvbuff, sendcount, datatype, comm, stream);
}

ncclResult_t ncclReduceScatter(
    const void* sendbuff,
    void* recvbuff,
    size_t recvcount,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    ncclComm_t comm,
    cudaStream_t stream
) {
    init_rank_info(comm);
    add_stream(stream);
    increment_seq_and_wait();

    LOAD_SYM(ncclReduceScatter);
    return real_ncclReduceScatter(sendbuff, recvbuff, recvcount, datatype, op, comm, stream);
}
