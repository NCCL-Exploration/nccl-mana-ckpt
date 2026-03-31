#define _GNU_SOURCE
#include <stdio.h>
#include <dlfcn.h>
#include <stdlib.h>
#include <unistd.h>
#include <nccl.h>
#include "nccl_wrappers.h"
#include "dmtcp_kvdb_c.h"

extern volatile int ckpt_pending; 
typedef struct{
    ncclComm_t comm;
    uint64_t seq;
    uint64_t target;
} group_entry_t;
// Might have to make it dynamically allocated at some point
#define NUM_GROUPS 5 
static group_entry_t groups[NUM_GROUPS]; // We need to change it to a Set at some point


static ncclResult_t (*real_ncclSend)(const void*, size_t, ncclDataType_t, int, ncclComm_t, cudaStream_t) = NULL;
static ncclResult_t (*real_ncclRecv)(void*, size_t, ncclDataType_t, int, ncclComm_t, cudaStream_t) = NULL;
static ncclResult_t (*real_ncclAllReduce)(const void*, void*, size_t, ncclDataType_t, ncclRedOp_t, ncclComm_t, cudaStream_t) = NULL;
static ncclResult_t (*real_ncclBroadcast)(const void*, void*, size_t, ncclDataType_t, int, ncclComm_t, cudaStream_t) = NULL;
static ncclResult_t (*real_ncclReduce)(const void*, void*, size_t, ncclDataType_t, ncclRedOp_t, int, ncclComm_t, cudaStream_t) = NULL;
static ncclResult_t (*real_ncclAllGather)(const void*, void*, size_t, ncclDataType_t, ncclComm_t, cudaStream_t) = NULL;
static ncclResult_t (*real_ncclGather)(const void*, void*, size_t, ncclDataType_t, int, ncclComm_t, cudaStream_t) = NULL;
static ncclResult_t (*real_ncclReduceScatter)(const void*, void*, size_t, ncclDataType_t, ncclRedOp_t, ncclComm_t, cudaStream_t) = NULL;
static ncclResult_t (*real_ncclCommInitRank)(ncclComm_t*, int, ncclUniqueId, int) = NULL;
static ncclResult_t (*real_ncclCommInitAll)(ncclComm_t*, int, const int*) = NULL;
static ncclResult_t (*real_ncclScatter)(const void*, void*, size_t, ncclDataType_t, int, ncclComm_t, cudaStream_t) = NULL;

#define LOAD_SYM(name) if (!real_##name) real_##name = dlsym(RTLD_NEXT, #name)

static group_entry_t* get_or_create_group(ncclComm_t comm) {
    for (int i = 0; i < NUM_GROUPS; i++) {
        if (groups[i].comm == comm) return &groups[i];
        if (groups[i].comm == NULL) {
            groups[i].comm = comm;
            groups[i].seq = 0;
            groups[i].target = 0;
            return &groups[i];
        }
    }
    return NULL; // Means list is full
}

static size_t get_nccl_type_size(ncclDataType_t type) {
    switch (type) {
        case ncclInt8:
        case ncclUint8:
            return 1;
        case ncclHalf:
            return 2;
        case ncclInt32:
        case ncclUint32:
        case ncclFloat32:
            return 4;
        case ncclInt64:
        case ncclUint64:
        case ncclFloat64:
            return 8;
        default:
            return 4;
    }
}

// Key format: "src|dest"
static void build_msg_key(char *buf, size_t len, int src, int dest) {
    snprintf(buf, len, "%d|%d", src, dest);
}

static void increment_seq_and_wait(ncclComm_t comm) {
    // If a checkpoint is pending, we must not start a new collective
    // until the coordinator has decided on the global target sequence.
    // This rank might be fast and reach the next sequence while others are lagging.
    group_entry_t* g = get_or_create_group(comm);
    while (ckpt_pending && g->target != 0 && g->seq >= g->target) {
        usleep(100);
    }
    g->seq++;
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

    char key[64];
    build_msg_key(key, sizeof(key), g_my_rank, peer);
    size_t bytes = count * get_nccl_type_size(datatype);

    if (!dmtcp_kvdb_set(key, (int64_t)bytes)) {
        fprintf(stderr, "[NCCL-MANA] Failed to set pending message size for %s\n", key);
        abort();
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

    char key[64];
    build_msg_key(key, sizeof(key), peer, g_my_rank);
    size_t bytes = count * get_nccl_type_size(datatype);

    if (!dmtcp_kvdb_set(key, 0)) {
        fprintf(stderr, "[NCCL-MANA] Failed to unset pending message size for %s\n", key);
        abort();
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
    increment_seq_and_wait(comm);

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
    increment_seq_and_wait(comm);

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
    increment_seq_and_wait(comm);

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
    increment_seq_and_wait(comm);

    LOAD_SYM(ncclAllGather);
    return real_ncclAllGather(sendbuff, recvbuff, sendcount, datatype, comm, stream);
}

ncclResult_t ncclGather(
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
    increment_seq_and_wait(comm);

    LOAD_SYM(ncclGather);
    return real_ncclGather(sendbuff, recvbuff, count, datatype, root, comm, stream);
}

ncclResult_t ncclScatter(
    const void* sendbuff,
    void* recvbuff,
    size_t sendcount,
    ncclDataType_t datatype,
    int root,
    ncclComm_t comm,
    cudaStream_t stream
) {
    init_rank_info(comm);
    add_stream(stream);
    increment_seq_and_wait(comm);

    LOAD_SYM(ncclScatter);
    return real_ncclScatter(sendbuff, recvbuff, sendcount, datatype, root, comm, stream);
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
    increment_seq_and_wait(comm);

    LOAD_SYM(ncclReduceScatter);
    return real_ncclReduceScatter(sendbuff, recvbuff, recvcount, datatype, op, comm, stream);
}
