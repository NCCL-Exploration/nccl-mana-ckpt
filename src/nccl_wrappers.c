#define _GNU_SOURCE
#include <stdio.h>
#include <dlfcn.h>
#include <stdlib.h>
#include <nccl.h>
#include "nccl_wrappers.h"
#include "dmtcp_kvdb_c.h"

extern volatile int ckpt_pending; 

static ncclResult_t (*real_ncclSend)(const void*, size_t, ncclDataType_t, int, ncclComm_t, cudaStream_t) = NULL;
static ncclResult_t (*real_ncclRecv)(void*, size_t, ncclDataType_t, int, ncclComm_t, cudaStream_t) = NULL;
static ncclResult_t (*real_ncclAllReduce)(const void*, void*, size_t, ncclDataType_t, ncclRedOp_t, ncclComm_t, cudaStream_t) = NULL;

#define LOAD_SYM(name) if (!real_##name) real_##name = dlsym(RTLD_NEXT, #name)

// Key format: "src|dest"
static void build_msg_key(char *buf, size_t len, int src, int dest) {
    snprintf(buf, len, "%d|%d", src, dest);
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
    // Call plugin helpers
    init_rank_info(comm);
    add_stream(stream);

    // Collectives don't need point-to-point tracking for now, just stream tracking
    
    LOAD_SYM(ncclAllReduce);
    return real_ncclAllReduce(sendbuff, recvbuff, count, datatype, op, comm, stream);
}
