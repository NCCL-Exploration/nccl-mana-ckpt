#define _GNU_SOURCE
#include <stdio.h>
#include <dlfcn.h>
#include <stdlib.h>
#include "nccl_wrappers.h"
#include "dmtcp_kvdb_c.h"

extern volatile int ckpt_pending; 
#include <stdbool.h>

extern int g_world_rank;

static ncclSend_t real_ncclSend = NULL;
static ncclRecv_t real_ncclRecv = NULL;

void init_real_ncclSend() {
    if (real_ncclSend == NULL) {
        real_ncclSend = (ncclSend_t)dlsym(RTLD_NEXT, "ncclSend");
    }
}

void init_real_ncclRecv() {
    if (real_ncclRecv == NULL) {
        real_ncclRecv = (ncclRecv_t)dlsym(RTLD_NEXT, "ncclRecv");
    }
}

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
    if (ckpt_pending) {
        char key[64];
        build_msg_key(key, sizeof(key), g_world_rank, peer);
        
        if (!dmtcp_kvdb_incr(key, 1)) {
             fprintf(stderr, "[NCCL-MANA] Failed to increment pending message count for %s\n", key);
             abort();
        }
    }

    init_real_ncclSend();
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
    if (ckpt_pending) {
        char key[64];
        build_msg_key(key, sizeof(key), peer, g_world_rank);
        if (!dmtcp_kvdb_incr(key, -1)) {
             fprintf(stderr, "[NCCL-MANA] Failed to decrement pending message count for %s\n", key);
             abort();
        }
    }

    init_real_ncclRecv();
    return real_ncclRecv(recvbuff, count, datatype, peer, comm, stream);
}
