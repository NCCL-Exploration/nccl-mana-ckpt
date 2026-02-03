#define _GNU_SOURCE
#include "nccl_wrappers.h"
#include <stdio.h>
#include <dlfcn.h>

#include "jassert.h"
#include "kvdb.h"

using dmtcp::kvdb::KVDBRequest;
using dmtcp::kvdb::KVDBResponse;

extern int g_world_rank;
extern volatile bool ckpt_pending;

static const char *NCCL_DB = "/plugin/MANA/nccl-pending";

ncclSend_t real_ncclSend = NULL;
ncclRecv_t real_ncclRecv = NULL;

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
        // Mark pending: increment by 1
        JASSERT(dmtcp::kvdb::request64(KVDBRequest::INCRBY, NCCL_DB, key, 1)
                == KVDBResponse::SUCCESS);
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
        // Clear pending: decrement by 1
        JASSERT(dmtcp::kvdb::request64(KVDBRequest::INCRBY, NCCL_DB, key, -1)
                == KVDBResponse::SUCCESS);
    }

    init_real_ncclRecv();
    return real_ncclRecv(recvbuff, count, datatype, peer, comm, stream);
}
