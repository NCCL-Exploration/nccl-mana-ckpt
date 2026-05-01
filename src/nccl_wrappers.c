#define _GNU_SOURCE
#include <stdio.h>
#include <dlfcn.h>
#include <stdlib.h>
#include <unistd.h>
#include <nccl.h>
#include <string.h>
#include "nccl_wrappers.h"
#include "inflight.h"
#include <string.h>


#define CUDA_CHECK(cmd)                                                        \
    do {                                                                       \
        cudaError_t e = cmd;                                                   \
        if (e != cudaSuccess) {                                                \
            printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__,               \
                   cudaGetErrorString(e));                                     \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

extern volatile int ckpt_pending;

/* group_entry_t is defined in nccl_wrappers.h */
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

static void prepare_collective(ncclComm_t comm, cudaStream_t stream);

ncclResult_t call_real_ncclSend(const void* sendbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm, cudaStream_t stream) {
    LOAD_SYM(ncclSend);
    return real_ncclSend(sendbuff, count, datatype, peer, comm, stream);
}

ncclResult_t call_real_ncclRecv(void* recvbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm, cudaStream_t stream) {
    LOAD_SYM(ncclRecv);
    return real_ncclRecv(recvbuff, count, datatype, peer, comm, stream);
}

group_entry_t* get_or_create_group(ncclComm_t comm) {
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


void increment_seq_and_wait(ncclComm_t comm) {
    // If a checkpoint is pending, we must not start a new collective
    // until the coordinator has decided on the global target sequence.
    // This rank might be fast and reach the next sequence while others are lagging.
    group_entry_t* g = get_or_create_group(comm);
    if (g == NULL) {
        fprintf(stderr, "[rank %d] get_or_create_group returned NULL\n", g_my_rank);
        abort();
    }
    while (ckpt_pending && g->target != 0 && g->seq >= g->target) {
        for (int r = 0; r < g_nranks; r++){
            if(r == g_my_rank) continue;
            char key[64];
            snprintf(key, sizeof(key), "target|%p|%d", (void*)comm, r);
            int64_t new_target = dmtcp_kvdb_get(key);
            if (new_target > g->target) g->target = new_target;
        }
        usleep(100);
    }
    g->seq++;

    if (ckpt_pending && g->seq > g->target){
        g->target = g->seq;
        char key[64];
        snprintf(key, sizeof(key), "target|%p|%d", (void*)comm, g_my_rank);
        dmtcp_kvdb_set(key, (int64_t)g->target);
    }
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

    // Local tracking — no coordinator I/O on the hot path
    size_t bytes = count * get_nccl_type_size(datatype);
    inflight_mark_send(g_my_rank, peer, bytes);

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
    // Fast-path: replay a message that was drained at checkpoint time
    drain_msg_t *drained = drain_queue_peek(peer);
    if (drained != NULL) {
        size_t bytes = count * get_nccl_type_size(datatype);
        size_t copy_bytes = (drained->bytes < bytes) ? drained->bytes : bytes;
        CUDA_CHECK(cudaMemcpy(recvbuff, drained->data, copy_bytes, cudaMemcpyHostToDevice));
        drain_queue_pop(peer);  // frees drained->data
        return ncclSuccess;
    }

    // Normal path — local tracking, no coordinator I/O
    init_rank_info(comm);
    add_stream(stream);
    inflight_mark_recv(peer, g_my_rank, count * get_nccl_type_size(datatype));

    LOAD_SYM(ncclRecv);
    return real_ncclRecv(recvbuff, count, datatype, peer, comm, stream);
}

static void prepare_collective(ncclComm_t comm, cudaStream_t stream){
    init_rank_info(comm);
    add_stream(stream);
    increment_seq_and_wait(comm);
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
    prepare_collective(comm, stream);
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
    prepare_collective(comm, stream);
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
    prepare_collective(comm, stream);
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
    prepare_collective(comm, stream);
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
    prepare_collective(comm, stream);
    LOAD_SYM(ncclGather);
    return real_ncclGather(sendbuff, recvbuff, count, datatype, root, comm, stream);
}

ncclResult_t ncclGather_p2p(
    const void       *sendbuff,
    void             *recvbuff,
    size_t            sendcount,
    ncclDataType_t    datatype,
    int               root,
    ncclComm_t        comm,
    cudaStream_t      stream)
{
    int rank, nranks;
    ncclResult_t ret;

    if ((ret = ncclCommUserRank(comm, &rank))   != ncclSuccess) return ret;
    if ((ret = ncclCommCount(comm, &nranks))    != ncclSuccess) return ret;

    size_t elem_bytes = get_nccl_type_size(datatype);
    if (elem_bytes == 0) return ncclInvalidArgument;

    /*
     * Open a group so NCCL can schedule all point-to-point ops together.
     * This is mandatory when driving multiple send/recv from one thread
     * to prevent deadlock on rendezvous-protocol transports.
     */
    if ((ret = ncclGroupStart()) != ncclSuccess) return ret;

    if (rank == root) {
        /* Root posts a Recv slot for every rank, including itself. */
        for (int r = 0; r < nranks; r++) {
            char *slot = (char *)recvbuff + (size_t)r * sendcount * elem_bytes;
            ret = ncclRecv(slot, sendcount, datatype, r, comm, stream);
            if (ret != ncclSuccess) {
                ncclGroupEnd(); /* best-effort cleanup */
                return ret;
            }
        }
    }

    /* Every rank (including root) sends its contribution. */
    ret = ncclSend(sendbuff, sendcount, datatype, root, comm, stream);
    if (ret != ncclSuccess) {
        ncclGroupEnd();
        return ret;
    }

    return ncclGroupEnd();
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
    prepare_collective(comm, stream);
    LOAD_SYM(ncclReduceScatter);
    return real_ncclReduceScatter(sendbuff, recvbuff, recvcount, datatype, op, comm, stream);
}

void reset_groups() {
    memset(groups, 0, sizeof(groups));
}
