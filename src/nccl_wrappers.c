#define _GNU_SOURCE
#include <stdio.h>
#include <dlfcn.h>
#include <stdlib.h>
#include <unistd.h>
#include <nccl.h>
#include <string.h>
#include "nccl_wrappers.h"
#include "dmtcp_kvdb_c.h"
#include "dmtcp.h"
#include "inflight.h"

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
static ncclResult_t (*real_ncclCommInitRankConfig)(ncclComm_t*, int, ncclUniqueId, int, ncclConfig_t*) = NULL;
static ncclResult_t (*real_ncclCommInitAll)(ncclComm_t*, int, const int*) = NULL;
static ncclResult_t (*real_ncclCommDestroy)(ncclComm_t) = NULL;
static ncclResult_t (*real_ncclScatter)(const void*, void*, size_t, ncclDataType_t, int, ncclComm_t, cudaStream_t) = NULL;
static ncclResult_t (*real_ncclCommUserRank)(ncclComm_t, int*) = NULL;
static ncclResult_t (*real_ncclCommCount)(ncclComm_t, int*) = NULL;
static ncclResult_t (*real_ncclGetUniqueId)(ncclUniqueId*) = NULL;
static const char* (*real_ncclGetErrorString)(ncclResult_t) = NULL;

static void* get_nccl_handle() {
    static void* handle = NULL;
    if (!handle) {
        handle = RTLD_NEXT;
        // Verify if RTLD_NEXT works by looking up a standard symbol
        if (!dlsym(handle, "ncclCommDestroy")) {
            // Fallback to loading the system libnccl.so dynamically
            handle = dlopen("libnccl.so", RTLD_LAZY | RTLD_GLOBAL);
            if (!handle) {
                fprintf(stderr, "[WRAPPER] Warning: Failed to dlopen libnccl.so: %s\n", dlerror());
                handle = RTLD_NEXT; // fallback to RTLD_NEXT anyway
            } else {
                fprintf(stderr, "[WRAPPER] Successfully loaded libnccl.so via dlopen\n");
            }
        }
    }
    return handle;
}

#define LOAD_SYM(name) \
    if (!real_##name) { \
        real_##name = dlsym(get_nccl_handle(), #name); \
        if (!real_##name) { \
            fprintf(stderr, "[WRAPPER] Error: Failed to resolve symbol %s: %s\n", #name, dlerror()); \
        } \
    }

static void prepare_collective(ncclComm_t comm, cudaStream_t stream);

ncclResult_t call_real_ncclSend(const void* sendbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm, cudaStream_t stream) {
    LOAD_SYM(ncclSend);
    return real_ncclSend(sendbuff, count, datatype, peer, comm, stream);
}

ncclResult_t call_real_ncclRecv(void* recvbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm, cudaStream_t stream) {
    LOAD_SYM(ncclRecv);
    return real_ncclRecv(recvbuff, count, datatype, peer, comm, stream);
}

ncclResult_t call_real_ncclCommUserRank(ncclComm_t comm, int* rank) {
    LOAD_SYM(ncclCommUserRank);
    if (real_ncclCommUserRank) return real_ncclCommUserRank(comm, rank);
    return ncclSuccess;
}

ncclResult_t call_real_ncclCommCount(ncclComm_t comm, int* count) {
    LOAD_SYM(ncclCommCount);
    if (real_ncclCommCount) return real_ncclCommCount(comm, count);
    return ncclSuccess;
}

ncclResult_t call_real_ncclGetUniqueId(ncclUniqueId* out) {
    LOAD_SYM(ncclGetUniqueId);
    if (real_ncclGetUniqueId) return real_ncclGetUniqueId(out);
    return ncclSuccess;
}

const char* call_real_ncclGetErrorString(ncclResult_t result) {
    LOAD_SYM(ncclGetErrorString);
    if (real_ncclGetErrorString) return real_ncclGetErrorString(result);
    return "Unknown NCCL Error";
}

group_entry_t* get_or_create_group(ncclComm_t comm) {
    for (int i = 0; i < NUM_GROUPS; i++) {
        if (groups[i].comm == comm) return &groups[i];
        if (groups[i].comm == NULL) {
            groups[i].comm = comm;
            groups[i].seq = 0;
            groups[i].target = -1;
            printf("[rank %d] Initialized sequence number to 0 for comm %p\n", g_my_rank, (void*)comm);
            fflush(stdout);
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
    for (;;) {
      DMTCP_PLUGIN_DISABLE_CKPT();
      if (!(ckpt_pending && g->seq >= g->target)) {
        DMTCP_PLUGIN_ENABLE_CKPT();
        break;                       // re-checked under protection; restart sees ckpt_pending==0 and exits
      }
      int64_t max_target = g->target;
      for (int r = 0; r < g_nranks; r++) {
        if (r == g_my_rank) continue;
        char key[64];
        snprintf(key, sizeof(key), "target|%p|%d", (void*)comm, r);
        int64_t new_target = dmtcp_kvdb_get(key);
        if (new_target > max_target) max_target = new_target;
      }
      g->target = max_target;
      DMTCP_PLUGIN_ENABLE_CKPT();
      usleep(100);
    }
    int64_t old_seq = g->seq;
    g->seq++;
    printf("[rank %d] Sequence number changed from %ld to %ld for comm %p\n", g_my_rank, (long)old_seq, (long)g->seq, (void*)comm);
    fflush(stdout);

    if (ckpt_pending && g->seq > g->target){
        g->target = g->seq;
        char key[64];
        snprintf(key, sizeof(key), "target|%p|%d", (void*)comm, g_my_rank);
        dmtcp_kvdb_set(key, (int64_t)g->target);
    }
}

ncclResult_t ncclCommInitRank(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank) {
    fprintf(stderr, "[WRAPPER] ncclCommInitRank called with rank %d, nranks %d\n", rank, nranks);
    DMTCP_PLUGIN_DISABLE_CKPT();
    LOAD_SYM(ncclCommInitRank);
    if (!real_ncclCommInitRank) {
        fprintf(stderr, "[WRAPPER] Error: real_ncclCommInitRank is NULL!\n");
        DMTCP_PLUGIN_ENABLE_CKPT();
        return ncclUnhandledCudaError;
    }
    ncclResult_t res = real_ncclCommInitRank(comm, nranks, commId, rank);
    fprintf(stderr, "[WRAPPER] real_ncclCommInitRank returned %d\n", res);
    if (res == ncclSuccess) {
        init_rank_info(*comm);
    }
    DMTCP_PLUGIN_ENABLE_CKPT();
    return res;
}

ncclResult_t ncclCommInitRankConfig(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank, ncclConfig_t* config) {
    fprintf(stderr, "[WRAPPER] ncclCommInitRankConfig called with rank %d, nranks %d\n", rank, nranks);
    DMTCP_PLUGIN_DISABLE_CKPT();
    LOAD_SYM(ncclCommInitRankConfig);
    if (!real_ncclCommInitRankConfig) {
        fprintf(stderr, "[WRAPPER] Error: real_ncclCommInitRankConfig is NULL!\n");
        DMTCP_PLUGIN_ENABLE_CKPT();
        return ncclUnhandledCudaError;
    }
    ncclResult_t res = real_ncclCommInitRankConfig(comm, nranks, commId, rank, config);
    if (res == ncclSuccess) {
        init_rank_info(*comm);
    }
    DMTCP_PLUGIN_ENABLE_CKPT();
    return res;
}

ncclResult_t ncclCommInitAll(ncclComm_t* comm, int nGpus, const int* devList) {
    DMTCP_PLUGIN_DISABLE_CKPT();
    LOAD_SYM(ncclCommInitAll);
    if (!real_ncclCommInitAll) {
        fprintf(stderr, "[WRAPPER] Error: real_ncclCommInitAll is NULL!\n");
        DMTCP_PLUGIN_ENABLE_CKPT();
        return ncclUnhandledCudaError;
    }
    ncclResult_t res = real_ncclCommInitAll(&g_comm, nGpus, devList);
    if (res == ncclSuccess) {
        init_rank_info(g_comm);
    }
    DMTCP_PLUGIN_ENABLE_CKPT();
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
    DMTCP_PLUGIN_DISABLE_CKPT();
    init_rank_info(comm);
    add_stream(stream);

    size_t bytes = count * get_nccl_type_size(datatype);
    inflight_mark_send(g_my_rank, peer, bytes);

    LOAD_SYM(ncclSend);
    int ret = real_ncclSend(sendbuff, count, datatype, peer, comm, stream);
    DMTCP_PLUGIN_ENABLE_CKPT();
    return ret;
}

ncclResult_t ncclRecv(
    void* recvbuff,
    size_t count,
    ncclDataType_t datatype,
    int peer,
    ncclComm_t comm,
    cudaStream_t stream
) {
    DMTCP_PLUGIN_DISABLE_CKPT();
    init_rank_info(comm);

    // Fast-path: replay a message that was drained at checkpoint time
    drain_msg_t *drained = drain_queue_peek(peer);
    if (drained != NULL) {
        size_t bytes = count * get_nccl_type_size(datatype);
        size_t copy_bytes = (drained->bytes < bytes) ? drained->bytes : bytes;
        CUDA_CHECK(cudaMemcpy(recvbuff, drained->data, copy_bytes, cudaMemcpyHostToDevice));
        drain_queue_pop(peer);  // frees drained->data
        DMTCP_PLUGIN_ENABLE_CKPT();
        return ncclSuccess;
    }

    add_stream(stream);

    int rank = g_my_rank;
    call_real_ncclCommUserRank(comm, &rank);
    inflight_mark_recv(peer, rank, count * get_nccl_type_size(datatype));

    LOAD_SYM(ncclRecv);
    int ret = real_ncclRecv(recvbuff, count, datatype, peer, comm, stream);
    DMTCP_PLUGIN_ENABLE_CKPT();
    return ret;
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
  DMTCP_PLUGIN_DISABLE_CKPT();
  LOAD_SYM(ncclAllReduce);
  ncclResult_t ret = real_ncclAllReduce(sendbuff, recvbuff, count, datatype, op, comm, stream);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret;
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
  DMTCP_PLUGIN_DISABLE_CKPT();
  LOAD_SYM(ncclBroadcast);
  int ret = real_ncclBroadcast(sendbuff, recvbuff, count, datatype, root, comm, stream);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret;
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
  DMTCP_PLUGIN_DISABLE_CKPT();
  LOAD_SYM(ncclReduce);
  int ret = real_ncclReduce(sendbuff, recvbuff, count, datatype, op, root, comm, stream);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret;
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
  DMTCP_PLUGIN_DISABLE_CKPT();
  LOAD_SYM(ncclAllGather);
  int ret = real_ncclAllGather(sendbuff, recvbuff, sendcount, datatype, comm, stream);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret;
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
  DMTCP_PLUGIN_DISABLE_CKPT();
  LOAD_SYM(ncclGather);
  int ret = real_ncclGather(sendbuff, recvbuff, count, datatype, root, comm, stream);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret;
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

  if ((ret = call_real_ncclCommUserRank(comm, &rank))   != ncclSuccess) return ret;
  if ((ret = call_real_ncclCommCount(comm, &nranks))    != ncclSuccess) return ret;

  size_t elem_bytes = get_nccl_type_size(datatype);
  if (elem_bytes == 0) return ncclInvalidArgument;
  if (root < 0 || root >= nranks) return ncclInvalidArgument;
  if (rank == root && recvbuff == NULL) return ncclInvalidArgument;

  if (rank == root) {
    for (int r = 0; r < nranks; r++) {
      char *slot = (char *)recvbuff + (size_t)r * sendcount * elem_bytes;
      ret = ncclRecv(slot, sendcount, datatype, r, comm, stream);
      if (ret != ncclSuccess) {
        return ret;
      }
    }
  }

  ret = ncclSend(sendbuff, sendcount, datatype, root, comm, stream);
  if (ret != ncclSuccess) return ret;

  return ncclSuccess;
}

ncclResult_t ncclScatter_p2p(
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

    if ((ret = call_real_ncclCommUserRank(comm, &rank)) != ncclSuccess) return ret;
    if ((ret = call_real_ncclCommCount(comm, &nranks))  != ncclSuccess) return ret;

    size_t elem_bytes = get_nccl_type_size(datatype);
    if (elem_bytes == 0) return ncclInvalidArgument;
    if (root < 0 || root >= nranks) return ncclInvalidArgument;
    if (recvbuff == NULL) return ncclInvalidArgument;
    if (rank == root && sendbuff == NULL) return ncclInvalidArgument;

    ret = ncclRecv(recvbuff, sendcount, datatype, root, comm, stream);
    if (ret != ncclSuccess) return ret;

    if (rank == root) {
        for (int r = 0; r < nranks; r++) {
            const char *slot =
                (const char *)sendbuff + (size_t)r * sendcount * elem_bytes;
            ret = ncclSend(slot, sendcount, datatype, r, comm, stream);
            if (ret != ncclSuccess) {
                return ret;
            }
        }
    }

    return ncclSuccess;
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
  prepare_collective(comm, stream);
  DMTCP_PLUGIN_DISABLE_CKPT();
  LOAD_SYM(ncclScatter);
  if (!real_ncclScatter) {
    fprintf(stderr, "[WRAPPER] ncclScatter is not available in this NCCL version\n");
    DMTCP_PLUGIN_ENABLE_CKPT();
    return ncclInvalidUsage;
  }
  int ret = real_ncclScatter(sendbuff, recvbuff, sendcount, datatype, root, comm, stream);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret;
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
  DMTCP_PLUGIN_DISABLE_CKPT();
  LOAD_SYM(ncclReduceScatter);
  int ret = real_ncclReduceScatter(sendbuff, recvbuff, recvcount, datatype, op, comm, stream);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret;
}

void reset_groups() {
    printf("[rank %d] reset_groups called, resetting sequence numbers of all groups\n", g_my_rank);
    fflush(stdout);
    memset(groups, 0, sizeof(groups));
}

/*
 * Interposed ncclCommDestroy. Resolves the app's (possibly stale, post-
 * restart) comm pointer to whatever communicator is actually live, destroys
 * that one, and clears g_comm if it was the tracked handle. Used both by
 * application code that calls ncclCommDestroy directly, and internally by
 * libncclplugin.c to tear down the communicator before checkpoint.
 */
ncclResult_t ncclCommDestroy(ncclComm_t comm) {
    ncclComm_t live = g_comm;
    if (live == NULL) {
        return ncclSuccess;
    }
    LOAD_SYM(ncclCommDestroy);
    if (!real_ncclCommDestroy) {
        fprintf(stderr, "[WRAPPER] Error: real_ncclCommDestroy is NULL!\n");
        return ncclUnhandledCudaError;
    }
    ncclResult_t r = real_ncclCommDestroy(live);
    if (live == g_comm) {
        g_comm = NULL;
    }
    return r;
}


void clear_wrapper_pointers() {
    real_ncclSend = NULL;
    real_ncclRecv = NULL;
    real_ncclAllReduce = NULL;
    real_ncclBroadcast = NULL;
    real_ncclReduce = NULL;
    real_ncclAllGather = NULL;
    real_ncclGather = NULL;
    real_ncclReduceScatter = NULL;
    real_ncclCommInitRank = NULL;
    real_ncclCommInitAll = NULL;
    real_ncclScatter = NULL;
    real_ncclCommDestroy = NULL;
    real_ncclCommUserRank = NULL;
    real_ncclCommCount = NULL;
    real_ncclGetUniqueId = NULL;
    real_ncclGetErrorString = NULL;
}
