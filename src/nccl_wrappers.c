#define _GNU_SOURCE
#include "nccl_wrappers.h"
#include "dmtcp.h"
#include "dmtcp_kvdb_c.h"
#include "inflight.h"
#include <dlfcn.h>
#include <nccl.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define CUDA_CHECK(cmd)                                                                            \
    do {                                                                                           \
        cudaError_t e = cmd;                                                                       \
        if (e != cudaSuccess) {                                                                    \
            printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e));           \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)

extern volatile int ckpt_pending;

/* group_entry_t is defined in nccl_wrappers.h. */
static group_entry_t *groups;
static pthread_mutex_t groups_lock = PTHREAD_MUTEX_INITIALIZER;
static ncclComm_t stale_primary_comm;

static ncclResult_t (*real_ncclSend)(const void *, size_t, ncclDataType_t, int, ncclComm_t,
                                     cudaStream_t) = NULL;
static ncclResult_t (*real_ncclRecv)(void *, size_t, ncclDataType_t, int, ncclComm_t,
                                     cudaStream_t) = NULL;
static ncclResult_t (*real_ncclAllReduce)(const void *, void *, size_t, ncclDataType_t, ncclRedOp_t,
                                          ncclComm_t, cudaStream_t) = NULL;
static ncclResult_t (*real_ncclBroadcast)(const void *, void *, size_t, ncclDataType_t, int,
                                          ncclComm_t, cudaStream_t) = NULL;
static ncclResult_t (*real_ncclReduce)(const void *, void *, size_t, ncclDataType_t, ncclRedOp_t,
                                       int, ncclComm_t, cudaStream_t) = NULL;
static ncclResult_t (*real_ncclAllGather)(const void *, void *, size_t, ncclDataType_t, ncclComm_t,
                                          cudaStream_t) = NULL;
static ncclResult_t (*real_ncclGather)(const void *, void *, size_t, ncclDataType_t, int,
                                       ncclComm_t, cudaStream_t) = NULL;
static ncclResult_t (*real_ncclReduceScatter)(const void *, void *, size_t, ncclDataType_t,
                                              ncclRedOp_t, ncclComm_t, cudaStream_t) = NULL;
static ncclResult_t (*real_ncclCommInitRank)(ncclComm_t *, int, ncclUniqueId, int) = NULL;
static ncclResult_t (*real_ncclCommInitRankConfig)(ncclComm_t *, int, ncclUniqueId, int,
                                                   ncclConfig_t *) = NULL;
static ncclResult_t (*real_ncclCommSplit)(ncclComm_t, int, int, ncclComm_t *, ncclConfig_t *) =
    NULL;
static ncclResult_t (*real_ncclCommFinalize)(ncclComm_t) = NULL;
static ncclResult_t (*real_ncclCommAbort)(ncclComm_t) = NULL;
static ncclResult_t (*real_ncclCommInitAll)(ncclComm_t *, int, const int *) = NULL;
static ncclResult_t (*real_ncclCommDestroy)(ncclComm_t) = NULL;
static ncclResult_t (*real_ncclScatter)(const void *, void *, size_t, ncclDataType_t, int,
                                        ncclComm_t, cudaStream_t) = NULL;

#define LOAD_SYM(name)                                                                             \
    if (!real_##name)                                                                              \
    real_##name = dlsym(RTLD_NEXT, #name)

static void prepare_collective(ncclComm_t comm, cudaStream_t stream);

static void register_primary_comm(ncclComm_t comm, int rank, int nranks) {
    if (g_comm != NULL)
        return;
    g_comm = comm;
    g_my_rank = rank;
    g_nranks = nranks;
    printf("[NCCL Drain] Rank %d of %d\n", g_my_rank, g_nranks);
}

ncclResult_t call_real_ncclSend(const void *sendbuff, size_t count, ncclDataType_t datatype,
                                int peer, ncclComm_t comm, cudaStream_t stream) {
    LOAD_SYM(ncclSend);
    return real_ncclSend(sendbuff, count, datatype, peer, comm, stream);
}

ncclResult_t call_real_ncclRecv(void *recvbuff, size_t count, ncclDataType_t datatype, int peer,
                                ncclComm_t comm, cudaStream_t stream) {
    LOAD_SYM(ncclRecv);
    return real_ncclRecv(recvbuff, count, datatype, peer, comm, stream);
}

static group_entry_t *find_group_locked(ncclComm_t comm) {
    for (group_entry_t *group = groups; group != NULL; group = group->next) {
        if (group->comm == comm)
            return group;
    }
    return NULL;
}

group_entry_t *find_group(ncclComm_t comm) {
    pthread_mutex_lock(&groups_lock);
    group_entry_t *group = find_group_locked(comm);
    pthread_mutex_unlock(&groups_lock);
    return group;
}

static group_entry_t *register_group(ncclComm_t comm, int rank, int nranks) {
    if (comm == NULL)
        return NULL;

    group_entry_t *new_group = calloc(1, sizeof(*new_group));
    if (new_group == NULL) {
        fprintf(stderr, "[WRAPPER] Failed to allocate state for comm %p\n", (void *)comm);
        return NULL;
    }
    new_group->comm = comm;
    new_group->rank = rank;
    new_group->nranks = nranks;
    new_group->target = -1;

    pthread_mutex_lock(&groups_lock);
    group_entry_t *group = find_group_locked(comm);
    if (group == NULL) {
        new_group->next = groups;
        groups = new_group;
        group = new_group;
        new_group = NULL;
    } else if (group->rank < 0 && rank >= 0) {
        group->rank = rank;
        group->nranks = nranks;
    }
    pthread_mutex_unlock(&groups_lock);

    bool inserted = new_group == NULL;
    free(new_group);

    if (inserted) {
        printf("[rank %d] Initialized sequence number to 0 for comm %p\n", group->rank,
               (void *)comm);
        fflush(stdout);
    }
    return group;
}

group_entry_t *get_or_create_group(ncclComm_t comm) {
    group_entry_t *group = find_group(comm);
    if (group != NULL && group->rank >= 0)
        return group;
    bool pending = group != NULL;

    int rank;
    int nranks;
    if (comm == NULL || ncclCommUserRank(comm, &rank) != ncclSuccess ||
        ncclCommCount(comm, &nranks) != ncclSuccess) {
        if (!pending)
            fprintf(stderr, "[WRAPPER] Failed to query rank information for comm %p\n",
                    (void *)comm);
        return NULL;
    }
    return register_group(comm, rank, nranks);
}

void remove_group(ncclComm_t comm) {
    pthread_mutex_lock(&groups_lock);
    group_entry_t **link = &groups;
    while (*link != NULL) {
        if ((*link)->comm == comm) {
            group_entry_t *removed = *link;
            *link = removed->next;
            pthread_mutex_unlock(&groups_lock);
            free(removed);
            return;
        }
        link = &(*link)->next;
    }
    pthread_mutex_unlock(&groups_lock);
}

static ncclComm_t resolve_lifecycle_comm(ncclComm_t comm) {
    if (find_group(comm) != NULL)
        return comm;
    if (comm == stale_primary_comm && g_comm != NULL)
        return g_comm;
    return comm;
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
    group_entry_t *g = get_or_create_group(comm);
    if (g == NULL) {
        fprintf(stderr, "[rank %d] get_or_create_group returned NULL\n", g_my_rank);
        abort();
    }
    for (;;) {
        DMTCP_PLUGIN_DISABLE_CKPT();
        if (!(ckpt_pending && g->seq >= g->target)) {
            DMTCP_PLUGIN_ENABLE_CKPT();
            break; // re-checked under protection; restart sees ckpt_pending==0 and exits
        }
        int64_t max_target = g->target;
        for (int r = 0; r < g->nranks; r++) {
            if (r == g->rank)
                continue;
            char key[64];
            snprintf(key, sizeof(key), "target|%p|%d", (void *)comm, r);
            int64_t new_target = dmtcp_kvdb_get(key);
            if (new_target > max_target)
                max_target = new_target;
        }
        g->target = max_target;
        DMTCP_PLUGIN_ENABLE_CKPT();
        usleep(100);
    }
    int64_t old_seq = g->seq;
    g->seq++;
    printf("[rank %d] Sequence number changed from %ld to %ld for comm %p\n", g->rank,
           (long)old_seq, (long)g->seq, (void *)comm);
    fflush(stdout);

    if (ckpt_pending && g->seq > g->target) {
        g->target = g->seq;
        char key[64];
        snprintf(key, sizeof(key), "target|%p|%d", (void *)comm, g->rank);
        dmtcp_kvdb_set(key, (int64_t)g->target);
    }
}

ncclResult_t ncclCommInitRank(ncclComm_t *comm, int nranks, ncclUniqueId commId, int rank) {
    fprintf(stderr, "[WRAPPER] ncclCommInitRank called with rank %d, nranks %d\n", rank, nranks);
    DMTCP_PLUGIN_DISABLE_CKPT();
    LOAD_SYM(ncclCommInitRank);
    ncclResult_t res = real_ncclCommInitRank(comm, nranks, commId, rank);
    fprintf(stderr, "[WRAPPER] real_ncclCommInitRank returned %d\n", res);
    if (res == ncclSuccess) {
        register_primary_comm(*comm, rank, nranks);
        register_group(*comm, rank, nranks);
    }
    DMTCP_PLUGIN_ENABLE_CKPT();
    return res;
}

ncclResult_t ncclCommInitRankConfig(ncclComm_t *comm, int nranks, ncclUniqueId commId, int rank,
                                    ncclConfig_t *config) {
    DMTCP_PLUGIN_DISABLE_CKPT();
    LOAD_SYM(ncclCommInitRankConfig);
    if (real_ncclCommInitRankConfig == NULL) {
        DMTCP_PLUGIN_ENABLE_CKPT();
        return ncclSystemError;
    }
    ncclResult_t res = real_ncclCommInitRankConfig(comm, nranks, commId, rank, config);
    if (comm != NULL && *comm != NULL) {
        register_group(*comm, rank, nranks);
        if (res == ncclSuccess || res == ncclInProgress)
            register_primary_comm(*comm, rank, nranks);
    }
    DMTCP_PLUGIN_ENABLE_CKPT();
    return res;
}

ncclResult_t ncclCommSplit(ncclComm_t comm, int color, int key, ncclComm_t *newcomm,
                           ncclConfig_t *config) {
    DMTCP_PLUGIN_DISABLE_CKPT();
    LOAD_SYM(ncclCommSplit);
    if (real_ncclCommSplit == NULL) {
        DMTCP_PLUGIN_ENABLE_CKPT();
        return ncclSystemError;
    }
    ncclResult_t res = real_ncclCommSplit(comm, color, key, newcomm, config);
    if (newcomm != NULL && *newcomm != NULL) {
        register_group(*newcomm, -1, -1);
        if (res == ncclSuccess)
            get_or_create_group(*newcomm);
    }
    DMTCP_PLUGIN_ENABLE_CKPT();
    return res;
}

ncclResult_t ncclCommFinalize(ncclComm_t comm) {
    DMTCP_PLUGIN_DISABLE_CKPT();
    LOAD_SYM(ncclCommFinalize);
    if (real_ncclCommFinalize == NULL) {
        DMTCP_PLUGIN_ENABLE_CKPT();
        return ncclSystemError;
    }
    ncclComm_t live = resolve_lifecycle_comm(comm);
    ncclResult_t res = real_ncclCommFinalize(live);
    DMTCP_PLUGIN_ENABLE_CKPT();
    return res;
}

ncclResult_t ncclCommAbort(ncclComm_t comm) {
    DMTCP_PLUGIN_DISABLE_CKPT();
    LOAD_SYM(ncclCommAbort);
    if (real_ncclCommAbort == NULL) {
        DMTCP_PLUGIN_ENABLE_CKPT();
        return ncclSystemError;
    }
    ncclComm_t live = resolve_lifecycle_comm(comm);
    ncclResult_t res = real_ncclCommAbort(live);
    if (res == ncclSuccess) {
        remove_group(live);
        if (live == g_comm)
            g_comm = NULL;
    }
    DMTCP_PLUGIN_ENABLE_CKPT();
    return res;
}

ncclResult_t ncclCommInitAll(ncclComm_t *comm, int nGpus, const int *devList) {
    DMTCP_PLUGIN_DISABLE_CKPT();
    LOAD_SYM(ncclCommInitAll);
    ncclResult_t res = real_ncclCommInitAll(comm, nGpus, devList);
    if (res == ncclSuccess) {
        for (int i = 0; i < nGpus; i++)
            register_group(comm[i], i, nGpus);
        if (nGpus > 0)
            register_primary_comm(comm[0], 0, nGpus);
    }
    DMTCP_PLUGIN_ENABLE_CKPT();
    return res;
}

ncclResult_t ncclSend(const void *sendbuff, size_t count, ncclDataType_t datatype, int peer,
                      ncclComm_t comm, cudaStream_t stream) {
    DMTCP_PLUGIN_DISABLE_CKPT();
    init_rank_info(comm);
    group_entry_t *group = get_or_create_group(comm);
    if (group == NULL) {
        DMTCP_PLUGIN_ENABLE_CKPT();
        return ncclSystemError;
    }
    add_stream(stream);

    // Local tracking — no coordinator I/O on the hot path
    size_t bytes = count * get_nccl_type_size(datatype);
    inflight_mark_send(group->rank, peer, bytes);

    LOAD_SYM(ncclSend);
    int ret = real_ncclSend(sendbuff, count, datatype, peer, comm, stream);
    DMTCP_PLUGIN_ENABLE_CKPT();
    return ret;
}

ncclResult_t ncclRecv(void *recvbuff, size_t count, ncclDataType_t datatype, int peer,
                      ncclComm_t comm, cudaStream_t stream) {
    DMTCP_PLUGIN_DISABLE_CKPT();
    // Fast-path: replay a message that was drained at checkpoint time
    drain_msg_t *drained = comm == g_comm ? drain_queue_peek(peer) : NULL;
    if (drained != NULL) {
        size_t bytes = count * get_nccl_type_size(datatype);
        size_t copy_bytes = (drained->bytes < bytes) ? drained->bytes : bytes;
        CUDA_CHECK(cudaMemcpy(recvbuff, drained->data, copy_bytes, cudaMemcpyHostToDevice));
        drain_queue_pop(peer); // frees drained->data
        DMTCP_PLUGIN_ENABLE_CKPT();
        return ncclSuccess;
    }

    // Normal path — local tracking, no coordinator I/O
    init_rank_info(comm);
    group_entry_t *group = get_or_create_group(comm);
    if (group == NULL) {
        DMTCP_PLUGIN_ENABLE_CKPT();
        return ncclSystemError;
    }
    add_stream(stream);

    inflight_mark_recv(peer, group->rank, count * get_nccl_type_size(datatype));

    LOAD_SYM(ncclRecv);
    int ret = real_ncclRecv(recvbuff, count, datatype, peer, comm, stream);
    DMTCP_PLUGIN_ENABLE_CKPT();
    return ret;
}

static void prepare_collective(ncclComm_t comm, cudaStream_t stream) {
    init_rank_info(comm);
    add_stream(stream);
    increment_seq_and_wait(comm);
}

ncclResult_t ncclAllReduce(const void *sendbuff, void *recvbuff, size_t count,
                           ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm,
                           cudaStream_t stream) {
    prepare_collective(comm, stream);
    DMTCP_PLUGIN_DISABLE_CKPT();
    LOAD_SYM(ncclAllReduce);
    ncclResult_t ret = real_ncclAllReduce(sendbuff, recvbuff, count, datatype, op, comm, stream);
    DMTCP_PLUGIN_ENABLE_CKPT();
    return ret;
}

ncclResult_t ncclBroadcast(const void *sendbuff, void *recvbuff, size_t count,
                           ncclDataType_t datatype, int root, ncclComm_t comm,
                           cudaStream_t stream) {
    prepare_collective(comm, stream);
    DMTCP_PLUGIN_DISABLE_CKPT();
    LOAD_SYM(ncclBroadcast);
    int ret = real_ncclBroadcast(sendbuff, recvbuff, count, datatype, root, comm, stream);
    DMTCP_PLUGIN_ENABLE_CKPT();
    return ret;
}

ncclResult_t ncclReduce(const void *sendbuff, void *recvbuff, size_t count, ncclDataType_t datatype,
                        ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream) {
    prepare_collective(comm, stream);
    DMTCP_PLUGIN_DISABLE_CKPT();
    LOAD_SYM(ncclReduce);
    int ret = real_ncclReduce(sendbuff, recvbuff, count, datatype, op, root, comm, stream);
    DMTCP_PLUGIN_ENABLE_CKPT();
    return ret;
}

ncclResult_t ncclAllGather(const void *sendbuff, void *recvbuff, size_t sendcount,
                           ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {
    prepare_collective(comm, stream);
    DMTCP_PLUGIN_DISABLE_CKPT();
    LOAD_SYM(ncclAllGather);
    int ret = real_ncclAllGather(sendbuff, recvbuff, sendcount, datatype, comm, stream);
    DMTCP_PLUGIN_ENABLE_CKPT();
    return ret;
}

ncclResult_t ncclGather(const void *sendbuff, void *recvbuff, size_t count, ncclDataType_t datatype,
                        int root, ncclComm_t comm, cudaStream_t stream) {
    prepare_collective(comm, stream);
    DMTCP_PLUGIN_DISABLE_CKPT();
    LOAD_SYM(ncclGather);
    int ret = real_ncclGather(sendbuff, recvbuff, count, datatype, root, comm, stream);
    DMTCP_PLUGIN_ENABLE_CKPT();
    return ret;
}

ncclResult_t ncclGather_p2p(const void *sendbuff, void *recvbuff, size_t sendcount,
                            ncclDataType_t datatype, int root, ncclComm_t comm,
                            cudaStream_t stream) {
    int rank, nranks;
    ncclResult_t ret;

    if ((ret = ncclCommUserRank(comm, &rank)) != ncclSuccess)
        return ret;
    if ((ret = ncclCommCount(comm, &nranks)) != ncclSuccess)
        return ret;

    size_t elem_bytes = get_nccl_type_size(datatype);
    if (elem_bytes == 0)
        return ncclInvalidArgument;
    if (root < 0 || root >= nranks)
        return ncclInvalidArgument;
    if (rank == root && recvbuff == NULL)
        return ncclInvalidArgument;

    /*
     * The caller owns ncclGroupStart/ncclGroupEnd.  This lets a single-thread
     * multi-GPU caller batch every rank's p2p gather work in one outer group.
     */
    if (rank == root) {
        /* Root posts a Recv slot for every rank, including itself. */
        for (int r = 0; r < nranks; r++) {
            char *slot = (char *)recvbuff + (size_t)r * sendcount * elem_bytes;
            ret = ncclRecv(slot, sendcount, datatype, r, comm, stream);
            if (ret != ncclSuccess) {
                return ret;
            }
        }
    }

    /* Every rank (including root) sends its contribution. */
    ret = ncclSend(sendbuff, sendcount, datatype, root, comm, stream);
    if (ret != ncclSuccess)
        return ret;

    return ncclSuccess;
}

ncclResult_t ncclScatter_p2p(const void *sendbuff, void *recvbuff, size_t sendcount,
                             ncclDataType_t datatype, int root, ncclComm_t comm,
                             cudaStream_t stream) {
    int rank, nranks;
    ncclResult_t ret;

    if ((ret = ncclCommUserRank(comm, &rank)) != ncclSuccess)
        return ret;
    if ((ret = ncclCommCount(comm, &nranks)) != ncclSuccess)
        return ret;

    size_t elem_bytes = get_nccl_type_size(datatype);
    if (elem_bytes == 0)
        return ncclInvalidArgument;
    if (root < 0 || root >= nranks)
        return ncclInvalidArgument;
    if (recvbuff == NULL)
        return ncclInvalidArgument;
    if (rank == root && sendbuff == NULL)
        return ncclInvalidArgument;

    /*
     * The caller owns ncclGroupStart/ncclGroupEnd.  This mirrors
     * ncclGather_p2p and lets a single-thread multi-GPU caller batch every
     * rank's p2p scatter work in one outer group.
     */
    ret = ncclRecv(recvbuff, sendcount, datatype, root, comm, stream);
    if (ret != ncclSuccess)
        return ret;

    if (rank == root) {
        /* Root sends slot r to rank r, including itself. */
        for (int r = 0; r < nranks; r++) {
            const char *slot = (const char *)sendbuff + (size_t)r * sendcount * elem_bytes;
            ret = ncclSend(slot, sendcount, datatype, r, comm, stream);
            if (ret != ncclSuccess) {
                return ret;
            }
        }
    }

    return ncclSuccess;
}

ncclResult_t ncclScatter(const void *sendbuff, void *recvbuff, size_t sendcount,
                         ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream) {
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

ncclResult_t ncclReduceScatter(const void *sendbuff, void *recvbuff, size_t recvcount,
                               ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm,
                               cudaStream_t stream) {
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
    pthread_mutex_lock(&groups_lock);
    group_entry_t *group = groups;
    groups = NULL;
    pthread_mutex_unlock(&groups_lock);
    while (group != NULL) {
        group_entry_t *next = group->next;
        free(group);
        group = next;
    }
}

/*
 * Interposed ncclCommDestroy. Resolves the app's (possibly stale, post-
 * restart) comm pointer to whatever communicator is actually live, destroys
 * that one, and clears g_comm if it was the tracked handle. Used both by
 * application code that calls ncclCommDestroy directly, and internally by
 * libncclplugin.c to tear down the communicator before checkpoint.
 */
ncclResult_t ncclCommDestroy(ncclComm_t comm) {
    ncclComm_t live = resolve_lifecycle_comm(comm);
    if (live == NULL) {
        return ncclSuccess;
    }
    LOAD_SYM(ncclCommDestroy);
    if (real_ncclCommDestroy == NULL)
        return ncclSystemError;
    ncclResult_t r = real_ncclCommDestroy(live);
    if (r == ncclSuccess) {
        remove_group(live);
        if (live == g_comm) {
            stale_primary_comm = comm;
            g_comm = NULL;
        }
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
    real_ncclCommInitRankConfig = NULL;
    real_ncclCommSplit = NULL;
    real_ncclCommFinalize = NULL;
    real_ncclCommAbort = NULL;
    real_ncclCommInitAll = NULL;
    real_ncclScatter = NULL;
    real_ncclCommDestroy = NULL;
}
