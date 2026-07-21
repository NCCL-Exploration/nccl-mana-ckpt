#ifndef NCCL_WRAPPERS_H
#define NCCL_WRAPPERS_H

#include <nccl.h>
#include <stdint.h>
#include "inflight.h"

#ifdef __cplusplus
extern "C" {
#endif

// Shared globals (defined in libncclplugin.c mostly)
extern int g_my_rank;
extern int g_nranks;
extern ncclComm_t g_comm;
extern volatile int ckpt_pending;

// exposed types for group management in ncclwrappers
typedef struct {
    ncclComm_t comm;
    int64_t seq;
    int64_t target;
} group_entry_t;

group_entry_t* get_or_create_group(ncclComm_t comm);
void increment_seq_and_wait(ncclComm_t comm);
void reset_groups();
void clear_wrapper_pointers();

// Helper functions provided by libncclplugin.c
void add_stream(cudaStream_t stream);
void init_rank_info(ncclComm_t comm);

ncclResult_t ncclCommInitRank(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank);
ncclResult_t ncclCommInitRankConfig(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank, void* config);
ncclResult_t ncclCommDestroy(ncclComm_t comm);
ncclResult_t ncclCommInitAll(ncclComm_t* comm, int nGpus, const int* devList);
ncclResult_t ncclIprobe(int source, int dest, ncclComm_t comm, int* flag, int64_t* pending);

ncclResult_t ncclBroadcast(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclReduce(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount, ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclGather_p2p(const void* sendbuff, void* recvbuff, size_t sendcount, ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclScatter_p2p(const void* sendbuff, void* recvbuff, size_t sendcount, ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclReduceScatter(const void* sendbuff, void* recvbuff, size_t recvcount, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream);

ncclResult_t call_real_ncclSend(const void* sendbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm, cudaStream_t stream);
ncclResult_t call_real_ncclRecv(void* recvbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm, cudaStream_t stream);
ncclResult_t call_real_ncclCommUserRank(ncclComm_t comm, int* rank);
ncclResult_t call_real_ncclCommCount(ncclComm_t comm, int* count);
ncclResult_t call_real_ncclGetUniqueId(ncclUniqueId* out);
const char* call_real_ncclGetErrorString(ncclResult_t result);

#ifdef __cplusplus
}
#endif

#endif
