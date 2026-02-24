#ifndef NCCL_WRAPPERS_H
#define NCCL_WRAPPERS_H

#include <nccl.h>

#ifdef __cplusplus
extern "C" {
#endif

// Shared globals (defined in libncclplugin.c mostly)
extern int g_my_rank;
extern int g_nranks;
extern ncclComm_t g_comm;
extern volatile int ckpt_pending;

// Helper functions provided by libncclplugin.c
void add_stream(cudaStream_t stream);
void init_rank_info(ncclComm_t comm);

ncclResult_t ncclCommInitRank(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank);
ncclResult_t ncclCommInitAll(ncclComm_t* comm, int nGpus, const int* devList);
ncclResult_t ncclIprobe(int source, ncclComm_t comm, int* flag, size_t* count);

#ifdef __cplusplus
}
#endif

#endif
