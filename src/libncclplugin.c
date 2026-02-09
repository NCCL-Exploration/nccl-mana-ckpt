#include <stdio.h>
#include <nccl.h>
#include "dmtcp.h"

#define MAX_STREAMS 64

static cudaStream_t g_active_streams[MAX_STREAMS];
static int g_num_streams = 0;

static ncclComm_t g_comm = NULL;
static int g_nranks = 0;
static int g_my_rank = -1;

static ncclResult_t (*real_ncclSend)(const void*, size_t, ncclDataType_t, int, ncclComm_t, cudaStream_t) = NULL;
static ncclResult_t (*real_ncclRecv)(void*, size_t, ncclDataType_t, int, ncclComm_t, cudaStream_t) = NULL;

#define LOAD_SYM(name) if (!real_##name) real_##name = dlsym(RTLD_NEXT, #name)

static void add_stream(cudaStream_t stream) {
    for (int i = 0; i < g_num_streams; i++) {
        if (g_active_streams[i] == stream) return;
    }
    if (g_num_streams < MAX_STREAMS) {
        g_active_streams[g_num_streams++] = stream;
    }
}

static void init_rank_info(ncclComm_t comm) {
    if (g_my_rank == -1) {
        g_comm = comm;
        ncclCommUserRank(comm, &g_my_rank);
        ncclCommCount(comm, &g_nranks);
        printf("[NCCL Drain] Rank %d of %d\n", g_my_rank, g_nranks);
    }
}

static void drain_streams() {
    for (int i = 0; i < g_num_streams; i++) {
        cudaStreamSynchronize(g_active_streams[i]);
    }
}

static int all_drained() {
    for (int source = 0; source < g_nranks; source++) {
        if (source == g_my_rank) continue;
        
        char key[64];
        snprintf(key, sizeof(key), "%d|%d", source, g_my_rank);
        const char* valueStr = dmtcp_kv_get(key);
        
        if (valueStr != NULL && strtoul(valueStr, NULL, 10) > 0) {
            return 0;
        }
    }
    return 1;
}

static void drain_nccl() {
    printf("[NCCL Drain] Starting drain on rank %d\n", g_my_rank);
    
    drain_streams();
    
    while (!all_drained()) {
        // TODO: Handle pending messages
    }
    
    printf("[NCCL Drain] Drain complete on rank %d\n", g_my_rank);
}

ncclResult_t ncclSend(const void* buff, size_t count, ncclDataType_t datatype,
                      int peer, ncclComm_t comm, cudaStream_t stream) {
    LOAD_SYM(ncclSend);
    init_rank_info(comm);
    add_stream(stream);
    
    // TODO: Increment kvdb count for "g_my_rank|peer"
    
    return real_ncclSend(buff, count, datatype, peer, comm, stream);
}

ncclResult_t ncclRecv(void* buff, size_t count, ncclDataType_t datatype,
                      int peer, ncclComm_t comm, cudaStream_t stream) {
    LOAD_SYM(ncclRecv);
    init_rank_info(comm);
    add_stream(stream);
    
    // TODO: Decrement kvdb count for "peer|g_my_rank"
    
    return real_ncclRecv(buff, count, datatype, peer, comm, stream);
}

static void
checkpoint()
{
  printf("\n*** The plugin is being called before checkpointing. ***\n");
  drain_nccl();
}

static void
resume()
{
  printf("*** The application has now been checkpointed. ***\n");
  g_num_streams = 0;
}

static void
restart()
{
  printf("The application is now restarting from a checkpoint.\n");
  g_num_streams = 0;
  g_my_rank = -1;
  g_nranks = 0;
  g_comm = NULL;
}

static void
example_event_hook(DmtcpEvent_t event, DmtcpEventData_t *data)
{
  /* NOTE:  See warning in plugin/README about calls to printf here. */
  switch (event) {
  case DMTCP_EVENT_INIT:
    printf("The plugin containing %s has been initialized.\n", __FILE__);
    break;

  case DMTCP_EVENT_EXIT:
    printf("The plugin is being called before exiting.\n");
    break;

  case DMTCP_EVENT_PRECHECKPOINT:
    checkpoint();
    break;

  case DMTCP_EVENT_RESUME:
    resume();
    break;

  case DMTCP_EVENT_RESTART:
    restart();
    break;

  default:
    break;
  }
}

DmtcpPluginDescriptor_t example_plugin = {
  DMTCP_PLUGIN_API_VERSION,
  DMTCP_PACKAGE_VERSION,
  "example",
  "DMTCP",
  "dmtcp@ccs.neu.edu",
  "Example plugin",
  example_event_hook
};

DMTCP_DECL_PLUGIN(example_plugin);
