#include <stdio.h>
#include <nccl.h>
#include "dmtcp.h"
#include "dmtcp_kvdb_c.h"
#include "nccl_wrappers.h"
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <unistd.h>

#define MAX_STREAMS 64

static cudaStream_t g_active_streams[MAX_STREAMS];
static int g_num_streams = 0;

ncclComm_t g_comm = NULL;
int g_nranks = 0;
int g_my_rank = -1;
volatile int ckpt_pending = 0;

void add_stream(cudaStream_t stream) {
    for (int i = 0; i < g_num_streams; i++) {
        if (g_active_streams[i] == stream) return;
    }
    if (g_num_streams < MAX_STREAMS) {
        g_active_streams[g_num_streams++] = stream;
    }
}

void init_rank_info(ncclComm_t comm) {
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

// Key format: "src|dest"
static void build_msg_key(char *buf, size_t len, int src, int dest) {
    snprintf(buf, len, "%d|%d", src, dest);
}

static int all_drained() {
    for (int source = 0; source < g_nranks; source++) {
        if (source == g_my_rank) continue;
        
        char key[64];
        build_msg_key(key, sizeof(key), source, g_my_rank);
        
        int64_t val = dmtcp_kvdb_get(key);
        if (val > 0) {
            return 0;
        }
    }
    return 1;
}

static void drain_nccl() {
    printf("[NCCL Drain] Starting drain on rank %d\n", g_my_rank);
    
    // Set pending flag to capture any late messages if threads are still running
    ckpt_pending = 1;
    __sync_synchronize();

    drain_streams();
    
    while (!all_drained()) {
        // TODO: Backoff?
         usleep(1000);
    }
    
    printf("[NCCL Drain] Drain complete on rank %d\n", g_my_rank);
}

// ncclSend and ncclRecv are now in nccl_wrappers.c

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
  ckpt_pending = 0;
  g_num_streams = 0;
}

static void
restart()
{
  printf("The application is now restarting from a checkpoint.\n");
  ckpt_pending = 0;
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
