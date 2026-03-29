#include <stdio.h>
#include <nccl.h>
#include "dmtcp.h"
#include "dmtcp_kvdb_c.h"
#include "nccl_wrappers.h"
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <unistd.h>

#define CUDA_CHECK(cmd)                                                        \
    do {                                                                       \
        cudaError_t e = cmd;                                                   \
        if (e != cudaSuccess) {                                                \
            printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__,               \
                   cudaGetErrorString(e));                                     \
            exit(1);                                                           \
        }                                                                      \
    } while (0)


#define MAX_STREAMS 64
#define MESSAGE_COUNT 1024
#define MESSAGE_SIZE 4096

static void *drain_buffer[MESSAGE_COUNT];
static cudaStream_t g_active_streams[MAX_STREAMS];
static int g_num_streams = 0;
uint64_t g_target_seq = 0;
uint64_t g_seq_num = 0;

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

ncclResult_t ncclIprobe(int source, ncclComm_t comm, int* flag, size_t* pending) {
  if (flag == NULL || pending == NULL) return ncclInvalidArgument;

  // format key and query
  char key[64];
  snprintf(key, sizeof(key), "%d|%d", source, g_my_rank);
  
  // Use our wrapper to get the count
  int64_t count = dmtcp_kvdb_get(key);

  if (count > 0) {
      *flag = 1;
      *pending = (size_t)count;
      printf("[NCCL-MANA] Probe: Found %ld messages from %d to %d\n", count, source, g_my_rank);
  } else {
      *flag = 0;
      *pending = 0;
  }

  return ncclSuccess;
}

/*
 * Renaming as this is conflicting with ncclBroadcast defined in the header
 * This function is already implemented in nccl_wrappers.c
 *
 * Takes in the variables needed to perform a ncclSend. 
 * Checks to make sure rank tracking has been initialized. 
 * Then loops through all ranks and sends messages.
 * Skips if the rank to send to matches the current rank.
 */
void ncclBroadcast_(
		const void* sendbuff, 
		size_t count, 
		ncclDataType_t datatype, 
		int peer, 
		ncclComm_t comm, 
		cudaStream_t stream
) {
	if (g_my_rank == -1) init_rank_info(comm);
	for (int i = 0; i < g_nranks; i++) {
		if (i == g_my_rank) continue;
		ncclSend(sendbuff, count, datatype, peer, comm, stream);
	}
}

// Key format: "src|dest"
static void build_msg_key(char *buf, size_t len, int src, int dest) {
    snprintf(buf, len, "%d|%d", src, dest);
}


static void drain_messages() {
    printf("DRAIN: %d ranks found\n", g_nranks);
    for (size_t src = 0; src < g_nranks; src++) {
        if (src == g_my_rank) continue;
        int flag;
        size_t pending;
        ncclIprobe(src, g_comm, &flag, &pending);

        if (!flag) continue;

        for (size_t i = 0; i < pending; i++) {
            void *tmp_buf;
            CUDA_CHECK(cudaSetDevice(g_my_rank));
            CUDA_CHECK(cudaMalloc(&tmp_buf, MESSAGE_SIZE));
            printf("Pending messages found\n");

            printf("[NCCL Drain] Rank %d: draining %zu messages from %zu\n", g_my_rank, pending, src);
            fflush(stdout);

            ncclRecv(tmp_buf, MESSAGE_SIZE / sizeof(float), ncclFloat, src, g_comm, 0);
            printf("[NCCL Drain] Rank %d: recv posted, syncing...\n", g_my_rank);
            fflush(stdout);

            cudaStreamSynchronize(0);
            printf("[NCCL Drain] Rank %d: sync done\n", g_my_rank);
            fflush(stdout);

            // Save to drain buffer for replay after resume if needed
            void *copy = malloc(MESSAGE_SIZE);
            CUDA_CHECK(cudaMemcpy(copy, tmp_buf, MESSAGE_SIZE, cudaMemcpyDeviceToHost));
            drain_buffer[i] = copy;

            CUDA_CHECK(cudaFree(tmp_buf));
        }
    }
}

static void sequence_number_consensus() {
    char my_seq_key[64];
    snprintf(my_seq_key, sizeof(my_seq_key), "rank_%d_seq", g_my_rank);
    // Use SET instead of INCR so we don't accumulate across checkpoints
    dmtcp_kvdb_set(my_seq_key, g_seq_num);

    // Use a DMTCP global barrier to ensure everyone has published their sequence
    // before we try to calculate the maximum.
    dmtcp_global_barrier("nccl-consensus-seq-published");

    // Wait for all ranks to publish their sequence and find the MAX
    uint64_t max_seq = g_seq_num;
    for (int i = 0; i < g_nranks; i++) {
        char key[64];
        snprintf(key, sizeof(key), "rank_%d_seq", i);
        uint64_t rank_seq = 0;
        // Busy wait for other ranks to publish (in a real distributed KVDB)
        while ((rank_seq = dmtcp_kvdb_get(key)) == 0 && i != g_my_rank) {
            usleep(1000);
        }
        if (rank_seq > max_seq) max_seq = rank_seq;
    }

    g_target_seq = max_seq;
    printf("[NCCL Drain] Rank %d consensus target seq: %lu (current: %lu)\n", g_my_rank, g_target_seq, g_seq_num);

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
    printf("[NCCL Drain] Starting drain on rank %d (current seq: %lu)\n", g_my_rank, g_seq_num);
    
    // Set pending flag to capture any late messages if threads are still running
    ckpt_pending = 1;
    // __sync_synchronize();

    // Sync any active streams first
    for (int i = 0; i < g_num_streams; i++) {
        cudaStreamSynchronize(g_active_streams[i]);
    }

    // Drain messages
    drain_messages();

    // Sequence number consensus
    sequence_number_consensus();

    // Wait for application thread to reach target_seq
    while (g_seq_num < g_target_seq) {
        usleep(1000);
    }
    
    while (!all_drained()) {
        // TODO: Backoff?
         usleep(1000);
    }
    
    printf("[NCCL Drain] Drain complete on rank %d at seq %lu\n", g_my_rank, g_seq_num);
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
  g_target_seq = 0;
  g_num_streams = 0;
}

static void
restart()
{
  printf("The application is now restarting from a checkpoint.\n");
  ckpt_pending = 0;
  g_target_seq = 0;
  g_seq_num = 0; // Sequence should probably reset or be restored from ckpt
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

  case DMTCP_EVENT_PRESUSPEND:
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
