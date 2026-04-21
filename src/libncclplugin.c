#include <stdio.h>
#include <nccl.h>
#include "dmtcp.h"
#include "dmtcp_kvdb_c.h"
#include "nccl_wrappers.h"
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <unistd.h>
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


#define MAX_STREAMS 64
#define MESSAGE_SIZE 4096

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

ncclResult_t ncclIprobe(int source, ncclComm_t comm, int* flag, size_t* pending) {
  if (flag == NULL || pending == NULL) return ncclInvalidArgument;

  // Read from local pending table — no coordinator I/O
  if (inflight_has_pending(source, g_my_rank)) {
      *flag = 1;
      *pending = inflight_pending_bytes(source, g_my_rank);
      printf("[NCCL-MANA] Probe: Found message of size %zu from %d to %d\n", *pending, source, g_my_rank);
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
	if (g_my_rank != peer) return;
	for (int i = 0; i < g_nranks; i++) {
		if (i == g_my_rank) continue;
		ncclSend(sendbuff, count, datatype, i, comm, stream);
	}
}

// Key format: "src|dest"
static void build_msg_key(char *buf, int src, int dest) {
    sprintf(buf, "%d|%d", src, dest);
}


static void drain_messages() {
    printf("DRAIN: %d ranks found\n", g_nranks);
    for (int src = 0; src < g_nranks; src++) {
        if (src == g_my_rank) continue;

        // Check local pending table — no KVDB reads
        if (!inflight_has_pending(src, g_my_rank)) continue;

        size_t message_size = inflight_pending_bytes(src, g_my_rank);
        printf("[NCCL Drain] Rank %d: draining message of size %zu from %d\n",
               g_my_rank, message_size, src);
        fflush(stdout);

        void *tmp_buf;
        CUDA_CHECK(cudaSetDevice(g_my_rank));
        CUDA_CHECK(cudaMalloc(&tmp_buf, message_size));

        ncclRecv(tmp_buf, message_size / sizeof(float), ncclFloat, src, g_comm, 0);
        printf("[NCCL Drain] Rank %d: recv posted, syncing...\n", g_my_rank);
        fflush(stdout);

        cudaStreamSynchronize(0);
        printf("[NCCL Drain] Rank %d: sync done\n", g_my_rank);
        fflush(stdout);

        // Copy to host and push into drain queue for replay on resume
        void *host_copy = malloc(message_size);
        CUDA_CHECK(cudaMemcpy(host_copy, tmp_buf, message_size, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(tmp_buf));

        drain_queue_push(src, host_copy, message_size);

        // Mark as received locally
        inflight_mark_recv(src, g_my_rank);
    }
}

static void sequence_number_consensus() {
	group_entry_t *g = get_or_create_group(g_comm);
	
    char my_seq_key[64];
    snprintf(my_seq_key, sizeof(my_seq_key), "rank_%d_seq", g_my_rank);
    // Use SET instead of INCR so we don't accumulate across checkpoints
    dmtcp_kvdb_set(my_seq_key, g->seq);

    // Use a DMTCP global barrier to ensure everyone has published their sequence
    // before we try to calculate the maximum.
    dmtcp_global_barrier("nccl-consensus-seq-published");

    // Wait for all ranks to publish their sequence and find the MAX
    uint64_t max_seq = g->seq;
    for (int i = 0; i < g_nranks; i++) {
        char key[64];
        snprintf(key, sizeof(key), "rank_%d_seq", i);
        uint64_t rank_seq = 0;
        // Busy wait for other ranks to publish (in a real distributed KVDB)
		// TODO: We can remove this right? because the barrier already guarantees all ranks are synchronized
        while ((rank_seq = dmtcp_kvdb_get(key)) == 0 && i != g_my_rank) {
            usleep(1000);
        }
        if (rank_seq > max_seq) max_seq = rank_seq;
    }

    g->target = max_seq;
    printf("[NCCL Drain] Rank %d consensus target seq: %lu (current: %lu)\n", g_my_rank, g->target, g->seq);

}

static void flush_pending_to_kvdb() {
    // One-time batch flush of the local pending table to KVDB.
    // Only called at checkpoint time for multi-process consensus.
    for (int src = 0; src < g_nranks; src++) {
        for (int dst = 0; dst < g_nranks; dst++) {
            if (src == dst) continue;
            char key[64];
            build_msg_key(key, src, dst);
            size_t bytes = inflight_pending_bytes(src, dst);
            dmtcp_kvdb_set(key, (int64_t)bytes);
        }
    }
}

static int all_drained() {
    // Check local pending table — no KVDB reads
    for (int source = 0; source < g_nranks; source++) {
        if (source == g_my_rank) continue;
        if (inflight_has_pending(source, g_my_rank)) {
            return 0;
        }
    }
    return 1;
}

static void drain_nccl() {
    group_entry_t *g = get_or_create_group(g_comm);
    printf("[NCCL Drain] Starting drain on rank %d (current seq: %lu)\n", g_my_rank, g->seq);
    
    // Set pending flag to capture any late messages if threads are still running
    ckpt_pending = 1;

    // Sync any active streams first
    for (int i = 0; i < g_num_streams; i++) {
        cudaStreamSynchronize(g_active_streams[i]);
    }

    // Check how many DMTCP peers are connected.
    int numPeers = 0, isRunning = 0;
    dmtcp_get_coordinator_status(&numPeers, &isRunning);
    printf("[NCCL Drain] DMTCP peers: %d, NCCL ranks: %d\n", numPeers, g_nranks);

    if (numPeers > 1) {
        // Flush the local pending table to KVDB (one batch of writes)
        flush_pending_to_kvdb();

        // Multi-process: full drain + consensus
        drain_messages();
        sequence_number_consensus();

        // Wait for application thread to reach target_seq
        while (g->seq < g->target) {
            usleep(1000);
        }

        while (!all_drained()) {
            usleep(1000);
        }
    } else {
        printf("[NCCL Drain] Single process — skipping multi-process consensus\n");
    }
    
    printf("[NCCL Drain] Drain complete on rank %d at seq %lu\n", g_my_rank, g->seq);
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
  reset_groups();
  inflight_init();
  // NOTE: drain_queues are NOT reset — they hold messages to replay via ncclRecv
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
  reset_groups();
  inflight_init();
  // NOTE: drain_queues are NOT reset — they survive restart for ncclRecv replay
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
