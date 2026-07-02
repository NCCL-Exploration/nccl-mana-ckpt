#include <stdio.h>
#include <nccl.h>
#include <cuda_runtime.h>
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

static void pre_init_cuda();

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

ncclResult_t ncclIprobe(int source, int dest, ncclComm_t comm, int* flag, int64_t* pending) {
  if (flag == NULL || pending == NULL) return ncclInvalidArgument;

  char key_send[64], key_recv[64];
  sprintf(key_send, "%d|%d|send", source, dest);
  sprintf(key_recv, "%d|%d|recv", source, dest);
  
  int64_t send_bytes = dmtcp_kvdb_get(key_send);
  int64_t recv_bytes = dmtcp_kvdb_get(key_recv);
  
  int64_t diff = send_bytes - recv_bytes;

  if (diff != 0) {
      *flag = 1;
      *pending = diff;
      printf("[NCCL-MANA] Probe: Found diff of %ld between %d and %d\n", diff, source, dest);
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
    CUDA_CHECK(cudaSetDevice(g_my_rank));
    for (int peer = 0; peer < g_nranks; peer++) {
        if (peer == g_my_rank) continue;

        int flag;
        int64_t message_size;

        // Inbound: did peer send to me? (diff > 0)
        ncclIprobe(peer, g_my_rank, g_comm, &flag, &message_size);
        if (flag && message_size > 0) {
            void *tmp_buf;
            CUDA_CHECK(cudaMalloc(&tmp_buf, message_size));
            printf("[NCCL Drain] Rank %d: draining recv of %ld bytes from %d\n",
                   g_my_rank, message_size, peer);
            fflush(stdout);

            call_real_ncclRecv(tmp_buf, message_size, ncclInt8, peer, g_comm, 0);
            cudaStreamSynchronize(0);

            void *drain_buffer = malloc(message_size);
            CUDA_CHECK(cudaMemcpy(drain_buffer, tmp_buf, message_size, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaFree(tmp_buf));

            drain_queue_push(peer, drain_buffer, message_size);
            inflight_mark_recv(peer, g_my_rank, message_size);
        }

        // Outbound: do I owe peer a send? (diff < 0)
        ncclIprobe(g_my_rank, peer, g_comm, &flag, &message_size);
        if (flag && message_size < 0) {
            size_t size = (size_t)(-message_size);
            void *tmp_buf;
            CUDA_CHECK(cudaMalloc(&tmp_buf, size));
            CUDA_CHECK(cudaMemset(tmp_buf, 0, size));
            printf("[NCCL Drain] Rank %d: draining send of %zu bytes to %d\n",
                   g_my_rank, size, peer);
            fflush(stdout);

            call_real_ncclSend(tmp_buf, size, ncclInt8, peer, g_comm, 0);
            cudaStreamSynchronize(0);

            CUDA_CHECK(cudaFree(tmp_buf));
            inflight_mark_send(g_my_rank, peer, size);
        }
    }
}

static void sequence_number_consensus() {
    group_entry_t *g =  get_or_create_group(g_comm);
    printf("[rank %d] publishing seq %lu\n", g_my_rank, g->seq);
    char my_seq_key[64];
    snprintf(my_seq_key, sizeof(my_seq_key), "rank_%d_seq", g_my_rank);
    // Use SET instead of INCR so we don't accumulate across checkpoints
    dmtcp_kvdb_set(my_seq_key, g->seq);
    printf("[rank %d] reaching barrier\n", g_my_rank);
    // Use a DMTCP global barrier to ensure everyone has published their sequence
    // before we try to calculate the maximum.
    dmtcp_global_barrier("nccl-consensus-seq-published");
    printf("[rank %d] past barrier\n", g_my_rank);
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
            printf("[rank %d] waiting\n", g_my_rank);
            fflush(stdout);
        }
        if (rank_seq > max_seq) max_seq = rank_seq;
    }

    g->target = max_seq;
    printf("[NCCL Drain] Rank %d consensus target seq: %lu (current: %lu)\n", g_my_rank, g->target, g->seq);

}

static void flush_pending_to_kvdb() {
    // One-time batch flush of the local pending table to KVDB.
    // Only called at checkpoint time for multi-process consensus.
    for (int peer = 0; peer < g_nranks; peer++) {
        if (peer == g_my_rank) continue;
        
        char key_send[64], key_recv[64];
        sprintf(key_send, "%d|%d|send", g_my_rank, peer);
        sprintf(key_recv, "%d|%d|recv", peer, g_my_rank);
        
        // We track positive for sends (what we sent to peer)
        int64_t sent = inflight_pending_bytes(g_my_rank, peer);
        dmtcp_kvdb_set(key_send, sent);
        
        // We track negative for receives (what we received from peer)
        int64_t recvd = -inflight_pending_bytes(peer, g_my_rank);
        dmtcp_kvdb_set(key_recv, recvd);
    }
}

static int all_drained() {
    // Check local pending table — no KVDB reads
    for (int source = 0; source < g_nranks; source++) {
        if (source == g_my_rank) continue;
        int flag;
        int64_t pending;
        ncclIprobe(source, g_my_rank, g_comm, &flag, &pending);
        if (flag) {
            return 0;
        }
    }
    return 1;
}

static void drain_nccl() {
    if (g_my_rank == -1) {
        printf("[NCCL Drain] NCCL not yet initialized — skipping drain\n");
        return;
    }
    pre_init_cuda();
    group_entry_t *g = get_or_create_group(g_comm);
    printf("[NCCL Drain] Starting drain on rank %d (current seq: %lu)\n", g_my_rank, g->seq);
    
    // Set pending flag to capture any late messages if threads are still running
    ckpt_pending = 1;
    // __sync_synchronize();
    printf("[rank %d] syncing %d streams\n", g_my_rank, g_num_streams);
	    // Sync any active streams first
    if(!all_drained()){	
	    for (int i = 0; i < g_num_streams; i++) {
        	printf("[rank %d] syncing stream %d\n", g_my_rank, i);
        	cudaStreamSynchronize(g_active_streams[i]);
    	}
    }
    printf("[rank %d] streams synced\n", g_my_rank);
    
    // Check how many DMTCP peers are connected.
    int numPeers = 0, isRunning = 0;
    dmtcp_get_coordinator_status(&numPeers, &isRunning);
    printf("[NCCL Drain] DMTCP peers: %d, NCCL ranks: %d\n", numPeers, g_nranks);
    
    if (numPeers > 1) {
        // Flush the local pending table to KVDB (one batch of writes)
        flush_pending_to_kvdb();

        // Wait for all ranks to flush before reading via ncclIprobe
        dmtcp_global_barrier("nccl-consensus-pending-flushed");

        // Multi-process: full drain + consensus
        drain_messages();
        sequence_number_consensus();

        // Wait for application thread to reach target_seq
        while (g->seq < g->target) {
            usleep(1000);
            printf("[Rank %d] Loop 1 seq: %d target: %d\n", g_my_rank, g->seq, g->target);
            fflush(stdout);
        }

        while (!all_drained()) {
            usleep(1000);
            printf("[Rank %d] Loop 2 seq: %d target: %d\n", g_my_rank, g->seq, g->target);
            fflush(stdout);
        }

        printf("[Rank %d] draindraindrain complete\n", g_my_rank);
        fflush(stdout);

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
  printf("[rank %d] RESUME called\n", g_my_rank);
  printf("*** The application has now been checkpointed. ***\n");
  ckpt_pending = 0;
  reset_groups();
  inflight_init();
  printf("[rank %d] RESUME complete\n", g_my_rank);
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

static void pre_init_cuda() {
  printf("[NCCL Plugin] pre_init_cuda: getting executable path...\n");
  fflush(stdout);
  const char* exec_path = dmtcp_get_executable_path();
  printf("[NCCL Plugin] pre_init_cuda: exec_path pointer is %p\n", exec_path);
  fflush(stdout);
  if (exec_path != NULL) {
      printf("[NCCL Plugin] pre_init_cuda: exec_path is %s\n", exec_path);
      fflush(stdout);
      const char* basename = strrchr(exec_path, '/');
      basename = basename ? basename + 1 : exec_path;
      if (strcmp(basename, "srun") == 0 ||
          strcmp(basename, "mpirun") == 0 ||
          strcmp(basename, "mpiexec") == 0 ||
          strcmp(basename, "bash") == 0 ||
          strcmp(basename, "sh") == 0 ||
          strcmp(basename, "orted") == 0 ||
          strcmp(basename, "slurmd") == 0 ||
          strcmp(basename, "slurmstepd") == 0 ||
          strcmp(basename, "lower-half") == 0 ||
          strncmp(basename, "mana_", 5) == 0 ||
          strncmp(basename, "dmtcp_", 6) == 0) {
          printf("[NCCL Plugin] pre_init_cuda: skipping CUDA init for %s\n", basename);
          fflush(stdout);
          return;
      }
  }


  printf("[NCCL Plugin] pre_init_cuda: about to call cudaGetDeviceCount\n");
  fflush(stdout);
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  printf("[NCCL Plugin] pre_init_cuda: cudaGetDeviceCount returned %d, device_count = %d\n", (int)err, device_count);
  fflush(stdout);
  if (err == cudaSuccess) {

      printf("[NCCL Plugin] Pre-initializing CUDA on control thread for %d devices...\n", device_count);
      for (int i = 0; i < device_count; i++) {
          cudaSetDevice(i);
          cudaFree(0);
      }
      printf("[NCCL Plugin] CUDA pre-initialization complete.\n");
  } else {
      printf("[NCCL Plugin] Failed to get device count during pre-init: %s\n", cudaGetErrorString(err));
  }
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
