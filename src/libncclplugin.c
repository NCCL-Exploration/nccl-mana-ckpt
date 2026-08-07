#include "dmtcp.h"
#include "dmtcp_kvdb_c.h"
#include "inflight.h"
#include "nccl_wrappers.h"
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <nccl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

#define CUDA_CHECK(cmd)                                                                            \
    do {                                                                                           \
        cudaError_t e = cmd;                                                                       \
        if (e != cudaSuccess) {                                                                    \
            printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e));           \
            exit(1);                                                                               \
        }                                                                                          \
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
        if (g_active_streams[i] == stream)
            return;
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

ncclResult_t ncclIprobe(int source, int dest, ncclComm_t comm, int *flag, int64_t *pending) {
    if (flag == NULL || pending == NULL)
        return ncclInvalidArgument;

    char key_send[64], key_recv[64];
    snprintf(key_send, sizeof(key_send), "%d|%d|send", source, dest);
    snprintf(key_recv, sizeof(key_recv), "%d|%d|recv", source, dest);

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
void ncclBroadcast_(const void *sendbuff, size_t count, ncclDataType_t datatype, int peer,
                    ncclComm_t comm, cudaStream_t stream) {
    if (g_my_rank == -1)
        init_rank_info(comm);
    if (g_my_rank != peer)
        return;
    for (int i = 0; i < g_nranks; i++) {
        if (i == g_my_rank)
            continue;
        ncclSend(sendbuff, count, datatype, i, comm, stream);
    }
}

// Key format: "src|dest"
static void build_msg_key(char *buf, size_t size, int src, int dest) {
    snprintf(buf, size, "%d|%d", src, dest);
}

static void drain_messages() {
    printf("DRAIN: %d ranks found\n", g_nranks);
    CUDA_CHECK(cudaSetDevice(g_my_rank));
    for (int peer = 0; peer < g_nranks; peer++) {
        if (peer == g_my_rank)
            continue;

        int flag;
        int64_t message_size;

        // Inbound: did peer send to me? (diff > 0)
        ncclIprobe(peer, g_my_rank, g_comm, &flag, &message_size);
        if (flag && message_size > 0) {
            void *tmp_buf;
            CUDA_CHECK(cudaMalloc(&tmp_buf, message_size));
            printf("[NCCL Drain] Rank %d: draining recv of %ld bytes from %d\n", g_my_rank,
                   message_size, peer);
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
            printf("[NCCL Drain] Rank %d: draining send of %zu bytes to %d\n", g_my_rank, size,
                   peer);
            fflush(stdout);

            call_real_ncclSend(tmp_buf, size, ncclInt8, peer, g_comm, 0);
            cudaStreamSynchronize(0);

            CUDA_CHECK(cudaFree(tmp_buf));
            inflight_mark_send(g_my_rank, peer, size);
        }
    }
}

static void sequence_number_consensus() {
    group_entry_t *g = get_or_create_group(g_comm);
    printf("[rank %d] publishing seq %lu\n", g_my_rank, g->seq);
    char my_seq_key[64];
    snprintf(my_seq_key, sizeof(my_seq_key), "rank_%d_seq", g_my_rank);
    // Publish g->seq + 1 so that we can distinguish '0' (not found) from actual seq=0
    dmtcp_kvdb_set(my_seq_key, g->seq + 1);

    printf("[rank %d] reaching barrier\n", g_my_rank);
    // Use a DMTCP global barrier to ensure everyone has published their sequence
    // before we try to calculate the maximum.
    dmtcp_global_barrier("nccl-consensus-seq-published");
    printf("[rank %d] past barrier\n", g_my_rank);

    // Wait for all ranks to publish their sequence and find the MAX
    uint64_t max_seq = g->seq;
    for (int i = 0; i < g_nranks; i++) {
        uint64_t rank_seq;

        char key[64];
        snprintf(key, sizeof(key), "rank_%d_seq", i);
        uint64_t rank_seq_plus_one = 0;

        if (i == g_my_rank) {
            rank_seq = g->seq;
        }

        else {
            // Wait for the key to be available (value > 0).
            // The barrier should guarantee it's immediately available.
            while ((rank_seq_plus_one = dmtcp_kvdb_get(key)) == 0 && i != g_my_rank) {
                usleep(1000);
            }

            rank_seq = rank_seq_plus_one - 1;
        }

        if (rank_seq > max_seq) {
            max_seq = rank_seq;
        }
    }

    g->target = max_seq;
    printf("[NCCL Drain] Rank %d consensus target seq: %lu (current: %lu)\n", g_my_rank, g->target,
           g->seq);
}

static void flush_pending_to_kvdb() {
    // One-time batch flush of the local pending table to KVDB.
    // Only called at checkpoint time for multi-process consensus.
    for (int peer = 0; peer < g_nranks; peer++) {
        if (peer == g_my_rank)
            continue;

        char key_send[64], key_recv[64];
        snprintf(key_send, sizeof(key_send), "%d|%d|send", g_my_rank, peer);
        snprintf(key_recv, sizeof(key_recv), "%d|%d|recv", peer, g_my_rank);

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
        if (source == g_my_rank)
            continue;
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
    if (!all_drained()) {
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

/*
 * Destroy the live NCCL communicator so the CUDA driver holds no CUDA
 * IPC / P2P handles (or GDR mappings established during connection setup)
 * when cuCheckpointProcessCheckpoint runs. This must run AFTER drain_nccl()
 * has finished draining in-flight messages and reaching sequence consensus —
 * destroying mid-collective would corrupt the drain accounting.
 *
 * ncclCommDestroy() (in nccl_wrappers.c) resolves g_comm to the real,
 * currently-live communicator and clears g_comm to NULL on success, so the
 * checkpoint image is written with g_comm == NULL and reinit_nccl() (called
 * from resume()/restart()) knows a fresh communicator must be created.
 */
static void teardown_nccl_for_checkpoint() {
    if (g_comm == NULL) {
        printf("[NCCL Plugin] teardown: no active comm, nothing to destroy\n");
        return;
    }
    printf("[rank %d] teardown: destroying NCCL comm %p before checkpoint\n", g_my_rank,
           (void *)g_comm);
    fflush(stdout);

    ncclResult_t r = ncclCommDestroy(g_comm);
    if (r != ncclSuccess) {
        fprintf(stderr, "[rank %d] teardown: ncclCommDestroy failed: %s\n", g_my_rank,
                ncclGetErrorString(r));
    }
    // Defensive: make sure g_comm is NULL even if destroy reported an error,
    // so we never try to checkpoint or reuse a half-torn-down handle.
    g_comm = NULL;
}

static void checkpoint() {
    printf("\n*** The plugin is being called before checkpointing. ***\n");
    struct timeval start, end;
    gettimeofday(&start, NULL);

    drain_nccl();
    teardown_nccl_for_checkpoint();

    gettimeofday(&end, NULL);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0;
    printf("[NCCL PLUGIN] Checkpoint overhead: %.6f seconds\n", elapsed);
    fflush(stdout);
}

/*
 * ncclUniqueId is a fixed-size opaque byte blob (128 bytes as of current
 * NCCL versions). dmtcp_kvdb only stores int64_t, so we move the id through
 * the coordinator's KVDB in 8-byte chunks — the same "publish, barrier,
 * read" pattern sequence_number_consensus() already uses for seq numbers.
 */
#define NCCL_ID_NCHUNKS (sizeof(ncclUniqueId) / sizeof(int64_t))

static void publish_nccl_id(const ncclUniqueId *id) {
    const int64_t *chunks = (const int64_t *)id;
    for (size_t i = 0; i < NCCL_ID_NCHUNKS; i++) {
        char key[64];
        snprintf(key, sizeof(key), "nccl-reinit-id-%zu", i);
        dmtcp_kvdb_set(key, chunks[i]);
    }
}

static void fetch_nccl_id(ncclUniqueId *id) {
    int64_t *chunks = (int64_t *)id;
    for (size_t i = 0; i < NCCL_ID_NCHUNKS; i++) {
        char key[64];
        snprintf(key, sizeof(key), "nccl-reinit-id-%zu", i);
        chunks[i] = dmtcp_kvdb_get(key);
    }
}

/*
 * Recreate the NCCL communicator after a checkpoint/restart cycle.
 *
 * g_my_rank / g_nranks are ordinary process globals: DMTCP restores them as
 * part of normal memory, so they still hold their pre-checkpoint values here
 * — they are NOT reset in restart() (see below). g_comm, on the other hand,
 * was explicitly nulled by teardown_nccl_for_checkpoint() before the
 * checkpoint was taken, which is exactly the signal we use to know a fresh
 * communicator is needed.
 *
 * Rank 0 generates a new ncclUniqueId (the old one is meaningless — it was
 * tied to bootstrap state that no longer exists) and publishes it through
 * the KVDB; every rank reads it back after a barrier, then all ranks call
 * ncclCommInitRank together, exactly like the very first init.
 *
 * NOTE (PoC-scope limitation): this assumes one NCCL rank per OS process,
 * i.e. programs using ncclGetUniqueId + ncclCommInitRank, matching both
 * nccl_test.cu and tests/multi/cc_simple.c. It does not handle a single
 * process driving multiple communicators via ncclCommInitAll.
 *
 * NOTE (needs verification on your setup): this must run AFTER the CUDA
 * plugin has finished restoring CUDA driver/context state for this process.
 * Confirm the event ordering between libncclplugin.so and libdmtcp_cuda.so
 * empirically (e.g. add prints on both sides) before relying on this.
 */
static void reinit_nccl() {
    if (g_comm != NULL) {
        printf("[rank %d] reinit_nccl: comm already present, skipping\n", g_my_rank);
        return;
    }

    printf("[rank %d] reinit_nccl: recreating comm (%d ranks)\n", g_my_rank, g_nranks);
    fflush(stdout);

    pre_init_cuda();
    CUDA_CHECK(cudaSetDevice(g_my_rank));

    ncclUniqueId id;
    if (g_my_rank == 0) {
        ncclResult_t idr = ncclGetUniqueId(&id);
        if (idr != ncclSuccess) {
            fprintf(stderr, "[NCCL Plugin] reinit_nccl: ncclGetUniqueId failed: %s\n",
                    ncclGetErrorString(idr));
            exit(1);
        }
        publish_nccl_id(&id);
    }

    // Every rank waits here so rank 0's id is visible before anyone reads it.
    dmtcp_global_barrier("nccl-reinit-id-published");

    if (g_my_rank != 0) {
        fetch_nccl_id(&id);
    }

    ncclComm_t new_comm = NULL;
    ncclResult_t r = ncclCommInitRank(&new_comm, g_nranks, id, g_my_rank);
    if (r != ncclSuccess) {
        fprintf(stderr, "[rank %d] reinit_nccl: ncclCommInitRank failed: %s\n", g_my_rank,
                ncclGetErrorString(r));
        exit(1);
    }

    // init_rank_info() no-ops here since g_my_rank != -1, so set g_comm ourselves.
    g_comm = new_comm;
    printf("[rank %d] reinit_nccl: comm recreated at %p\n", g_my_rank, (void *)g_comm);
    fflush(stdout);
}

static void resume() {
    printf("[rank %d] RESUME called\n", g_my_rank);
    printf("*** The application has now been checkpointed. ***\n");
    ckpt_pending = 0;
    reset_groups();
    inflight_init();
    printf("[rank %d] RESUME complete\n", g_my_rank);
}

static void restart() {
    printf("The application is now restarting from a checkpoint.\n");
    ckpt_pending = 0;
    g_num_streams = 0;
    g_comm = NULL;
    clear_wrapper_pointers();
    reset_groups();
    inflight_init();
    // NOTE: drain_queues are NOT reset — they survive restart for ncclRecv replay
}

static void pre_init_cuda() {
    printf("[NCCL Plugin] pre_init_cuda: getting executable path...\n");
    fflush(stdout);
    const char *exec_path = dmtcp_get_executable_path();
    printf("[NCCL Plugin] pre_init_cuda: exec_path pointer is %p\n", exec_path);
    fflush(stdout);
    if (exec_path != NULL) {
        printf("[NCCL Plugin] pre_init_cuda: exec_path is %s\n", exec_path);
        fflush(stdout);
        const char *basename = strrchr(exec_path, '/');
        basename = basename ? basename + 1 : exec_path;
        if (strcmp(basename, "srun") == 0 || strcmp(basename, "mpirun") == 0 ||
            strcmp(basename, "mpiexec") == 0 || strcmp(basename, "bash") == 0 ||
            strcmp(basename, "sh") == 0 || strcmp(basename, "orted") == 0 ||
            strcmp(basename, "slurmd") == 0 || strcmp(basename, "slurmstepd") == 0 ||
            strcmp(basename, "lower-half") == 0 || strncmp(basename, "mana_", 5) == 0 ||
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
    printf("[NCCL Plugin] pre_init_cuda: cudaGetDeviceCount returned %d, device_count = %d\n",
           (int)err, device_count);
    fflush(stdout);
    if (err == cudaSuccess) {
        int active_dev = (g_my_rank >= 0 && g_my_rank < device_count) ? g_my_rank : 0;
        printf("[NCCL Plugin] Pre-initializing CUDA on control thread for device %d...\n",
               active_dev);
        cudaSetDevice(active_dev);
        cudaFree(0);
        printf("[NCCL Plugin] CUDA pre-initialization complete.\n");
    } else {
        printf("[NCCL Plugin] Failed to get device count during pre-init: %s\n",
               cudaGetErrorString(err));
    }
}

static void example_event_hook(DmtcpEvent_t event, DmtcpEventData_t *data) {
    static int first_time = 1;
    switch (event) {
    case DMTCP_EVENT_INIT:
        printf("NCCL PLUGIN\n");
        break;

    case DMTCP_EVENT_RUNNING_AFTER:
        if (first_time) {
            first_time = 0;
            break;
        }
        printf("[NCCL PLUGIN] Start reinialization\n");
        fflush(stdout);
        reinit_nccl();
        printf("[NCCL PLUGIN] Finished reinitialization\n");
        fflush(stdout);
        dmtcp_global_barrier("");
        break;

    case DMTCP_EVENT_EXIT:
        break;

    case DMTCP_EVENT_PRESUSPEND:
        printf("[NCCL PLUGIN] Presuspend event begin\n");
        fflush(stdout);
        checkpoint();
        printf("[NCCL PLUGIN] Presuspend event finished\n");
        fflush(stdout);
        break;

    case DMTCP_EVENT_RESUME:
        printf("[NCCL PLUGIN] Resume event begin\n");
        fflush(stdout);
        resume();
        printf("[NCCL PLUGIN] Resume event finished\n");
        fflush(stdout);
        break;

    case DMTCP_EVENT_RESTART:
        printf("[NCCL PLUGIN] Restart event begin\n");
        fflush(stdout);
        restart();
        printf("[NCCL PLUGIN] Restart event finished\n");
        fflush(stdout);
        break;

    default:
        break;
    }
}

DmtcpPluginDescriptor_t example_plugin = {
    DMTCP_PLUGIN_API_VERSION, DMTCP_PACKAGE_VERSION, "example",         "DMTCP",
    "dmtcp@ccs.neu.edu",      "Example plugin",      example_event_hook};

DMTCP_DECL_PLUGIN(example_plugin);
