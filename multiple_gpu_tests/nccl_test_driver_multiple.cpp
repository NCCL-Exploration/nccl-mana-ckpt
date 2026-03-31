/**
 * nccl_test_driver_multiple.cpp — Multi-GPU NCCL P2P checkpoint/restart test harness.
 *
 * Tests NCCL send/recv across multiple GPUs within a single process.
 * Each GPU gets its own communicator, stream, and device/host buffers,
 * grouped in a GpuCtx struct.
 *
 * Usage: ./nccl_test_driver_multiple <test_id> [msg_count] [size_kb] [num_gpus]
 *
 *   Test 1: Basic P2P — ring send/recv across all GPUs, verify data.
 *   Test 2: Post-completion checkpoint — send/recv, sleep for ckpt.
 *   Test 3: Pre-init checkpoint — sleep before NCCL init, then send/recv.
 *   Test 4: In-transit — send all, sleep for ckpt, then recv all.
 */

#include <stdio.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <stdlib.h>
#include <string.h>

#define CUDA_CHECK(cmd) do { \
    cudaError_t e = cmd; \
    if (e != cudaSuccess) { \
        printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(1); \
    } \
} while(0)

#define NCCL_CHECK(cmd) do { \
    ncclResult_t r = cmd; \
    if (r != ncclSuccess) { \
        printf("NCCL error %s:%d: %s\n", __FILE__, __LINE__, ncclGetErrorString(r)); \
        exit(1); \
    } \
} while(0)

/* ── Per-GPU state ────────────────────────────────────────────────── */
struct GpuCtx {
    ncclComm_t   comm;
    cudaStream_t stream;
    float       *send_buf;    /* device */
    float       *recv_buf;    /* device */
    float       *h_send;      /* host mirror of send_buf */
};

/* ── Globals ──────────────────────────────────────────────────────── */
static int       test_id;
static int       num_msgs;
static size_t    count;       /* number of floats per message */
static int       num_gpus;
static GpuCtx   *gpu;        /* array[num_gpus] */

/* ── Setup / Teardown ─────────────────────────────────────────────── */

/**
 * Initialize N communicators with ncclCommInitAll, then create per-GPU
 * streams, device buffers, and host send mirrors.
 */
void setup(void)
{
    /* --- communicators ------------------------------------------------ */
    ncclComm_t *comms = new ncclComm_t[num_gpus];

    /* Build device list {0, 1, …, N-1} */
    int *dev_list = new int[num_gpus];
    for (int i = 0; i < num_gpus; i++)
        dev_list[i] = i;

    NCCL_CHECK(ncclCommInitAll(comms, num_gpus, dev_list));
    printf("[SETUP] ncclCommInitAll completed for %d GPUs\n", num_gpus);

    /* --- per-GPU resources ------------------------------------------- */
    gpu = new GpuCtx[num_gpus];

    for (int i = 0; i < num_gpus; i++) {
        gpu[i].comm = comms[i];

        CUDA_CHECK(cudaSetDevice(i));

        CUDA_CHECK(cudaStreamCreate(&gpu[i].stream));

        CUDA_CHECK(cudaMalloc(&gpu[i].send_buf, count * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&gpu[i].recv_buf, count * sizeof(float)));

        gpu[i].h_send = (float *)malloc(count * sizeof(float));
        if (!gpu[i].h_send) {
            fprintf(stderr, "malloc failed for gpu[%d].h_send\n", i);
            exit(1);
        }

        /* Fill send buffer with a per-GPU pattern so we can verify later */
        for (size_t j = 0; j < count; j++)
            gpu[i].h_send[j] = (float)(i * 1000 + j);

        CUDA_CHECK(cudaMemcpy(gpu[i].send_buf, gpu[i].h_send,
                              count * sizeof(float), cudaMemcpyHostToDevice));
    }

    delete[] comms;
    delete[] dev_list;

    printf("[SETUP] Test %d starting. GPUs: %d, Msgs: %d, Size: %zu KB each\n",
           test_id, num_gpus, num_msgs, (count * sizeof(float)) / 1024);
}

/**
 * Verify that every GPU received the data its *sender* produced.
 * Ring pattern: gpu[i] sends to gpu[(i+1) % N],
 *   so gpu[i] should have received from gpu[(i - 1 + N) % N].
 * Returns total number of mismatches across all GPUs.
 */
int verify(void)
{
    int total_errors = 0;

    for (int i = 0; i < num_gpus; i++) {
        CUDA_CHECK(cudaSetDevice(i));

        float *h_recv = (float *)malloc(count * sizeof(float));
        CUDA_CHECK(cudaMemcpy(h_recv, gpu[i].recv_buf,
                              count * sizeof(float), cudaMemcpyDeviceToHost));

        int sender = (i - 1 + num_gpus) % num_gpus;
        int errors = 0;
        for (size_t j = 0; j < count; j++) {
            float expected = (float)(sender * 1000 + j);
            if (h_recv[j] != expected) {
                if (errors < 5)
                    printf("[GPU %d] Error at %zu: expected %f (from GPU %d), got %f\n",
                           i, j, expected, sender, h_recv[j]);
                errors++;
            }
        }

        if (errors == 0)
            printf("[GPU %d] SUCCESS: Data integrity verified.\n", i);
        else
            printf("[GPU %d] FAILURE: %d data errors found.\n", i, errors);

        total_errors += errors;
        free(h_recv);
    }

    return total_errors;
}

/** Free all per-GPU resources and destroy communicators. */
void teardown(void)
{
    for (int i = 0; i < num_gpus; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaFree(gpu[i].send_buf));
        CUDA_CHECK(cudaFree(gpu[i].recv_buf));
        CUDA_CHECK(cudaStreamDestroy(gpu[i].stream));
        free(gpu[i].h_send);
        NCCL_CHECK(ncclCommDestroy(gpu[i].comm));
    }
    delete[] gpu;
    gpu = nullptr;
}

/* ── Communication helpers ────────────────────────────────────────── */

/**
 * Issue one grouped ring send/recv across all GPUs.
 * gpu[i] sends to gpu[(i+1) % N] and receives from gpu[(i-1+N) % N].
 */
void send_recv_ring(void)
{
    NCCL_CHECK(ncclGroupStart());
    for (int i = 0; i < num_gpus; i++) {
        int dst = (i + 1) % num_gpus;
        int src = (i - 1 + num_gpus) % num_gpus;
        NCCL_CHECK(ncclSend(gpu[i].send_buf, count, ncclFloat, dst, gpu[i].comm, gpu[i].stream));
        NCCL_CHECK(ncclRecv(gpu[i].recv_buf, count, ncclFloat, src, gpu[i].comm, gpu[i].stream));
    }
    NCCL_CHECK(ncclGroupEnd());
}

/** Synchronize all GPU streams. */
void sync_all(void)
{
    for (int i = 0; i < num_gpus; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaStreamSynchronize(gpu[i].stream));
    }
}

/** Send/recv all messages (ring) and synchronize. */
void send_recv_all(void)
{
    for (int m = 0; m < num_msgs; m++)
        send_recv_ring();
    sync_all();
}

/* ── Test Cases ───────────────────────────────────────────────────── */

/** Test 1: Baseline — ring send/recv all messages and verify. */
int basic_p2p(void)
{
    setup();
    send_recv_all();
    printf("[TEST 1] Ring communication complete.\n");
    int errors = verify();
    teardown();
    return errors;
}

/** Test 2: Complete all comms, then sleep to allow post-completion checkpoint. */
int post_ckpt_p2p(void)
{
    setup();
    send_recv_all();
    printf("[TEST 2] Messaging finished. Sleeping 20s for post-ckpt...\n");
    sleep(20);
    int errors = verify();
    teardown();
    return errors;
}

/** Test 3: Sleep before NCCL init to allow pre-init checkpoint, then run comms. */
int pre_init_ckpt_p2p(void)
{
    printf("[TEST 3] Sleeping 10s before NCCL init...\n");
    sleep(10);
    setup();
    send_recv_all();
    printf("[TEST 3] Ring communication complete.\n");
    int errors = verify();
    teardown();
    return errors;
}

/**
 * Test 4: Send all messages on the ring, sleep for in-transit checkpoint,
 * then post matching receives.
 */
int in_transit_p2p(void)
{
    setup();

    printf("[TEST 4] Sending %d ring messages...\n", num_msgs);
    for (int m = 0; m < num_msgs; m++) {
        NCCL_CHECK(ncclGroupStart());
        for (int i = 0; i < num_gpus; i++) {
            int dst = (i + 1) % num_gpus;
            NCCL_CHECK(ncclSend(gpu[i].send_buf, count, ncclFloat, dst,
                                gpu[i].comm, gpu[i].stream));
        }
        NCCL_CHECK(ncclGroupEnd());
    }

    printf("[TEST 4] Messages SENT. Sleeping 20s for in-transit ckpt...\n");
    sleep(20);

    printf("[TEST 4] Waking up. Posting receives...\n");
    for (int m = 0; m < num_msgs; m++) {
        NCCL_CHECK(ncclGroupStart());
        for (int i = 0; i < num_gpus; i++) {
            int src = (i - 1 + num_gpus) % num_gpus;
            NCCL_CHECK(ncclRecv(gpu[i].recv_buf, count, ncclFloat, src,
                                gpu[i].comm, gpu[i].stream));
        }
        NCCL_CHECK(ncclGroupEnd());
    }

    sync_all();
    printf("[TEST 4] All receives complete.\n");

    int errors = verify();
    teardown();
    return errors;
}

/* ── Main ─────────────────────────────────────────────────────────── */

int main(int argc, char *argv[])
{
    if (argc < 2) {
        printf("Usage: %s <test_id> [msg_count] [size_kb] [num_gpus]\n", argv[0]);
        printf("Tests:\n");
        printf("  1: Basic P2P (ring)\n");
        printf("  2: Post-completion checkpoint\n");
        printf("  3: Pre-init checkpoint\n");
        printf("  4: In-transit checkpoint\n");
        return 1;
    }

    test_id  = atoi(argv[1]);
    num_msgs = (argc >= 3) ? atoi(argv[2]) : 1;
    int size_kb = (argc >= 4) ? atoi(argv[3]) : 4;
    count = (size_kb * 1024) / sizeof(float);

    /* Detect available GPUs or use user-supplied count */
    int available_gpus = 0;
    CUDA_CHECK(cudaGetDeviceCount(&available_gpus));
    num_gpus = (argc >= 5) ? atoi(argv[4]) : available_gpus;

    if (num_gpus < 2) {
        fprintf(stderr, "Error: need at least 2 GPUs, found %d\n", num_gpus);
        return 1;
    }
    if (num_gpus > available_gpus) {
        fprintf(stderr, "Error: requested %d GPUs but only %d available\n",
                num_gpus, available_gpus);
        return 1;
    }

    printf("[MAIN] Using %d GPUs (of %d available)\n", num_gpus, available_gpus);

    switch (test_id) {
        case 1: return basic_p2p();
        case 2: return post_ckpt_p2p();
        case 3: return pre_init_ckpt_p2p();
        case 4: return in_transit_p2p();
        default:
            fprintf(stderr, "Unknown test_id: %d\n", test_id);
            return 1;
    }
}