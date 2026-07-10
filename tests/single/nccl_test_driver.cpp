/**
 * nccl_test.c — NCCL P2P checkpoint/restart test harness.
 *
 * Tests NCCL send/recv under various checkpoint scenarios using a
 * Usage: ./nccl_test <test_id> [msg_count] [size_kb]
 *
 *   Test 1: Basic P2P — complete send/recv, verify data.
 *   Test 2: Post-checkpoint — complete send/recv, sleep for checkpoint.
 *   Test 3: Pre-init checkpoint — sleep before NCCL init, then send/recv.
 *   Test 4: In-transit — send all, sleep for checkpoint, then recv all.
 *           Use msg_count=1 for single message, msg_count>1 for multi.
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

int test_id, num_msgs;
size_t count;
ncclComm_t comm;
cudaStream_t stream;
float *send_buff, *recv_buff;
float *h_send;

/**
 * Initialize NCCL communicator, CUDA stream, device buffers,
 * and populate send data.
 */
void setup(void)
{
    int rank = 0;
    CUDA_CHECK(cudaSetDevice(0));
    NCCL_CHECK(ncclCommInitAll(&comm, 1, &rank));
    CUDA_CHECK(cudaStreamCreate(&stream));

    CUDA_CHECK(cudaMalloc((void **)&send_buff, count * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&recv_buff, count * sizeof(float)));

    h_send = (float *)malloc(count * sizeof(float));
    for (size_t i = 0; i < count; i++)
        h_send[i] = (float)(test_id + i);
    CUDA_CHECK(cudaMemcpy(send_buff, h_send, count * sizeof(float), cudaMemcpyHostToDevice));

    printf("[TEST] Test %d starting. Msg Count: %d, Size: %zu KB\n",
           test_id, num_msgs, (count * sizeof(float)) / 1024);
}

/**
 * Copy recv buffer back to host and compare against sent data.
 * Returns the number of mismatches (0 on success).
 */
int verify(void)
{
    float *h_recv = (float *)malloc(count * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_recv, recv_buff, count * sizeof(float), cudaMemcpyDeviceToHost));

    int errors = 0;
    for (size_t i = 0; i < count; i++) {
        if (h_recv[i] != h_send[i]) {
            if (errors < 5)
                printf("Error at %zu: expected %f, got %f\n", i, h_send[i], h_recv[i]);
            errors++;
        }
    }

    if (errors == 0)
        printf("[TEST] SUCCESS: Data integrity verified.\n");
    else
        printf("[TEST] FAILURE: %d data errors found.\n", errors);

    free(h_recv);
    return errors;
}

/** Free device buffers, host buffer, and destroy NCCL communicator. */
void teardown(void)
{
    CUDA_CHECK(cudaFree(send_buff));
    CUDA_CHECK(cudaFree(recv_buff));
    free(h_send);
    NCCL_CHECK(ncclCommDestroy(comm));
}

/** Issue one grouped send/recv pair to avoid deadlock. */
void send_recv_grouped(void)
{
    NCCL_CHECK(ncclGroupStart());
    NCCL_CHECK(ncclSend(send_buff, count, ncclFloat, 0, comm, stream));
    NCCL_CHECK(ncclRecv(recv_buff, count, ncclFloat, 0, comm, stream));
    NCCL_CHECK(ncclGroupEnd());
}

/** Send/recv all messages and synchronize. */
void send_recv_all(void)
{
    for (int i = 0; i < num_msgs; i++)
        send_recv_grouped();
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

/** Test 1: Baseline — send/recv all messages and verify. */
int basic_p2p(void)
{
    setup();
    send_recv_all();
    printf("[TEST] Test 1: Communication complete.\n");
    int errors = verify();
    teardown();
    return errors;
}

/** Test 2: Complete all comms, then sleep to allow post-completion checkpoint. */
int post_ckpt_p2p(void)
{
    setup();
    send_recv_all();
    printf("[TEST] Test 2: Messaging finished. Sleeping 20s for post-ckpt...\n");
    sleep(20);
    int errors = verify();
    teardown();
    return errors;
}

/** Test 3: Sleep before NCCL init to allow pre-init checkpoint, then run comms. */
int pre_init_ckpt_p2p(void)
{
    printf("[TEST] Test 3: Sleeping 10s before NCCL init...\n");
    sleep(10);
    setup();
    send_recv_all();
    printf("[TEST] Test 3: Communication complete.\n");
    int errors = verify();
    teardown();
    return errors;
}

/**
 * Test 4: Send all messages, sleep to allow in-transit checkpoint,
 * then post matching receives. Use msg_count arg for single vs multi.
 */
int in_transit_p2p(void)
{
    setup();

    printf("[TEST] Test %d: Sending %d messages...\n", test_id, num_msgs);
    for (int i = 0; i < num_msgs; i++)
        NCCL_CHECK(ncclSend(send_buff, count, ncclFloat, 0, comm, stream));

    printf("[TEST] Test %d: Messages SENT. Sleeping 20s for in-transit ckpt...\n", test_id);
    sleep(20);

    printf("[TEST] Test %d: Waking up. Starting receives...\n", test_id);
    for (int i = 0; i < num_msgs; i++)
        NCCL_CHECK(ncclRecv(recv_buff, count, ncclFloat, 0, comm, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));
    printf("[TEST] Test %d: All receives complete.\n", test_id);

    int errors = verify();
    teardown();
    return errors;
}

int main(int argc, char *argv[])
{
    if (argc < 2) {
        printf("Usage: %s <test_id> [msg_count] [size_kb]\n", argv[0]);
        printf("Tests:\n");
        printf("  1: Basic P2P\n");
        printf("  2: Post-completion checkpoint\n");
        printf("  3: Pre-init checkpoint\n");
        printf("  4: In-transit checkpoint\n");
        return 1;
    }

    test_id = atoi(argv[1]);
    num_msgs = (argc >= 3) ? atoi(argv[2]) : 1;
    int size_kb = (argc >= 4) ? atoi(argv[3]) : 4;
    count = (size_kb * 1024) / sizeof(float);

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
