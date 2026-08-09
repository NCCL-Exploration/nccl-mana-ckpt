#include "nccl_wrappers.h"
#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <unistd.h>

// gcc -o test_nccl_wrappers tests/seq_algorithm_test.c src/nccl_wrappers.c tests/seq_stubs.c -I
// include -lpthread
//./test_nccl_wrappers
extern volatile int ckpt_pending;

#define TEST_COMM ((ncclComm_t)0x1)
#define TEST_COMM_2 ((ncclComm_t)0x2)

extern ncclComm_t fake_last_allreduce_comm;
extern ncclComm_t fake_last_destroyed_comm;
extern int fake_finalize_resolve_count;
extern ncclResult_t fake_init_result;
extern ncclResult_t fake_split_result;
extern int fake_last_init_nranks;
extern int fake_last_init_rank;
extern ncclUniqueId fake_last_init_id;
extern ncclConfig_t *fake_last_init_config;
extern ncclComm_t fake_last_split_parent;
extern int fake_last_split_color;
extern int fake_last_split_key;
extern ncclConfig_t *fake_last_split_config;
extern ncclComm_t fake_last_finalized_comm;
extern ncclComm_t fake_last_aborted_comm;
extern int fake_use_alternate_comm;
extern int fake_split_queries_ready;

void get_or_create_group_test(ncclComm_t comm) {
    reset_groups();
    group_entry_t *g = get_or_create_group(comm);
    assert(g->seq == 0);
    assert(g->target == -1);
    assert(get_or_create_group(comm) == g);
    printf("get_or_create_group: PASS\n");
}

void reset_groups_test(ncclComm_t comm) {
    reset_groups();
    group_entry_t *g = get_or_create_group(comm);
    g->seq = 10;
    g->target = 5;
    reset_groups();
    g = get_or_create_group(comm);
    assert(g->seq == 0);
    assert(g->target == -1);
    printf("reset_groups: PASS\n");
}

void increment_seq_and_wait_nockpt_test(ncclComm_t comm) {
    reset_groups();
    ckpt_pending = 0;
    increment_seq_and_wait(comm);
    increment_seq_and_wait(comm);
    assert(get_or_create_group(comm)->seq == 2);
    printf("increment_seq_and_wait (no checkpoint): PASS\n");
}

static void *test_dmtcp_thread(void *arg) {
    (void)arg;
    usleep(100 * 1000);
    ckpt_pending = 0;
    printf("DMTCP checkpoint complete, ckpt_pending set to 0\n");
    return NULL;
}

void increment_seq_and_wait_ckpt_test(ncclComm_t comm) {
    reset_groups();
    ckpt_pending = 1;
    group_entry_t *g = get_or_create_group(comm);
    g->seq = 3;
    g->target = 3;
    pthread_t tid;
    pthread_create(&tid, NULL, test_dmtcp_thread, NULL);
    increment_seq_and_wait(comm); /* dmtcp goes and checkpoints */
    pthread_join(tid, NULL);
    assert(g->seq == 4);
    printf("increment_seq_and_wait (blocks at target): PASS\n");
}

void multi_group_test(ncclComm_t comm1, ncclComm_t comm2) {
    reset_groups();
    ckpt_pending = 1;

    group_entry_t *g1 = get_or_create_group(comm1);
    group_entry_t *g2 = get_or_create_group(comm2);

    /* group 1 is at target, group 2 is behind */
    g1->seq = 5;
    g1->target = 5;
    g2->seq = 3;
    g2->target = 5;

    /* group 2 is behind so increment should not block */
    increment_seq_and_wait(comm2);
    assert(g2->seq == 4);

    /* group 1 should be unaffected */
    assert(g1->seq == 5);

    /* now group 2 is still behind, increment again */
    increment_seq_and_wait(comm2);
    assert(g2->seq == 5);

    /* now both groups are at target, group 1 should block */
    pthread_t tid;
    pthread_create(&tid, NULL, test_dmtcp_thread, NULL);
    increment_seq_and_wait(comm1);
    pthread_join(tid, NULL);

    assert(g1->seq == 6);
    assert(g2->seq == 5); /* group 2 unaffected by group 1 unblocking */

    printf("two independent groups: PASS\n");
}

void configured_init_test(void) {
    reset_groups();
    ncclComm_t comm = NULL;
    ncclUniqueId id = {{42}};
    ncclConfig_t config = {0};

    assert(ncclCommInitRankConfig(&comm, 4, id, 2, &config) == ncclSuccess);
    group_entry_t *g = find_group(comm);
    assert(g != NULL);
    assert(g->rank == 2);
    assert(g->nranks == 4);
    assert(g->seq == 0);
    assert(fake_last_init_nranks == 4);
    assert(fake_last_init_rank == 2);
    assert(fake_last_init_id.internal[0] == 42);
    assert(fake_last_init_config == &config);
    printf("configured communicator registration: PASS\n");
}

void split_lifecycle_test(void) {
    reset_groups();
    ncclComm_t parent = NULL;
    ncclComm_t child = NULL;
    ncclUniqueId id = {{0}};
    ncclConfig_t config = {0};

    assert(ncclCommInitRankConfig(&parent, 4, id, 2, &config) == ncclSuccess);
    assert(ncclCommSplit(parent, 1, 7, &child, &config) == ncclSuccess);
    assert(fake_last_split_parent == parent);
    assert(fake_last_split_color == 1);
    assert(fake_last_split_key == 7);
    assert(fake_last_split_config == &config);
    assert(find_group(parent) != NULL);
    assert(find_group(child) != NULL);

    increment_seq_and_wait(child);
    assert(find_group(parent)->seq == 0);
    assert(find_group(child)->seq == 1);

    assert(ncclCommFinalize(child) == ncclSuccess);
    assert(fake_last_finalized_comm == child);
    assert(find_group(child) != NULL);
    int resolve_count = fake_finalize_resolve_count;
    clear_wrapper_pointers();
    assert(ncclCommFinalize(parent) == ncclSuccess);
    assert(fake_finalize_resolve_count == resolve_count + 1);
    assert(ncclCommAbort(child) == ncclSuccess);
    assert(fake_last_aborted_comm == child);
    assert(find_group(child) == NULL);

    child = NULL;
    assert(ncclCommSplit(parent, 1, 8, &child, &config) == ncclSuccess);
    fake_last_destroyed_comm = NULL;
    assert(ncclCommDestroy(child) == ncclSuccess);
    assert(fake_last_destroyed_comm == child);
    assert(find_group(child) == NULL);
    assert(find_group(parent) != NULL);
    printf("split communicator lifecycle: PASS\n");
}

void nonblocking_communicator_lifecycle_test(void) {
    reset_groups();
    ncclComm_t parent = NULL;
    ncclComm_t child = NULL;
    ncclUniqueId id = {{0}};
    ncclConfig_t config = {0};

    fake_init_result = ncclInProgress;
    assert(ncclCommInitRankConfig(&parent, 4, id, 2, &config) == ncclInProgress);
    assert(g_comm == parent);
    assert(find_group(parent) != NULL);
    assert(find_group(parent)->rank == 2);
    assert(find_group(parent)->nranks == 4);
    fake_init_result = ncclSuccess;

    fake_split_result = ncclInProgress;
    fake_split_queries_ready = 0;
    assert(ncclCommSplit(parent, 1, 7, &child, &config) == ncclInProgress);
    assert(child != NULL);
    assert(find_group(child) != NULL);
    assert(find_group(child)->rank == -1);
    assert(get_or_create_group(child) == NULL);
    fake_split_queries_ready = 1;
    assert(get_or_create_group(child) != NULL);
    assert(find_group(child)->rank == 1);
    assert(find_group(child)->nranks == 2);
    fake_last_destroyed_comm = NULL;
    assert(ncclCommDestroy(child) == ncclSuccess);
    assert(fake_last_destroyed_comm == child);
    assert(find_group(child) == NULL);
    assert(find_group(parent) != NULL);
    fake_split_result = ncclSuccess;

    child = (ncclComm_t)0x1;
    assert(ncclCommSplit(parent, NCCL_SPLIT_NOCOLOR, 0, &child, &config) == ncclSuccess);
    assert(child == NULL);

    assert(ncclCommAbort(parent) == ncclSuccess);
    printf("nonblocking communicator lifecycle: PASS\n");
}

void stale_primary_destroy_mapping_test(void) {
    reset_groups();
    ncclComm_t stale = NULL;
    ncclComm_t live = NULL;
    ncclUniqueId id = {{0}};
    ncclConfig_t config = {0};

    fake_use_alternate_comm = 0;
    assert(ncclCommInitRankConfig(&stale, 4, id, 2, &config) == ncclSuccess);
    assert(ncclCommDestroy(stale) == ncclSuccess);
    assert(g_comm == NULL);

    fake_use_alternate_comm = 1;
    assert(ncclCommInitRankConfig(&live, 4, id, 2, &config) == ncclSuccess);
    assert(live != stale);
    assert(g_comm == live);
    fake_last_destroyed_comm = NULL;
    assert(ncclCommDestroy(stale) == ncclSuccess);
    assert(fake_last_destroyed_comm == live);
    assert(g_comm == NULL);
    fake_use_alternate_comm = 0;
    printf("stale primary destroy mapping: PASS\n");
}

void collective_uses_supplied_comm_test(void) {
    reset_groups();
    ncclComm_t parent = NULL;
    ncclComm_t child = NULL;
    ncclUniqueId id = {{0}};
    ncclConfig_t config = {0};

    assert(ncclCommInitRankConfig(&parent, 4, id, 2, &config) == ncclSuccess);
    assert(ncclCommSplit(parent, 1, 7, &child, &config) == ncclSuccess);
    g_comm = parent;
    fake_last_allreduce_comm = NULL;
    assert(ncclAllReduce(NULL, NULL, 0, ncclInt8, ncclSum, child, NULL) == ncclSuccess);
    assert(fake_last_allreduce_comm == child);
    printf("collective communicator routing: PASS\n");
}

void primary_reinit_after_abort_test(void) {
    reset_groups();
    ncclComm_t comm = NULL;
    ncclUniqueId id = {{0}};
    ncclConfig_t config = {0};

    assert(ncclCommInitRankConfig(&comm, 4, id, 2, &config) == ncclSuccess);
    assert(g_comm == comm);
    assert(ncclCommAbort(comm) == ncclSuccess);
    assert(g_comm == NULL);

    comm = NULL;
    assert(ncclCommInitRankConfig(&comm, 4, id, 2, &config) == ncclSuccess);
    assert(g_comm == comm);
    printf("primary communicator abort and re-init: PASS\n");
}

int main() {

    get_or_create_group_test(TEST_COMM);
    reset_groups_test(TEST_COMM);
    increment_seq_and_wait_nockpt_test(TEST_COMM);
    increment_seq_and_wait_ckpt_test(TEST_COMM);
    multi_group_test(TEST_COMM, TEST_COMM_2);
    configured_init_test();
    split_lifecycle_test();
    collective_uses_supplied_comm_test();
    primary_reinit_after_abort_test();
    nonblocking_communicator_lifecycle_test();
    stale_primary_destroy_mapping_test();
    printf("\nAll tests passed.\n");
    return 0;
}
