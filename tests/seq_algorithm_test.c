#include <stdio.h>
#include <assert.h>
#include <pthread.h>
#include <unistd.h>
#include "nccl_wrappers.h"


// gcc -o test_nccl_wrappers tests/seq_algorithm_test.c src/nccl_wrappers.c tests/stubs.c -I include -lpthread ./test_nccl_wrappers
extern volatile int ckpt_pending;
 
#define TEST_COMM ((ncclComm_t)0x1)
#define TEST_COMM_2 ((ncclComm_t)0x2)

void get_or_create_group_test(ncclComm_t comm) {
    reset_groups();
    group_entry_t *g = get_or_create_group(comm);
    assert(g->seq == 0);
    assert(g->target == 0);
    assert(get_or_create_group(comm) == g);
    printf("get_or_create_group: PASS\n");
}

void reset_groups_test(ncclComm_t comm){
    reset_groups();
    group_entry_t *g = get_or_create_group(comm);
    g->seq = 10;
    g->target = 5;
    reset_groups();
    g = get_or_create_group(comm);
    assert(g->seq == 0);
    assert(g->target == 0);
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

void multi_group_test(ncclComm_t comm1, ncclComm_t comm2){
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

int main() {
    
    get_or_create_group_test(TEST_COMM);
    reset_groups_test(TEST_COMM);
    increment_seq_and_wait_nockpt_test(TEST_COMM);
    increment_seq_and_wait_ckpt_test(TEST_COMM);
    multi_group_test(TEST_COMM, TEST_COMM_2);
    printf("\nAll tests passed.\n");
    return 0;

}
