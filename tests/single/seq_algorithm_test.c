/*
 * Host-only test for the collective sequence gate.
 *
 *   make tests/cc
 *
 * The gate decides when an application thread may issue its next collective
 * while a checkpoint is pending. It is CONJUNCTIVE across communicators, and
 * that is the property worth pinning down: a per-communicator gate deadlocks a
 * program whose per-comm collective counts differ, because a rank told to
 * advance one communicator may have a different one next in program order.
 *
 * tests/multi/cc_complex.c is exactly that program -- 1 AllReduce on comm_world,
 * 2 on comm_A, 1 on comm_B per iteration -- so these cases model it directly.
 */

#include "nccl_wrappers.h"
#include "vcomm.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

static int g_checks;

#define CHECK(cond)                                                                                \
    do {                                                                                           \
        g_checks++;                                                                                \
        if (!(cond)) {                                                                             \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);                        \
            return 1;                                                                              \
        }                                                                                          \
    } while (0)

static void registry_clear(void) {
    int i;
    for (i = 0; i < MAX_VCOMMS; i++)
        vcomm_free(&g_vcomms[i]);
}

static vcomm_t *add_comm(int seed, int64_t seq, int64_t target) {
    uint8_t ident[VCOMM_IDENT_LEN];
    vcomm_t *v = NULL;
    size_t i;
    ncclUniqueId id;

    memset(&id, 0, sizeof(id));
    for (i = 0; i < sizeof(id); i++)
        ((char *)&id)[i] = (char)(seed * 7 + (int)i);
    vcomm_ident_from_id(&id, ident);

    if (vcomm_new(VCOMM_REMOTE, ident, &v) != VCOMM_OK)
        return NULL;
    v->nranks = 2;
    v->rank = 0;
    v->seq = seq;
    v->target = target;
    return v;
}

/* nccl_gate_closed_locked() expects the seq lock held. */
static int gate_closed(void) {
    int closed;
    nccl_seq_lock();
    closed = nccl_gate_closed_locked();
    nccl_seq_unlock();
    return closed;
}

static int test_no_checkpoint_pending(void) {
    registry_clear();
    ckpt_pending = 0;

    /* Nothing may be held when no checkpoint is in flight, whatever the
     * sequence numbers say. */
    CHECK(add_comm(1, 5, 5) != NULL);
    CHECK(!gate_closed());
    CHECK(add_comm(2, 0, 0) != NULL);
    CHECK(!gate_closed());

    printf("gate open with no checkpoint pending: PASS\n");
    return 0;
}

static int test_single_comm(void) {
    vcomm_t *v;

    registry_clear();
    ckpt_pending = 1;

    /* target == -1 means consensus has not run yet. That must count as
     * at-target, so the frontier freezes the instant a checkpoint is requested
     * rather than letting one more collective slip through. */
    v = add_comm(1, 3, -1);
    CHECK(v != NULL);
    CHECK(gate_closed());

    /* Behind target: this rank still owes progress, so it must be let through. */
    v->target = 5;
    CHECK(!gate_closed());

    v->seq = 4;
    CHECK(!gate_closed());

    /* At target: held. */
    v->seq = 5;
    CHECK(gate_closed());

    /* Overshoot, which the chase deliberately tolerates and absorbs in a later
     * round. Still held. */
    v->seq = 6;
    CHECK(gate_closed());

    printf("gate on a single communicator: PASS\n");
    return 0;
}

/*
 * The case that motivates the whole design. Rank holds three communicators; one
 * is behind. A per-comm gate would hold a thread at comm_A because comm_A is at
 * its target -- but the thread's next collective in program order is on comm_B,
 * which is behind, so holding it deadlocks. The conjunctive gate lets it
 * through.
 */
static int test_conjunctive(void) {
    vcomm_t *world, *a, *b;

    registry_clear();
    ckpt_pending = 1;

    world = add_comm(1, 12, 12);
    a = add_comm(2, 24, 24);
    b = add_comm(3, 10, 11); /* behind */
    CHECK(world != NULL && a != NULL && b != NULL);

    /* Some comm owes progress -> the thread may issue on ANY comm. */
    CHECK(!gate_closed());

    /* Once the laggard catches up, everything is held. */
    b->seq = 11;
    CHECK(gate_closed());

    /* Any single comm falling behind reopens the gate, regardless of which. */
    world->target = 13;
    CHECK(!gate_closed());
    world->seq = 13;
    CHECK(gate_closed());

    a->target = 26;
    CHECK(!gate_closed());
    a->seq = 26;
    CHECK(gate_closed());

    /* Unequal per-comm counts are the point: these targets are all different
     * and all satisfied simultaneously, which a single global counter cannot
     * express. */
    CHECK(world->seq == 13 && a->seq == 26 && b->seq == 11);

    printf("gate is conjunctive across communicators: PASS\n");
    return 0;
}

static int test_local_clique_is_local(void) {
    vcomm_t *v;
    uint8_t ident[VCOMM_IDENT_LEN];

    registry_clear();
    ckpt_pending = 1;

    /* A process-local clique has all its ranks in this process, so its target is
     * whatever it has already reached -- it can never hold the gate open waiting
     * for an absent peer. */
    vcomm_ident_local(0, 0, ident);
    CHECK(vcomm_new(VCOMM_LOCAL_CLIQUE, ident, &v) == VCOMM_OK);
    v->nranks = 1;
    v->rank = 0;
    v->seq = 4;
    v->target = 4;
    CHECK(gate_closed());

    printf("local clique does not stall the gate: PASS\n");
    return 0;
}

static int test_empty_registry(void) {
    registry_clear();
    ckpt_pending = 1;
    /*
     * A process with no communicator has no frontier to freeze, so the gate must
     * let it through. Holding it -- what the empty loop used to do -- deadlocks
     * against a peer already inside the creation rendezvous, whose raised
     * activity count spins quiesce_nccl() forever.
     *
     * Its checkpoint phases are still vacuous and it still reaches every
     * barrier; that is asserted by the end-to-end pre-init test, not here.
     */
    CHECK(!gate_closed());

    ckpt_pending = 0;
    CHECK(!gate_closed());

    printf("empty registry: PASS\n");
    return 0;
}

static int test_reset_reopens(void) {
    vcomm_t *v;

    registry_clear();
    ckpt_pending = 1;
    v = add_comm(1, 9, 9);
    CHECK(v != NULL);
    CHECK(gate_closed());

    /* What resume()/restart() do. Sequence state is cleared but identity is
     * kept, and clearing ckpt_pending is what actually reopens the gate. */
    vcomm_reset_seq();
    ckpt_pending = 0;
    CHECK(v->seq == 0);
    CHECK(v->target == -1);
    CHECK(!gate_closed());

    printf("reset reopens the gate: PASS\n");
    return 0;
}

static int test_activity_starts_idle(void) {
    /* The quiesce waits on this before the drain posts anything. */
    CHECK(nccl_activity() == 0);
    printf("activity count starts idle: PASS\n");
    return 0;
}

int main(void) {
    if (test_no_checkpoint_pending())
        return 1;
    if (test_single_comm())
        return 1;
    if (test_conjunctive())
        return 1;
    if (test_local_clique_is_local())
        return 1;
    if (test_empty_registry())
        return 1;
    if (test_reset_reopens())
        return 1;
    if (test_activity_starts_idle())
        return 1;

    printf("\nAll sequence-gate tests passed (%d checks).\n", g_checks);
    return 0;
}
