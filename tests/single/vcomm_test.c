/*
 * Host-only unit test for the virtual communicator registry.
 *
 *   gcc -o test_vcomm tests/single/vcomm_test.c src/vcomm.c -I src -lpthread
 *   ./test_vcomm
 *
 * No GPU, no DMTCP, no NCCL runtime -- only <nccl.h> for the opaque types. This
 * covers the parts of the design whose failure modes are otherwise invisible
 * until a multi-rank restart hangs: identity agreement, rebuild ordering, and
 * the two-same-color-splits case.
 */

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

/* A distinct fake ncclUniqueId per seed, the way distinct comms get distinct
 * ids in a real run. */
static ncclUniqueId fake_id(int seed) {
    ncclUniqueId id;
    size_t i;
    memset(&id, 0, sizeof(id));
    for (i = 0; i < sizeof(id); i++)
        ((char *)&id)[i] = (char)(seed * 31 + (int)i);
    return id;
}

static vcomm_t *make_remote(int seed, int nranks, int rank) {
    uint8_t ident[VCOMM_IDENT_LEN];
    ncclUniqueId id = fake_id(seed);
    vcomm_t *v = NULL;
    vcomm_ident_from_id(&id, ident);
    if (vcomm_new(VCOMM_REMOTE, ident, &v) != VCOMM_OK)
        return NULL;
    v->nranks = nranks;
    v->rank = rank;
    v->cuda_dev = rank;
    return v;
}

/* ── handle encoding ──────────────────────────────────────────────── */

static int test_handle_roundtrip(void) {
    int i;
    registry_clear();

    for (i = 0; i < MAX_VCOMMS; i++) {
        vcomm_t *v = make_remote(i + 1, 2, 0);
        CHECK(v != NULL);
        CHECK(vcomm_index(v) == i);
        CHECK(vcomm_lookup(vcomm_handle(v)) == v);
    }

    /* A real NCCL handle, and NULL, must both read as "not ours" so the
     * wrapper forwards them verbatim. */
    CHECK(vcomm_lookup(NULL) == NULL);
    CHECK(vcomm_lookup((ncclComm_t)(uintptr_t)0x7f9c12340000ULL) == NULL);

    /* Tag matches but the slot is free -> still not ours. */
    vcomm_free(&g_vcomms[3]);
    CHECK(vcomm_lookup(VCOMM_HANDLE(3)) == NULL);
    /* ...and an out-of-range index must not be dereferenced. */
    CHECK(vcomm_lookup(VCOMM_HANDLE(MAX_VCOMMS)) == NULL);

    /* The handed-out value must be non-canonical, so a missed translation
     * faults instead of reaching real NCCL. An address is canonical on x86-64
     * only when bits 63:47 are all equal (bit 47 sign-extended); require that
     * they are not, for every slot. */
    for (i = 0; i < MAX_VCOMMS; i++) {
        uint64_t top = (uint64_t)(uintptr_t)VCOMM_HANDLE(i) >> 47;
        CHECK(top != 0x0 && top != 0x1ffff);
    }

    printf("handle encoding: PASS\n");
    return 0;
}

/* ── allocation ───────────────────────────────────────────────────── */

static int test_alloc(void) {
    uint8_t ident[VCOMM_IDENT_LEN];
    ncclUniqueId id = fake_id(7);
    vcomm_t *v = NULL, *v2 = NULL;
    int i;

    registry_clear();
    vcomm_ident_from_id(&id, ident);

    CHECK(vcomm_new(VCOMM_REMOTE, ident, &v) == VCOMM_OK);
    CHECK(v != NULL);
    /* Defaults that matter: target -1 means "no consensus yet" (0 would be a
     * legitimate sequence number), and real == NULL blocks wrappers until a
     * rebuild publishes a handle. */
    CHECK(v->target == -1);
    CHECK(v->seq == 0);
    CHECK(v->clique == -1);
    CHECK(v->real == NULL);
    CHECK(v->has_config == 0);

    /* Reusing one uniqueId for two comms is illegal; the registry must say so
     * rather than silently aliasing them into one KVDB key namespace. */
    CHECK(vcomm_new(VCOMM_REMOTE, ident, &v2) == VCOMM_ERR_DUPLICATE);
    CHECK(v2 == NULL);

    CHECK(vcomm_live_count() == 1);
    vcomm_free(v);
    CHECK(vcomm_live_count() == 0);

    /* Exhaustion is reported, not silently capped. */
    registry_clear();
    for (i = 0; i < MAX_VCOMMS; i++)
        CHECK(make_remote(i + 100, 2, 0) != NULL);
    CHECK(vcomm_live_count() == MAX_VCOMMS);
    {
        uint8_t extra[VCOMM_IDENT_LEN];
        ncclUniqueId eid = fake_id(999);
        vcomm_ident_from_id(&eid, extra);
        CHECK(vcomm_new(VCOMM_REMOTE, extra, &v2) == VCOMM_ERR_FULL);
    }

    /* Freeing a slot makes it reusable. */
    vcomm_free(&g_vcomms[5]);
    CHECK(make_remote(999, 2, 0) != NULL);
    CHECK(vcomm_live_count() == MAX_VCOMMS);

    printf("allocation: PASS\n");
    return 0;
}

/* ── identity ─────────────────────────────────────────────────────── */

static int test_identity(void) {
    uint8_t a[VCOMM_IDENT_LEN], b[VCOMM_IDENT_LEN], c[VCOMM_IDENT_LEN];
    ncclUniqueId id1 = fake_id(1), id2 = fake_id(2);
    char hex[VCOMM_IDENT_HEX];

    /* Deterministic: this is what makes all members agree with no traffic. */
    vcomm_ident_from_id(&id1, a);
    vcomm_ident_from_id(&id1, b);
    CHECK(memcmp(a, b, VCOMM_IDENT_LEN) == 0);

    /* Distinct ids -> distinct identities. */
    vcomm_ident_from_id(&id2, c);
    CHECK(memcmp(a, c, VCOMM_IDENT_LEN) != 0);

    /* A one-bit change anywhere in the id, including the tail, must change the
     * digest -- the id is 128 bytes and only part of it is the bootstrap addr. */
    {
        ncclUniqueId id3 = fake_id(1);
        ((char *)&id3)[sizeof(id3) - 1] ^= 0x01;
        vcomm_ident_from_id(&id3, b);
        CHECK(memcmp(a, b, VCOMM_IDENT_LEN) != 0);
    }

    vcomm_ident_hex(a, hex);
    CHECK(strlen(hex) == 32);

    printf("identity from uniqueId: PASS\n");
    return 0;
}

static int test_split_identity(void) {
    uint8_t parent[VCOMM_IDENT_LEN], other_parent[VCOMM_IDENT_LEN];
    uint8_t s0[VCOMM_IDENT_LEN], s1[VCOMM_IDENT_LEN], s2[VCOMM_IDENT_LEN];
    ncclUniqueId pid = fake_id(11), oid = fake_id(12);

    vcomm_ident_from_id(&pid, parent);
    vcomm_ident_from_id(&oid, other_parent);

    /* Same parent, same color, same split index -> same identity. This is the
     * agreement every member of the child relies on. */
    vcomm_ident_from_split(parent, 3, 0, s0);
    vcomm_ident_from_split(parent, 3, 0, s1);
    CHECK(memcmp(s0, s1, VCOMM_IDENT_LEN) == 0);

    /* THE case a parent-plus-color scheme gets wrong: splitting the same parent
     * twice with the same color yields two independent communicators, so the
     * split index must separate them. */
    vcomm_ident_from_split(parent, 3, 1, s2);
    CHECK(memcmp(s0, s2, VCOMM_IDENT_LEN) != 0);

    /* Different color -> different child. */
    vcomm_ident_from_split(parent, 4, 0, s2);
    CHECK(memcmp(s0, s2, VCOMM_IDENT_LEN) != 0);

    /* Different parent, same color/index -> different child. */
    vcomm_ident_from_split(other_parent, 3, 0, s2);
    CHECK(memcmp(s0, s2, VCOMM_IDENT_LEN) != 0);

    /* A split child must not collide with a root comm. */
    CHECK(memcmp(s0, parent, VCOMM_IDENT_LEN) != 0);

    /* Negative colors are legal ints; make sure the packing keeps them
     * distinct rather than folding onto another value. */
    vcomm_ident_from_split(parent, -1, 0, s0);
    vcomm_ident_from_split(parent, -2, 0, s2);
    CHECK(memcmp(s0, s2, VCOMM_IDENT_LEN) != 0);

    printf("identity from split: PASS\n");
    return 0;
}

static int test_local_identity(void) {
    uint8_t a[VCOMM_IDENT_LEN], b[VCOMM_IDENT_LEN];

    vcomm_ident_local(0, 0, a);
    vcomm_ident_local(0, 1, b);
    CHECK(memcmp(a, b, VCOMM_IDENT_LEN) != 0); /* per-rank within a clique */

    vcomm_ident_local(1, 0, b);
    CHECK(memcmp(a, b, VCOMM_IDENT_LEN) != 0); /* per-clique */

    vcomm_ident_local(0, 0, b);
    CHECK(memcmp(a, b, VCOMM_IDENT_LEN) == 0); /* stable across restart */

    printf("identity for local clique: PASS\n");
    return 0;
}

/* ── rebuild ordering ─────────────────────────────────────────────── */

static int test_order_sorted(void) {
    int order[MAX_VCOMMS];
    int n, i;

    registry_clear();
    for (i = 0; i < 5; i++)
        CHECK(make_remote(i + 20, 2, 0) != NULL);

    n = vcomm_order(order);
    CHECK(n == 5);
    for (i = 1; i < n; i++)
        CHECK(vcomm_ident_cmp(g_vcomms[order[i - 1]].ident, g_vcomms[order[i]].ident) < 0);

    printf("rebuild order is sorted: PASS\n");
    return 0;
}

/*
 * The property that makes the collective rebuild deadlock-free: two processes
 * holding overlapping subsets of communicators, created in different orders,
 * must still agree on the relative order of the comms they share.
 *
 * Models cc_complex: rank 0 has {world, A}, rank 1 has {world, A, B},
 * rank 2 has {world, B} -- and each created them in its own order.
 */
/* Position of a given identity in the agreed rebuild order, or -1. */
static int order_pos(const uint8_t ident[VCOMM_IDENT_LEN]) {
    int order[MAX_VCOMMS];
    int n = vcomm_order(order);
    int i;
    for (i = 0; i < n; i++) {
        if (vcomm_ident_cmp(g_vcomms[order[i]].ident, ident) == 0)
            return i;
    }
    return -1;
}

static int test_order_agreement(void) {
    const int world = 1, A = 2, B = 3;
    uint8_t iw[VCOMM_IDENT_LEN], ia[VCOMM_IDENT_LEN];
    ncclUniqueId idw = fake_id(world), ida = fake_id(A);
    int r0, r1, r2_has_no_a;
    int truth;

    vcomm_ident_from_id(&idw, iw);
    vcomm_ident_from_id(&ida, ia);
    /* The order every rank should agree on, independent of creation order. */
    truth = (vcomm_ident_cmp(iw, ia) < 0);

    /* rank 0 holds {world, A}, created world first. */
    registry_clear();
    CHECK(make_remote(world, 3, 0) != NULL);
    CHECK(make_remote(A, 2, 0) != NULL);
    CHECK(order_pos(iw) >= 0 && order_pos(ia) >= 0);
    r0 = (order_pos(iw) < order_pos(ia));

    /* rank 1 holds {world, A, B} and created them in a different order --
     * B first, then A, then world -- and holds a comm rank 0 never sees. */
    registry_clear();
    CHECK(make_remote(B, 2, 0) != NULL);
    CHECK(make_remote(A, 2, 1) != NULL);
    CHECK(make_remote(world, 3, 1) != NULL);
    CHECK(order_pos(iw) >= 0 && order_pos(ia) >= 0);
    r1 = (order_pos(iw) < order_pos(ia));

    /* rank 2 holds {world, B} -- it never sees A at all, which must not
     * perturb where world sits relative to the comms it does share. */
    registry_clear();
    CHECK(make_remote(world, 3, 2) != NULL);
    CHECK(make_remote(B, 2, 1) != NULL);
    r2_has_no_a = (order_pos(ia) == -1);

    CHECK(r0 == truth);
    CHECK(r1 == truth);
    CHECK(r2_has_no_a);

    printf("rebuild order agrees across ranks: PASS\n");
    return 0;
}

/* ── seq reset ────────────────────────────────────────────────────── */

static int test_reset_seq(void) {
    vcomm_t *v;
    uint8_t saved[VCOMM_IDENT_LEN];

    registry_clear();
    v = make_remote(42, 4, 2);
    CHECK(v != NULL);
    memcpy(saved, v->ident, VCOMM_IDENT_LEN);
    v->seq = 17;
    v->target = 17;

    vcomm_reset_seq();

    CHECK(v->seq == 0);
    CHECK(v->target == -1);
    /* Identity and the rebuild arguments must survive: the old code memset the
     * whole table at restart, so a rebuilt comm landed in a fresh slot with no
     * way to match it to its peers. */
    CHECK(memcmp(v->ident, saved, VCOMM_IDENT_LEN) == 0);
    CHECK(v->nranks == 4);
    CHECK(v->rank == 2);

    printf("seq reset preserves identity: PASS\n");
    return 0;
}

/* ── config ───────────────────────────────────────────────────────── */

static int test_config(void) {
    ncclConfig_t cfg = NCCL_CONFIG_INITIALIZER;
    char net[] = "Socket";
    char name[] = "my-pg";
    vcomm_t *v;
    ncclConfig_t *out;

    registry_clear();
    v = make_remote(77, 2, 0);
    CHECK(v != NULL);

    CHECK(vcomm_config(v) == NULL); /* no config -> rebuild with defaults */

    cfg.blocking = 0;
    cfg.splitShare = 1;
    cfg.netName = net;
    cfg.commName = name;
    vcomm_store_config(v, &cfg);
    CHECK(v->has_config == 1);

    /* The app's strings are gone by rebuild time, so the copy must not point
     * at them. */
    CHECK(v->config.netName == NULL);
    CHECK(v->config.commName == NULL);
    memset(net, 'x', sizeof(net) - 1);
    memset(name, 'x', sizeof(name) - 1);

    out = vcomm_config(v);
    CHECK(out != NULL);
    CHECK(out->blocking == 0);
    CHECK(out->splitShare == 1);
    CHECK(out->netName != NULL && strcmp(out->netName, "Socket") == 0);
    CHECK(out->commName != NULL && strcmp(out->commName, "my-pg") == 0);
    CHECK(out->size == sizeof(ncclConfig_t));
    CHECK(out->magic == cfg.magic);

    /* Overlong names are truncated, not overflowed. */
    {
        char big[VCOMM_NAME_MAX * 2];
        memset(big, 'a', sizeof(big) - 1);
        big[sizeof(big) - 1] = '\0';
        cfg.netName = big;
        cfg.commName = NULL;
        vcomm_store_config(v, &cfg);
        out = vcomm_config(v);
        CHECK(out != NULL);
        CHECK(strlen(out->netName) == VCOMM_NAME_MAX - 1);
        CHECK(out->commName == NULL);
    }

    printf("config capture: PASS\n");
    return 0;
}

int main(void) {
    if (test_handle_roundtrip())
        return 1;
    if (test_alloc())
        return 1;
    if (test_identity())
        return 1;
    if (test_split_identity())
        return 1;
    if (test_local_identity())
        return 1;
    if (test_order_sorted())
        return 1;
    if (test_order_agreement())
        return 1;
    if (test_reset_seq())
        return 1;
    if (test_config())
        return 1;

    printf("\nAll vcomm tests passed (%d checks).\n", g_checks);
    return 0;
}
