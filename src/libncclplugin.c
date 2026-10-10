#include "dmtcp.h"
#include "dmtcp_kvdb_c.h"
#include "inflight.h"
#include "nccl_wrappers.h"
#include "vcomm.h"
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <nccl.h>
#include <pthread.h>
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
#define POLL_US 200
#define MAX_CHASE_ROUNDS 32 /* consensus rounds in which nobody waited for progress */

static cudaStream_t g_active_streams[MAX_STREAMS];
static int g_num_streams = 0;

int g_nranks = 0;
int g_my_rank = -1;
volatile int ckpt_pending = 0;

/*
 * Checkpoint epoch, bumped at the top of every presuspend. Every peer receives
 * every DMT_DO_CHECKPOINT, so these counters stay in lockstep, and a restart
 * restores them all from images taken at the same epoch.
 *
 * This tag is not cosmetic. The coordinator wipes the KVDB only when the first
 * RESTARTING worker connects -- not on a plain checkpoint-and-resume -- while
 * presuspend destroys communicators on EVERY checkpoint. Without the epoch, the
 * accumulating MAX target keys would still hold the previous checkpoint's
 * high-water mark after seq resets to 0, and the second checkpoint of a run
 * would never converge.
 */
static unsigned g_epoch = 0;

static void pre_init_cuda(void);
void add_stream(cudaStream_t stream) {
    int i;
    for (i = 0; i < g_num_streams; i++) {
        if (g_active_streams[i] == stream)
            return;
    }
    if (g_num_streams < MAX_STREAMS) {
        g_active_streams[g_num_streams++] = stream;
    }
}

/*
 * Latch job-level identity from the first cross-process communicator. This is
 * only used for logging and for job-wide rank-0 roles; a communicator's own
 * rank and size live in its registry entry, because a sub-communicator has a
 * different rank space entirely.
 */
void init_rank_info(vcomm_t *v) {
    if (g_my_rank == -1 && v->origin == VCOMM_REMOTE) {
        g_my_rank = v->rank;
        g_nranks = v->nranks;
        printf("[NCCL Plugin] job rank %d of %d\n", g_my_rank, g_nranks);
        fflush(stdout);
    }
}

/*
 * The two NCCL entry points the plugin calls but does not interpose. Linking
 * them directly leaves undefined symbols the loader can only satisfy from the
 * global scope, which holds no NCCL when the application dlopened it
 * RTLD_LOCAL -- so the first rebuild dies on a symbol lookup error.
 */
static const char *nccl_err_str(ncclResult_t r) {
    const char *(*fn)(ncclResult_t);
    *(void **)&fn = nccl_real_dlsym("ncclGetErrorString");
    return (fn != NULL) ? fn(r) : "<ncclGetErrorString unavailable>";
}

static ncclResult_t nccl_get_unique_id(ncclUniqueId *id) {
    ncclResult_t (*fn)(ncclUniqueId *);
    *(void **)&fn = nccl_real_dlsym("ncclGetUniqueId");
    return (fn != NULL) ? fn(id) : ncclInternalError;
}

/* ── KVDB keys ────────────────────────────────────────────────────── */

/*
 * One flat "nccl-mana" namespace. Identity goes in as 32 fixed-width hex chars,
 * so keys are fixed-length and greppable in the coordinator's JSON dump.
 *
 * Every key is epoch-tagged, and every rank index is a rank IN THAT
 * COMMUNICATOR'S rank space -- never a job-global rank.
 */
#define KVDB_KEY_MAX 96

/*
 * Silent snprintf truncation would collapse two distinct keys into one, which is
 * the same cross-communicator aliasing this whole change exists to remove. So
 * truncation is fatal rather than ignored.
 */
#define KEYF(buf, ...)                                                                             \
    do {                                                                                           \
        int _n = snprintf((buf), KVDB_KEY_MAX, __VA_ARGS__);                                       \
        if (_n < 0 || _n >= KVDB_KEY_MAX) {                                                        \
            fprintf(stderr, "[NCCL PLUGIN] FATAL: kvdb key truncated at %s:%d\n", __FILE__,        \
                    __LINE__);                                                                     \
            fflush(stderr);                                                                        \
            abort();                                                                               \
        }                                                                                          \
    } while (0)

static void key_target(char *buf, const vcomm_t *v) {
    char hex[VCOMM_IDENT_HEX];
    vcomm_ident_hex(v->ident, hex);
    KEYF(buf, "C/%s/%u/t", hex, g_epoch);
}

static void key_p2p(char *buf, const vcomm_t *v, int src, int dst, char kind) {
    char hex[VCOMM_IDENT_HEX];
    vcomm_ident_hex(v->ident, hex);
    KEYF(buf, "C/%s/%u/p/%d/%d/%c", hex, g_epoch, src, dst, kind);
}

static void key_uid(char *buf, const vcomm_t *v) {
    char hex[VCOMM_IDENT_HEX];
    vcomm_ident_hex(v->ident, hex);
    KEYF(buf, "C/%s/%u/id", hex, g_epoch);
}

/* One rank's p2p counters on one comm: sent_to[] then recvd_from[]. */
static void key_p2p_blob(char *buf, const vcomm_t *v, int rank) {
    char hex[VCOMM_IDENT_HEX];
    vcomm_ident_hex(v->ident, hex);
    KEYF(buf, "C/%s/%u/q/%d", hex, g_epoch, rank);
}

static void key_round(char *buf, int round, const char *what) {
    KEYF(buf, "R/%u/%d/%s", g_epoch, round, what);
}

/* ── epoch agreement ──────────────────────────────────────────────── */

/*
 * Confirms every peer is on the same epoch. The one failure mode of a
 * process-local counter is a process that somehow missed a presuspend, which
 * would otherwise show up as an unexplainable non-convergence; this turns it
 * into a printed line.
 *
 * A peer that does not load this plugin (a launcher process, say) never reaches
 * the barrier, so the checkpoint hangs here; that configuration cannot work
 * anyway, since every barrier below needs every peer.
 *
 * This used to compare the votes with dmtcp_get_coordinator_status()'s peer
 * count, but that call takes DMTCP's wrapper lock exclusively while the
 * application is running. It deadlocked twice in tests/multi/p2p_pending.cu:
 * against a thread blocked in a rank pair's first ncclRecv (which holds the
 * lock shared until the peer joins, and the gate holds the peer), and, called
 * after the quiesce, against a CUDA thread that held the dynamic loader's lock
 * while waiting for the wrapper lock, which the status call's dlsym() needed.
 */
static void epoch_vote(void) {
    char key[KVDB_KEY_MAX];
    char barrier[64];

    KEYF(key, "R/%u/votes", g_epoch);
    dmtcp_kvdb_incr(key, 1);

    snprintf(barrier, sizeof(barrier), "nccl-mana/%u/epoch-vote", g_epoch);
    dmtcp_global_barrier(barrier);
}

/* ── checkpoint cut: collectives and point-to-point ───────────────── */

/*
 * Drive every communicator to a consistent cut.
 *
 * Each round: read peers' published positions, let this rank's application
 * threads catch up to them, then publish our own position and report whether we
 * are settled. Repeat until no rank anywhere was unsettled.
 *
 * A position has two parts:
 *  - per communicator, its collective sequence number, MAX-accumulated into one
 *    key;
 *  - per communicator and rank, one blob with that rank's cumulative p2p byte
 *    counters (sent_to[], then recvd_from[]). NCCL has no wildcard receive, and
 *    a pair's sends and receives match in posting order, so the counters say
 *    exactly which operations still lack their counterpart. A rank whose counter
 *    is below what its peer has posted owes that peer an operation.
 * A rank that is behind on either passes the gate (nccl_gate_closed_locked), and
 * its application keeps running until it has caught up, while every other
 * rank's threads wait at the gate. So a receive whose send is not posted yet, or
 * the reverse, completes with its real data before the checkpoint, and the drain
 * finds nothing to do.
 *
 * Rounds are needed because catching up on one communicator can require issuing
 * collectives on another -- cc_complex does 1 AllReduce on comm_world, 2 on
 * comm_A and 1 on comm_B per iteration, so a rank told to advance comm_A may hit
 * comm_B first and overshoot its target -- and because posting a send can
 * require receiving from a third rank first. The overshoot is published in the
 * next round and the cut simply moves forward; targets are monotone, so this is
 * the fixpoint iteration of a monotone function and terminates.
 *
 * Catching up takes as long as the application computes before its next NCCL
 * call, so each round waits for this rank's threads for one bounded slice, then
 * publishes and votes anyway. Waiting until caught up instead would deadlock a
 * rank whose catch-up blocks on a receive its peer cannot know about: the peer
 * learns of it only from this rank's next publication.
 *
 * Communicator lifecycle is part of the cut too. ncclCommSplit and the first of
 * ncclCommFinalize/ncclCommDestroy are steps on a comm's sequence, and a
 * destroyed comm stays as a tombstone (VCOMM_DESTROYED) that keeps publishing
 * its final sequence, so a peer still holding it catches up to its own destroy.
 * A creation by uniqueId names no members, so each rank publishes how many
 * creations it has in flight; while any rank has one, every gate is open
 * (g_creation_pending) until it completes. Tombstones are freed once the
 * consensus converges: by then every member has destroyed the comm.
 *
 * Point-to-point catch-up waits as long as it takes, like collective catch-up,
 * and reports every REPORT_EVERY_MS what each rank still owes. Only rounds in
 * which nobody waited count against MAX_CHASE_ROUNDS. NCCL_PLUGIN_P2P_WAIT=N
 * gives up after N seconds of waiting (0: no catch-up at all); the drain then
 * completes what is left. A pending send is received into host memory and
 * replayed, which is correct. A pending receive could only get zeros, so every
 * rank aborts instead (refuse_zero_fill), unless NCCL_PLUGIN_P2P_ZERO_FILL=1.
 */
#define SLICE_MIN_MS 10
#define SLICE_MAX_MS 1000
#define REPORT_EVERY_MS 30000

static int g_p2p_chase;                       /* p2p catch-up active in this checkpoint */
static int64_t g_p2p_blob[2 * VCOMM_MAX_RANKS];
static int64_t g_p2p_start[MAX_VCOMMS][2];    /* bytes sent / received when it began */

static int p2p_npeers(const vcomm_t *v) {
    return v->nranks < VCOMM_MAX_RANKS ? v->nranks : VCOMM_MAX_RANKS;
}

static int64_t p2p_total(const int64_t *a, int n) {
    int64_t s = 0;
    int i;
    for (i = 0; i < n; i++)
        s += a[i];
    return s;
}

/* NCCL_PLUGIN_P2P_WAIT in ms; -1 when unset: wait as long as it takes. */
static long p2p_wait_ms(void) {
    const char *s = getenv("NCCL_PLUGIN_P2P_WAIT");
    char *end;
    long v;
    if (s == NULL || *s == '\0')
        return -1;
    v = strtol(s, &end, 10);
    if (*end != '\0' || v < 0) {
        fprintf(stderr, "[NCCL PLUGIN] NCCL_PLUGIN_P2P_WAIT=%s is not a number of seconds;"
                        " waiting as long as it takes\n",
                s);
        fflush(stderr);
        return -1;
    }
    return v * 1000L;
}

/* NCCL_PLUGIN_P2P_ZERO_FILL=1: complete a receive that was never matched with
 * zeros rather than abort (refuse_zero_fill). */
static int zero_fill_allowed(void) {
    const char *s = getenv("NCCL_PLUGIN_P2P_ZERO_FILL");
    return s != NULL && strcmp(s, "1") == 0;
}

static long ms_since(const struct timeval *t0) {
    struct timeval t;
    gettimeofday(&t, NULL);
    return (t.tv_sec - t0->tv_sec) * 1000L + (t.tv_usec - t0->tv_usec) / 1000L;
}

/* Publish this rank's p2p counters on one communicator. Caller holds the seq
 * lock, so they cannot change between this and the settled test. */
static void publish_p2p_blob(const vcomm_t *v) {
    char key[KVDB_KEY_MAX];
    int n = p2p_npeers(v);
    memcpy(g_p2p_blob, v->sent_to, n * sizeof(int64_t));
    memcpy(g_p2p_blob + n, v->recvd_from, n * sizeof(int64_t));
    key_p2p_blob(key, v, v->rank);
    if (!dmtcp_kvdb_set_blob(key, g_p2p_blob, 2 * n * sizeof(int64_t))) {
        fprintf(stderr, "[NCCL PLUGIN] cannot publish p2p counters (%s)\n", key);
        fflush(stderr);
    }
}

/* Read each peer's side of every pair into this rank's targets. A peer that has
 * not published yet keeps its target unknown (-1). Caller holds the seq lock. */
static void read_p2p_targets(vcomm_t *v) {
    char key[KVDB_KEY_MAX];
    int n = p2p_npeers(v), peer;
    for (peer = 0; peer < n; peer++) {
        if (peer == v->rank)
            continue;
        key_p2p_blob(key, v, peer);
        if (dmtcp_kvdb_get_blob(key, g_p2p_blob, 2 * n * sizeof(int64_t))) {
            v->send_target[peer] = g_p2p_blob[n + v->rank]; /* peer's recvd_from[us] */
            v->recv_target[peer] = g_p2p_blob[v->rank];     /* peer's sent_to[us] */
        }
    }
}

/* 1 if this rank still owes progress: a collective below its target, or (while
 * the catch-up is on) an unmatched p2p operation of a peer. Caller holds the
 * seq lock. */
static int behind_locked(const int *order, int n, int *p2p_owed) {
    int i, behind = 0;
    *p2p_owed = 0;
    for (i = 0; i < n; i++) {
        vcomm_t *v = &g_vcomms[order[i]];
        if (v->target >= 0 && v->seq < v->target)
            behind = 1;
        if (g_p2p_chase && v->origin == VCOMM_REMOTE && vcomm_p2p_owes(v))
            behind = *p2p_owed = 1;
    }
    return behind;
}

/* What this rank still owes, for a hung or abandoned catch-up. */
static void report_owed(const char *why) {
    int order[MAX_VCOMMS];
    int n, i, peer;
    nccl_seq_lock();
    n = vcomm_order(order);
    for (i = 0; i < n; i++) {
        vcomm_t *v = &g_vcomms[order[i]];
        char hex[VCOMM_IDENT_HEX];
        if (v->origin != VCOMM_REMOTE)
            continue;
        vcomm_ident_hex(v->ident, hex);
        if (v->target >= 0 && v->seq < v->target)
            fprintf(stderr, "[NCCL PLUGIN] %s: comm %s rank %d/%d is at collective %ld of %ld\n",
                    why, hex, v->rank, v->nranks, (long)v->seq, (long)v->target);
        for (peer = 0; peer < p2p_npeers(v); peer++) {
            if (peer == v->rank)
                continue;
            if (v->send_target[peer] > v->sent_to[peer])
                fprintf(stderr,
                        "[NCCL PLUGIN] %s: comm %s rank %d owes peer %d sends of %ld B"
                        " (its receives are posted)\n",
                        why, hex, v->rank, peer, (long)(v->send_target[peer] - v->sent_to[peer]));
            if (v->recv_target[peer] > v->recvd_from[peer])
                fprintf(stderr,
                        "[NCCL PLUGIN] %s: comm %s rank %d owes peer %d receives of %ld B"
                        " (its sends are posted)\n",
                        why, hex, v->rank, peer,
                        (long)(v->recv_target[peer] - v->recvd_from[peer]));
        }
    }
    nccl_seq_unlock();
    fflush(stderr);
}

static void sequence_consensus(void) {
    char key[KVDB_KEY_MAX];
    char barrier[80];
    int order[MAX_VCOMMS];
    int round, chase_rounds = 0, announced = 0;
    long slice_ms = SLICE_MIN_MS, waited_ms = 0, p2p_waited_ms = 0, reported_ms = 0;
    long p2p_limit_ms = p2p_wait_ms();
    int n, i;

    g_p2p_chase = p2p_limit_ms != 0;
    nccl_seq_lock();
    for (i = 0; i < MAX_VCOMMS; i++) {
        vcomm_t *v = &g_vcomms[i];
        g_p2p_start[i][0] = p2p_total(v->sent_to, VCOMM_MAX_RANKS);
        g_p2p_start[i][1] = p2p_total(v->recvd_from, VCOMM_MAX_RANKS);
    }
    nccl_seq_unlock();

    for (round = 0;; round++) {
        int settled, behind, p2p_owed, creating;
        int64_t unsettled, waiting, p2p_waiting, creations;
        struct timeval t0;

        /* 1. Adopt peers' targets. */
        nccl_seq_lock();
        n = vcomm_order(order);
        for (i = 0; i < n; i++) {
            vcomm_t *v = &g_vcomms[order[i]];
            int64_t published;
            if (v->origin == VCOMM_LOCAL_CLIQUE) {
                /* All ranks live in this process, so consensus is local. */
                v->target = v->seq;
                continue;
            }
            key_target(key, v);
            published = dmtcp_kvdb_get(key);
            /* 0 means "nobody published yet"; values carry a +1 bias so a
             * legitimate sequence of 0 is distinguishable from a missing key. */
            if (published > 0 && published - 1 > v->target)
                v->target = published - 1;
            if (g_p2p_chase && v->origin == VCOMM_REMOTE)
                read_p2p_targets(v);
        }
        behind = behind_locked(order, n, &p2p_owed);
        nccl_seq_unlock();
        creating = nccl_creating_poll();
        if (p2p_owed && !announced) {
            printf("[NCCL Plugin] rank %d: point-to-point operations of peers are waiting for"
                   " this rank; letting the application post their counterparts\n",
                   g_my_rank);
            fflush(stdout);
            announced = 1;
        }

        /* 2. Let our own application threads catch up -- for one slice at most.
         *    The gate lets them through precisely while this rank is behind. */
        gettimeofday(&t0, NULL);
        while ((behind || creating > 0 || nccl_activity() != 0) && ms_since(&t0) < slice_ms) {
            usleep(POLL_US);
            nccl_seq_lock();
            behind = behind_locked(order, n, &p2p_owed);
            nccl_seq_unlock();
            creating = nccl_creating_poll();
        }

        /* 3. Publish, and decide "settled" under the same lock -- otherwise a
         *    thread can slip one more operation through between the decision
         *    and the barrier, leaving it unmatched. */
        nccl_seq_lock();
        settled = 1;
        for (i = 0; i < n; i++) {
            vcomm_t *v = &g_vcomms[order[i]];
            if (v->origin == VCOMM_LOCAL_CLIQUE)
                continue;
            key_target(key, v);
            dmtcp_kvdb_max(key, v->seq + 1);
            if (v->target < 0 || v->seq != v->target)
                settled = 0;
            if (g_p2p_chase && v->origin == VCOMM_REMOTE) {
                publish_p2p_blob(v);
                if (!vcomm_p2p_balanced(v))
                    settled = 0;
            }
        }
        /* A thread inside a call or an open group region can still post, and a
         * creation in flight still has members to wait for. */
        if (nccl_activity() != 0)
            settled = 0;
        behind = behind_locked(order, n, &p2p_owed);
        nccl_seq_unlock();
        creating = nccl_creating_poll();
        if (creating > 0)
            settled = 0;

        /* 4. Global vote. Reached unconditionally by every peer, including one
         *    holding no communicators at all, so the barrier always releases. */
        key_round(key, round, "unsettled");
        dmtcp_kvdb_incr(key, settled ? 0 : 1);
        key_round(key, round, "waiting");
        dmtcp_kvdb_incr(key, behind ? 1 : 0);
        key_round(key, round, "p2p");
        dmtcp_kvdb_incr(key, p2p_owed ? 1 : 0);
        key_round(key, round, "creating");
        dmtcp_kvdb_incr(key, creating);
        snprintf(barrier, sizeof(barrier), "nccl-mana/%u/round/%d", g_epoch, round);
        dmtcp_global_barrier(barrier);

        key_round(key, round, "unsettled");
        unsettled = dmtcp_kvdb_get(key);
        key_round(key, round, "waiting");
        waiting = dmtcp_kvdb_get(key);
        key_round(key, round, "p2p");
        p2p_waiting = dmtcp_kvdb_get(key);
        key_round(key, round, "creating");
        creations = dmtcp_kvdb_get(key);
        nccl_seq_lock();
        g_creation_pending = creations > 0;
        nccl_seq_unlock();

        if (unsettled == 0) {
            int64_t sent = 0, recvd = 0;
            nccl_seq_lock();
            for (i = 0; i < MAX_VCOMMS; i++) {
                sent += p2p_total(g_vcomms[i].sent_to, VCOMM_MAX_RANKS) - g_p2p_start[i][0];
                recvd += p2p_total(g_vcomms[i].recvd_from, VCOMM_MAX_RANKS) - g_p2p_start[i][1];
            }
            /* Every member of each destroyed comm has destroyed it now. */
            vcomm_free_tombstones();
            nccl_seq_unlock();
            printf("[NCCL Plugin] rank %d: sequence consensus converged in %d round(s)\n",
                   g_my_rank, round + 1);
            if (sent > 0 || recvd > 0)
                printf("[NCCL Plugin] rank %d: posted %ld B of sends and %ld B of receives"
                       " while catching up\n",
                       g_my_rank, (long)sent, (long)recvd);
            fflush(stdout);
            return;
        }

        /* Every rank reads the same votes, so every decision below agrees. A
         * creation in flight is waiting for members, not a chase. */
        if (waiting > 0 || creations > 0) {
            waited_ms += slice_ms;
            if (p2p_waiting > 0)
                p2p_waited_ms += slice_ms;
            if (slice_ms < SLICE_MAX_MS)
                slice_ms = slice_ms * 2 < SLICE_MAX_MS ? slice_ms * 2 : SLICE_MAX_MS;
        } else if (++chase_rounds >= MAX_CHASE_ROUNDS) {
            break;
        }

        if (g_p2p_chase && p2p_limit_ms >= 0 && p2p_waited_ms >= p2p_limit_ms) {
            /* The application did not post the counterparts in the time it was
             * given. Stop waiting for them; the drain completes what is left,
             * or every rank aborts (refuse_zero_fill). */
            if (p2p_owed)
                report_owed("p2p catch-up abandoned");
            nccl_seq_lock();
            for (i = 0; i < MAX_VCOMMS; i++)
                if (g_vcomms[i].origin != VCOMM_FREE)
                    vcomm_reset_p2p_targets(&g_vcomms[i]);
            g_p2p_chase = 0;
            nccl_seq_unlock();
            if (g_my_rank <= 0) {
                fprintf(stderr,
                        "[NCCL PLUGIN] WARNING: point-to-point operations still lacked their"
                        " counterparts after %ld s (NCCL_PLUGIN_P2P_WAIT). Pending sends are"
                        " received into host memory and replayed; a pending receive aborts the"
                        " job unless NCCL_PLUGIN_P2P_ZERO_FILL=1\n",
                        p2p_waited_ms / 1000);
                fflush(stderr);
            }
        } else if (behind && waited_ms - reported_ms >= REPORT_EVERY_MS) {
            reported_ms = waited_ms;
            report_owed("checkpoint still waiting");
        }
    }

    fprintf(stderr, "[NCCL PLUGIN] sequence consensus did not converge in %d rounds:\n",
            MAX_CHASE_ROUNDS);
    n = vcomm_order(order);
    for (i = 0; i < n; i++) {
        vcomm_t *v = &g_vcomms[order[i]];
        char hex[VCOMM_IDENT_HEX];
        vcomm_ident_hex(v->ident, hex);
        fprintf(stderr, "              comm %s rank %d/%d seq %ld target %ld\n", hex, v->rank,
                v->nranks, (long)v->seq, (long)v->target);
    }
    report_owed("no convergence");
    fflush(stderr);
    abort();
}

/* ── quiesce ──────────────────────────────────────────────────────── */

/*
 * Wait until no application thread is inside a NCCL call or an open group
 * region. The gate is already closed by this point, so no thread can enter, and
 * the checkpoint thread becomes the only submitter on every communicator --
 * which is what makes the drain's send/recv safe to post.
 */
static void quiesce_nccl(void) {
    while (nccl_activity() != 0)
        usleep(POLL_US);
}

/* ── p2p drain ────────────────────────────────────────────────────── */

static void publish_p2p_counters(void) {
    char key[KVDB_KEY_MAX];
    int order[MAX_VCOMMS];
    int n = vcomm_order(order);
    int i, peer;

    for (i = 0; i < n; i++) {
        vcomm_t *v = &g_vcomms[order[i]];
        if (v->origin == VCOMM_LOCAL_CLIQUE)
            continue;
        for (peer = 0; peer < v->nranks && peer < VCOMM_MAX_RANKS; peer++) {
            if (peer == v->rank)
                continue;
            /* "bytes src sent to dst" and "bytes dst received from src", both
             * biased by 1 so a missing key stays distinguishable from zero. */
            key_p2p(key, v, v->rank, peer, 's');
            dmtcp_kvdb_set(key, v->sent_to[peer] + 1);
            key_p2p(key, v, peer, v->rank, 'r');
            dmtcp_kvdb_set(key, v->recvd_from[peer] + 1);
        }
    }
}

/*
 * Drain one communicator.
 *
 * An imbalance is asymmetric by construction: owed_in > 0 means the peer's
 * application already enqueued a send we never received, and owed_out > 0 means
 * the peer already enqueued a recv we never satisfied. Exactly one side acts, and
 * its counterpart is an operation already sitting in NCCL rather than something
 * the peer's checkpoint thread has to do in step with us. So no per-communicator
 * rendezvous is needed here -- which is just as well, since dmtcp_global_barrier
 * cannot synchronize a subset of peers.
 */
/*
 * One pair's imbalance from the counters publish_p2p_counters() wrote: owed_in
 * bytes the peer sent that this rank has not received, owed_out bytes the peer
 * posted to receive that this rank has not sent. Returns 0 if the peer published
 * nothing: both keys are written before the barrier that precedes this read, so
 * that is a protocol error, and treating it as a zero count would fabricate an
 * imbalance.
 */
static int pair_owed(const vcomm_t *v, int peer, int64_t *owed_in, int64_t *owed_out) {
    char key[KVDB_KEY_MAX];
    int64_t sent_by_peer, recvd_by_peer;

    key_p2p(key, v, peer, v->rank, 's');
    sent_by_peer = dmtcp_kvdb_get(key);
    key_p2p(key, v, v->rank, peer, 'r');
    recvd_by_peer = dmtcp_kvdb_get(key);
    if (sent_by_peer == 0 || recvd_by_peer == 0)
        return 0;
    /* Undo the +1 publication bias. Our own side is read locally rather than
     * round-tripped through the coordinator. */
    *owed_in = (sent_by_peer - 1) - v->recvd_from[peer];
    *owed_out = (recvd_by_peer - 1) - v->sent_to[peer];
    return 1;
}

/*
 * A receive whose send the application never posted can only be completed with
 * zeros: its data is lost and nothing tells the application. Unless
 * NCCL_PLUGIN_P2P_ZERO_FILL=1 asks for that, every rank aborts together, before
 * the drain posts anything, and the ranks that owe the sends say which. The vote
 * and barrier make the abort collective, so no rank is left in a later barrier
 * waiting for one that died. Reached by every rank unconditionally.
 */
static void refuse_zero_fill(void) {
    char key[KVDB_KEY_MAX];
    char barrier[80];
    int order[MAX_VCOMMS];
    int n = vcomm_order(order), k, peer;
    int64_t mine = 0, total;
    int allowed = zero_fill_allowed();

    for (k = 0; k < n; k++) {
        vcomm_t *v = &g_vcomms[order[k]];
        if (v->real == NULL || v->origin != VCOMM_REMOTE)
            continue;
        for (peer = 0; peer < v->nranks && peer < VCOMM_MAX_RANKS; peer++) {
            int64_t owed_in, owed_out;
            char hex[VCOMM_IDENT_HEX];
            if (peer == v->rank || !pair_owed(v, peer, &owed_in, &owed_out) || owed_out <= 0)
                continue;
            mine++;
            if (allowed)
                continue; /* drain_comm() warns about each one it zero-fills */
            vcomm_ident_hex(v->ident, hex);
            fprintf(stderr,
                    "[NCCL PLUGIN] ERROR: comm %s rank %d: peer %d posted a receive of %ld B"
                    " that this rank never sent\n",
                    hex, v->rank, peer, (long)owed_out);
        }
    }
    fflush(stderr);

    KEYF(key, "R/%u/unmatched-recvs", g_epoch);
    dmtcp_kvdb_incr(key, mine);
    snprintf(barrier, sizeof(barrier), "nccl-mana/%u/unmatched-recvs", g_epoch);
    dmtcp_global_barrier(barrier);
    total = dmtcp_kvdb_get(key);
    if (total == 0 || allowed)
        return;

    fprintf(stderr,
            "[NCCL PLUGIN] ERROR: rank %d: %ld point-to-point receive(s) in the job were never"
            " matched by a send, and completing them would hand the application zeros.\n"
            "              Aborting the checkpoint and the job. Set NCCL_PLUGIN_P2P_ZERO_FILL=1"
            " to complete them with zeros instead.\n",
            g_my_rank, (long)total);
    fflush(stderr);
    abort();
}

static void drain_comm(vcomm_t *v) {
    ncclComm_t real = v->real;
    int peer;

    if (real == NULL)
        return;

    CUDA_CHECK(cudaSetDevice(v->cuda_dev));

    for (peer = 0; peer < v->nranks && peer < VCOMM_MAX_RANKS; peer++) {
        int64_t owed_in, owed_out;

        if (peer == v->rank)
            continue;

        if (!pair_owed(v, peer, &owed_in, &owed_out)) {
            char hex[VCOMM_IDENT_HEX];
            vcomm_ident_hex(v->ident, hex);
            fprintf(stderr,
                    "[NCCL PLUGIN] comm %s rank %d: peer %d never published its p2p counters;"
                    " skipping drain for this pair\n",
                    hex, v->rank, peer);
            fflush(stderr);
            continue;
        }

        if (owed_in > 0) {
            /* The peer enqueued this send; consume it now and stash the payload
             * so the application's eventual ncclRecv can be satisfied from the
             * host copy after restart. */
            void *dev_buf = NULL;
            void *host_buf = NULL;
            size_t bytes = (size_t)owed_in;

            printf("[NCCL Drain] comm rank %d: draining recv of %zu bytes from %d\n", v->rank,
                   bytes, peer);
            fflush(stdout);

            CUDA_CHECK(cudaMalloc(&dev_buf, bytes));
            call_real_ncclRecv(dev_buf, bytes, ncclInt8, peer, real, 0);
            CUDA_CHECK(cudaStreamSynchronize(0));

            host_buf = malloc(bytes);
            if (host_buf == NULL) {
                fprintf(stderr, "[NCCL PLUGIN] out of memory stashing a drained message\n");
                fflush(stderr);
                abort();
            }
            CUDA_CHECK(cudaMemcpy(host_buf, dev_buf, bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaFree(dev_buf));

            if (!drain_log_push(v->ident, peer, host_buf, bytes)) {
                /* Dropping it would silently lose an application message. */
                fprintf(stderr,
                        "[NCCL PLUGIN] drain log full (MAX_DRAIN_MSGS=%d); cannot stash the"
                        " message from peer %d\n",
                        MAX_DRAIN_MSGS, peer);
                fflush(stderr);
                abort();
            }
            v->recvd_from[peer] += owed_in;
        }

        if (owed_out > 0) {
            /* The peer already enqueued a matching recv. It has to be satisfied
             * or the communicator cannot be destroyed.
             *
             * The payload is zero-filled, so the peer's buffer receives zeros
             * rather than the data its application expected: the real data was
             * never handed to NCCL. sequence_consensus() normally prevents this
             * by letting this rank's application post the send first; getting
             * here means that catch-up gave up (NCCL_PLUGIN_P2P_WAIT) and
             * NCCL_PLUGIN_P2P_ZERO_FILL=1 asked for this over an abort. */
            void *dev_buf = NULL;
            size_t bytes = (size_t)owed_out;

            printf("[NCCL Drain] comm rank %d: draining send of %zu bytes to %d\n", v->rank, bytes,
                   peer);
            fprintf(stderr,
                    "[NCCL PLUGIN] WARNING: comm rank %d: peer %d's pending receive of %zu bytes"
                    " gets zeros, not the data this rank had yet to send\n",
                    v->rank, peer, bytes);
            fflush(stdout);
            fflush(stderr);

            CUDA_CHECK(cudaMalloc(&dev_buf, bytes));
            CUDA_CHECK(cudaMemset(dev_buf, 0, bytes));
            call_real_ncclSend(dev_buf, bytes, ncclInt8, peer, real, 0);
            CUDA_CHECK(cudaStreamSynchronize(0));
            CUDA_CHECK(cudaFree(dev_buf));
            v->sent_to[peer] += owed_out;
        }
    }
}

/* ── teardown ─────────────────────────────────────────────────────── */

/*
 * Destroy every communicator so the CUDA driver holds no CUDA IPC / P2P handles
 * (or GDR mappings from connection setup) when the CUDA plugin snapshots the
 * process.
 *
 * Reverse identity order, mirroring the rebuild. Uses the real symbol directly:
 * the interposed ncclCommDestroy also frees the registry slot, which here would
 * throw away the very metadata the rebuild needs.
 */
static ncclResult_t wait_comm_ready(ncclComm_t real);

/*
 * Two passes, both in vcomm_order (the same on every rank): finalize every
 * communicator, then destroy them. ncclCommFinalize is NCCL's orderly,
 * collective shutdown -- it completes outstanding operations and flushes the
 * proxy's connections with the peers -- so no rank frees its side while a peer
 * is still using it. Destroy then only releases local resources.
 */
static void teardown_all_comms(void) {
    int order[MAX_VCOMMS];
    ncclComm_t reals[MAX_VCOMMS];
    int n = vcomm_order(order);
    int k;

    for (k = n - 1; k >= 0; k--) {
        vcomm_t *v = &g_vcomms[order[k]];
        ncclResult_t r;
        char hex[VCOMM_IDENT_HEX];

        reals[k] = v->real;
        if (reals[k] == NULL)
            continue;

        vcomm_ident_hex(v->ident, hex);
        printf("[NCCL Plugin] teardown: finalizing comm %s (rank %d/%d)\n", hex, v->rank,
               v->nranks);
        fflush(stdout);

        /* Cleared first, so any application thread arriving at a wrapper blocks
         * in wait_real() instead of using a handle we are about to destroy. */
        v->real = NULL;
        if (v->finalized) {
            /* The application finalized it already, and NCCL rejects a second
             * finalize; only the destroy below is left. */
            r = ncclSuccess;
        } else {
            r = call_real_ncclCommFinalize(reals[k]);
            if (r == ncclInProgress) /* non-blocking communicator */
                r = wait_comm_ready(reals[k]);
        }
        if (r != ncclSuccess) {
            fprintf(stderr, "[NCCL PLUGIN] teardown: ncclCommFinalize failed: %s\n",
                    nccl_err_str(r));
            fflush(stderr);
        }
    }

    for (k = n - 1; k >= 0; k--) {
        ncclResult_t r;

        if (reals[k] == NULL)
            continue;
        r = call_real_ncclCommDestroy(reals[k]);
        if (r != ncclSuccess) {
            fprintf(stderr, "[NCCL PLUGIN] teardown: ncclCommDestroy failed: %s\n",
                    nccl_err_str(r));
            fflush(stderr);
        }
    }
}

/* ── checkpoint ───────────────────────────────────────────────────── */

static void checkpoint(void) {
    struct timeval start, end;
    double elapsed;
    int order[MAX_VCOMMS];
    int n, k;
    char barrier[80];

    gettimeofday(&start, NULL);

    g_epoch++;
    ckpt_pending = 1;
    __sync_synchronize();

    /*
     * Every phase below runs on every peer unconditionally, including a process
     * that has not created a communicator yet -- its loops are simply empty. The
     * previous code returned early in that case and skipped barriers its peers
     * still reached, which hangs the job whenever ranks initialize at different
     * times.
     */
    epoch_vote();

    /* Freeze the collective and point-to-point frontier at a consistent cut. */
    sequence_consensus();

    /* From here the checkpoint thread is the only submitter. */
    quiesce_nccl();

    if (vcomm_live_count() > 0)
        pre_init_cuda();

    /* Snapshot p2p state, then drain against it. After the consensus every pair
     * is balanced and the drain finds nothing, unless the p2p catch-up gave up;
     * a "[NCCL Drain]" line otherwise means the cut was not consistent. */
    publish_p2p_counters();
    snprintf(barrier, sizeof(barrier), "nccl-mana/%u/p2p-published", g_epoch);
    dmtcp_global_barrier(barrier);
    refuse_zero_fill();

    n = vcomm_order(order);
    for (k = 0; k < n; k++)
        drain_comm(&g_vcomms[order[k]]);

    /*
     * Only now can the streams complete: the drain has posted the counterparts
     * for every unmatched operation. Syncing before the drain -- as the previous
     * code did, and under an inverted condition at that -- waits on exactly the
     * operations that cannot finish.
     */
    if (vcomm_live_count() > 0) {
        CUDA_CHECK(cudaStreamSynchronize(0));
        for (k = 0; k < g_num_streams; k++)
            cudaStreamSynchronize(g_active_streams[k]);
    }

    /* Nobody may destroy a communicator while a peer is still draining it. */
    snprintf(barrier, sizeof(barrier), "nccl-mana/%u/drained", g_epoch);
    dmtcp_global_barrier(barrier);

    teardown_all_comms();

    gettimeofday(&end, NULL);
    elapsed = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0;
    /* Parsed by tests/nas-benchmarks/IS/run_perlmutter_ckpt.sh -- keep the
     * format. */
    printf("[NCCL PLUGIN] Checkpoint overhead: %.6f seconds\n", elapsed);
    fflush(stdout);
}

/* ── rebuild ──────────────────────────────────────────────────────── */

/*
 * Rebuild a process-local clique with a single ncclCommInitAll.
 *
 * ncclCommInitRank "must be called by different threads/processes or use
 * ncclGroupStart/ncclGroupEnd", so reconstructing an N-rank clique as N
 * sequential InitRank calls from this one thread would hang. Being
 * process-local, it also needs no rendezvous at all.
 */
static void rebuild_clique(int clique) {
    ncclComm_t tmp[MAX_VCOMMS];
    int devlist[MAX_VCOMMS];
    vcomm_t *members[MAX_VCOMMS];
    int count = 0;
    int i;
    ncclResult_t r;

    for (i = 0; i < MAX_VCOMMS; i++) {
        vcomm_t *v = &g_vcomms[i];
        if (v->origin == VCOMM_LOCAL_CLIQUE && v->clique == clique)
            members[count++] = v;
    }
    if (count == 0)
        return;

    memset(devlist, 0, sizeof(devlist));
    for (i = 0; i < count; i++) {
        if (members[i]->rank < 0 || members[i]->rank >= count) {
            fprintf(stderr, "[NCCL PLUGIN] clique %d has an out-of-range rank %d\n", clique,
                    members[i]->rank);
            fflush(stderr);
            abort();
        }
        /* ncclCommInitAll returns comms in devlist order, so indexing by rank
         * reproduces the original rank-to-device mapping exactly. */
        devlist[members[i]->rank] = members[i]->cuda_dev;
    }

    r = call_real_ncclCommInitAll(tmp, count, devlist);
    if (r != ncclSuccess) {
        fprintf(stderr, "[NCCL PLUGIN] rebuild: ncclCommInitAll failed: %s\n",
                nccl_err_str(r));
        fflush(stderr);
        exit(1);
    }
    for (i = 0; i < count; i++)
        members[i]->real = tmp[members[i]->rank];
}

/*
 * Recreate every communicator after a checkpoint or restart.
 *
 * The registry survives as ordinary restored memory, so {identity, nranks, rank,
 * device, config} are all still here; only `real` went stale. Membership is
 * implicit in the ncclCommInitRank rendezvous, so nothing about how the
 * communicator was originally created needs replaying -- a comm made by
 * ncclCommSplit is rebuilt by exactly the same call as one made by
 * ncclCommInitRank.
 *
 * The old ncclUniqueId is useless: it encodes the bootstrap address of the
 * original root. Each comm's member 0 generates a fresh one and publishes it
 * under the comm's identity. Publishing is non-blocking, so ONE global barrier
 * covers every communicator at once, and no per-comm rendezvous is needed.
 */
static void rebuild_publish_ids(void) {
    char key[KVDB_KEY_MAX];
    int order[MAX_VCOMMS];
    int n, k;

    n = vcomm_order(order);

    /* Phase 1: publish a fresh id for each comm we are member 0 of. */
    for (k = 0; k < n; k++) {
        vcomm_t *v = &g_vcomms[order[k]];
        ncclUniqueId id;
        ncclResult_t r;

        if (v->origin != VCOMM_REMOTE || v->real != NULL || v->rank != 0)
            continue;

        r = nccl_get_unique_id(&id);
        if (r != ncclSuccess) {
            fprintf(stderr, "[NCCL PLUGIN] rebuild: ncclGetUniqueId failed: %s\n",
                    nccl_err_str(r));
            fflush(stderr);
            exit(1);
        }
        key_uid(key, v);
        if (!dmtcp_kvdb_set_blob(key, &id, sizeof(id))) {
            fprintf(stderr, "[NCCL PLUGIN] rebuild: could not publish a fresh ncclUniqueId\n");
            fflush(stderr);
            exit(1);
        }
    }
}

/*
 * The stored config is replayed verbatim, so a communicator the application
 * created with blocking = 0 is rebuilt non-blocking too and comes back
 * ncclInProgress: it exists but NCCL is still finishing it in the background.
 * v->real must not be published until that completes -- the store is what
 * releases application threads from wait_real(), and NCCL rejects a comm used
 * before its init returned ncclSuccess.
 *
 * The wait is unbounded, matching the blocking path: ncclCommInitRank is
 * collective, so a rebuild whose peers never arrive hangs either way.
 */
static ncclResult_t wait_comm_ready(ncclComm_t real) {
    for (;;) {
        ncclResult_t state = ncclSuccess;
        ncclResult_t r = call_real_ncclCommGetAsyncError(real, &state);
        if (r != ncclSuccess)
            return r;
        if (state != ncclInProgress)
            return state;
        usleep(POLL_US);
    }
}

/*
 * Fresh ids, fetched on the checkpoint thread.  dmtcp_kvdb_* is a coordinator
 * round-trip and is only legal there -- the same reason publication and the
 * barrier stay on that thread.  The worker only touches CUDA and NCCL.
 */
static ncclUniqueId g_rebuild_ids[MAX_VCOMMS];
static int g_rebuild_id_ok[MAX_VCOMMS];

static void rebuild_fetch_ids(void) {
    char key[KVDB_KEY_MAX];
    int order[MAX_VCOMMS];
    int n = vcomm_order(order);
    int k;

    for (k = 0; k < MAX_VCOMMS; k++)
        g_rebuild_id_ok[k] = 0;

    for (k = 0; k < n; k++) {
        int idx = order[k];
        vcomm_t *v = &g_vcomms[idx];

        if (v->real != NULL || v->origin != VCOMM_REMOTE)
            continue;
        key_uid(key, v);
        g_rebuild_id_ok[idx] =
            dmtcp_kvdb_get_blob(key, &g_rebuild_ids[idx], sizeof(g_rebuild_ids[idx]));
    }
}

static void rebuild_create_comms(void) {
    int order[MAX_VCOMMS];
    int n, k;

    n = vcomm_order(order);

    /*
     * Phase 2: rebuild in ascending identity order.
     *
     * ncclCommInitRank is collective and blocking, so ranks that share
     * communicators must agree on the order. Every member derives the same
     * identity, so each rank's list is a subsequence of one global total order,
     * and a rank can only ever be blocked on the minimal communicator whose
     * other members have therefore all reached it.
     */
    for (k = 0; k < n; k++) {
        vcomm_t *v = &g_vcomms[order[k]];
        ncclUniqueId id = g_rebuild_ids[order[k]];
        ncclComm_t real = NULL;
        ncclResult_t r;
        char hex[VCOMM_IDENT_HEX];

        if (v->real != NULL)
            continue; /* nothing was torn down, or a clique-mate did it */

        vcomm_ident_hex(v->ident, hex);

        if (v->origin == VCOMM_LOCAL_CLIQUE) {
            rebuild_clique(v->clique);
            continue;
        }

        /* NCCL binds a communicator to the CURRENT device, so this must be the
         * device the comm was originally created on -- not the job rank, which
         * is a different number whenever CUDA_VISIBLE_DEVICES is set per rank. */
        CUDA_CHECK(cudaSetDevice(v->cuda_dev));

        if (!g_rebuild_id_ok[order[k]]) {
            fprintf(stderr,
                    "[NCCL PLUGIN] rebuild: no fresh id published for comm %s"
                    " (rank %d/%d)\n",
                    hex, v->rank, v->nranks);
            fflush(stderr);
            exit(1);
        }

        printf("[NCCL Plugin] rebuild: comm %s rank %d/%d on device %d\n", hex, v->rank, v->nranks,
               v->cuda_dev);
        fflush(stdout);

        r = call_real_ncclCommInitRankConfig(&real, v->nranks, id, v->rank, vcomm_config(v));
        if (r == ncclInProgress && real != NULL)
            r = wait_comm_ready(real);
        if (r != ncclSuccess) {
            fprintf(stderr, "[NCCL PLUGIN] rebuild: ncclCommInitRank failed for %s: %s\n", hex,
                    nccl_err_str(r));
            fflush(stderr);
            exit(1);
        }

        /* Published last: this is what releases application threads parked in
         * wait_real(). */
        v->real = real;
    }
}

static void *rebuild_worker(void *arg) {
    (void)arg;
    rebuild_create_comms();
    return NULL;
}

static void rebuild_all_comms(void) {
    struct timeval start, end;
    char barrier[80];
    pthread_t worker;
    int rc;

    gettimeofday(&start, NULL);
    printf("[NCCL Plugin] rebuild: phase 1 cuda init\n");
    fflush(stdout);
    pre_init_cuda();

    printf("[NCCL Plugin] rebuild: phase 2 publish ids\n");
    fflush(stdout);
    rebuild_publish_ids();

    /* Unconditional on every peer: a process with no communicators still has to
     * reach this, or its peers block here forever. Barriers are only legal on
     * the checkpoint thread, which is why publication stays here. */
    printf("[NCCL Plugin] rebuild: phase 3 barrier\n");
    fflush(stdout);
    snprintf(barrier, sizeof(barrier), "nccl-mana/%u/ids-published", g_epoch);
    dmtcp_global_barrier(barrier);

    rebuild_fetch_ids();

    printf("[NCCL Plugin] rebuild: phase 4 worker thread\n");
    fflush(stdout);

    /*
     * Create the communicators on an ordinary thread rather than on this one.
     *
     * ncclCommInitRank crashes partway through its channel setup when called on
     * DMTCP's checkpoint thread after a restart, while the identical call from an
     * application thread succeeds -- verified by creating a communicator
     * post-restart with only the CUDA plugin loaded. So the checkpoint thread
     * keeps the parts that must run on it (KVDB publication and the barrier
     * above) and hands the NCCL work to a helper it joins.
     *
     * Joining matters: the rebuild has to finish before this event handler
     * returns, so that a checkpoint cannot begin against half-rebuilt state.
     */
    rc = pthread_create(&worker, NULL, rebuild_worker, NULL);
    if (rc != 0) {
        fprintf(stderr, "[NCCL PLUGIN] rebuild: pthread_create failed (%d)\n", rc);
        fflush(stderr);
        exit(1);
    }
    pthread_join(worker, NULL);

    gettimeofday(&end, NULL);
    /* Parsed by tests/report/ -- keep the format. */
    printf("[NCCL PLUGIN] Rebuild overhead: %.6f seconds\n",
           (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0);
    fflush(stdout);
}

/* ── DMTCP events ─────────────────────────────────────────────────── */

static void resume(void) {
    ckpt_pending = 0;
    vcomm_reset_seq();
    /*
     * The p2p byte counters are deliberately NOT reset. They are cumulative, the
     * drain has just equalized them across ranks, and a reset would have to
     * happen identically everywhere to keep the differences meaningful. Leaving
     * them alone is symmetric by construction.
     *
     * The drain log is not reset either -- see inflight.h.
     */
    printf("[NCCL Plugin] rank %d: resumed after checkpoint\n", g_my_rank);
    fflush(stdout);
}

static void restart(void) {
    int i;
    ckpt_pending = 0;
    g_num_streams = 0;
    /* Drop the dlsym results and re-resolve them immediately, here on the
     * checkpoint thread. Leaving them to be resolved lazily means the first
     * dlsym after a restart runs on whichever thread happens to call NCCL
     * first -- for the rebuild, a helper thread spawned seconds after the
     * address space was reconstructed -- and that faults. */
    clear_wrapper_pointers();
    resolve_wrapper_pointers();
    /* Teardown already nulled these, but a checkpoint image taken before this
     * code existed, or an aborted teardown, could leave one set. */
    for (i = 0; i < MAX_VCOMMS; i++) {
        if (g_vcomms[i].origin != VCOMM_FREE)
            g_vcomms[i].real = NULL;
    }
    vcomm_reset_seq();
    printf("[NCCL Plugin] restarting from checkpoint with %d communicator(s)\n",
           vcomm_live_count());
    fflush(stdout);
}

static void pre_init_cuda(void) {
    const char *exec_path = dmtcp_get_executable_path();
    int device_count = 0;
    cudaError_t err;

    if (exec_path != NULL) {
        const char *basename = strrchr(exec_path, '/');
        basename = basename ? basename + 1 : exec_path;
        /* Launcher processes have no GPU work; initializing CUDA in them is
         * both pointless and disruptive. */
        if (strcmp(basename, "srun") == 0 || strcmp(basename, "mpirun") == 0 ||
            strcmp(basename, "mpiexec") == 0 || strcmp(basename, "bash") == 0 ||
            strcmp(basename, "sh") == 0 || strcmp(basename, "orted") == 0 ||
            strcmp(basename, "hydra_pmi_proxy") == 0 || strcmp(basename, "slurmd") == 0 ||
            strcmp(basename, "slurmstepd") == 0 || strcmp(basename, "lower-half") == 0 ||
            strncmp(basename, "mana_", 5) == 0 || strncmp(basename, "dmtcp_", 6) == 0) {
            return;
        }
    }

    err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess) {
        printf("[NCCL Plugin] pre_init_cuda: cudaGetDeviceCount failed: %s\n",
               cudaGetErrorString(err));
        fflush(stdout);
        return;
    }
    if (device_count > 0) {
        /* Do not disturb the application's current device: if a communicator is
         * registered, its own recorded device is authoritative and the rebuild
         * sets it per comm. */
        cudaFree(0);
    }
}

/*
 * NCCL RAS (2.24+) is a per-process service that keeps its own TCP connections
 * to peer processes. They outlive every communicator, so teardown cannot close
 * them, and DMTCP's socket drain/handshake on such a connection deadlocked a
 * 2-node checkpoint. RAS is diagnostics only: turn it off unless the user chose
 * otherwise. DMTCP_EVENT_INIT runs before main(), so before NCCL reads it; the
 * setting is part of the environment that a restart restores.
 */
static void disable_nccl_ras(void) {
    const char *v = getenv("NCCL_RAS_ENABLE");
    if (v == NULL) {
        setenv("NCCL_RAS_ENABLE", "0", 0);
    } else if (strcmp(v, "0") != 0) {
        fprintf(stderr, "[NCCL PLUGIN] warning: NCCL_RAS_ENABLE=%s; RAS connections can "
                        "deadlock a multi-node checkpoint\n", v);
        fflush(stderr);
    }
}

static void nccl_event_hook(DmtcpEvent_t event, DmtcpEventData_t *data) {
    static int first_time = 1;
    (void)data;

    switch (event) {
    case DMTCP_EVENT_INIT:
        drain_log_init();
        disable_nccl_ras();
        break;

    case DMTCP_EVENT_RUNNING_AFTER:
        /*
         * The right place to rebuild, for two reasons: the CUDA plugin restores
         * GPU state in this same event (it needs user threads running), and this
         * event is dispatched in REVERSE plugin-registration order, so with
         * --with-plugin libncclplugin.so:libdmtcp_cuda.so the CUDA restore has
         * already happened by the time this runs. At DMTCP_EVENT_RESTART the
         * CUDA context is still dead.
         */
        if (first_time) {
            first_time = 0; /* initial launch: nothing to rebuild */
            break;
        }
        rebuild_all_comms();
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

DmtcpPluginDescriptor_t nccl_plugin = {DMTCP_PLUGIN_API_VERSION,
                                       DMTCP_PACKAGE_VERSION,
                                       "nccl",
                                       "DMTCP",
                                       "dmtcp@ccs.neu.edu",
                                       "NCCL checkpointing plugin",
                                       nccl_event_hook};

DMTCP_DECL_PLUGIN(nccl_plugin);
