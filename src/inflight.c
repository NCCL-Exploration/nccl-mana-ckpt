#include "inflight.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

/* ── Local pending table ─────────────────────────────────────────── */

pending_entry_t local_pending[MAX_RANKS][MAX_RANKS];

void inflight_init(void) {
    memset(local_pending, 0, sizeof(local_pending));
}

void inflight_mark_send(int src, int dst, size_t bytes) {
    local_pending[src][dst].bytes += bytes;
}

void inflight_mark_recv(int src, int dst, size_t bytes) {
    local_pending[src][dst].bytes -= bytes;
}

int inflight_has_pending(int src, int dst) {
    return local_pending[src][dst].bytes > 0;
}

int64_t inflight_pending_bytes(int src, int dst) {
    return local_pending[src][dst].bytes;
}

/* ── Per-source drain queue ──────────────────────────────────────── */

drain_queue_t drain_queues[MAX_RANKS];

int drain_queue_push(int source, void *data, size_t bytes) {
    drain_queue_t *q = &drain_queues[source];
    if (q->count >= MAX_DRAIN_QUEUE) {
        fprintf(stderr, "[inflight] drain queue full for source %d\n", source);
        return 0;
    }
    drain_msg_t *slot = &q->msgs[q->tail];
    slot->data   = data;
    slot->bytes  = bytes;
    slot->source = source;
    q->tail = (q->tail + 1) % MAX_DRAIN_QUEUE;
    q->count++;
    return 1;
}

drain_msg_t *drain_queue_peek(int source) {
    drain_queue_t *q = &drain_queues[source];
    if (q->count == 0) return NULL;
    return &q->msgs[q->head];
}

void drain_queue_pop(int source) {
    drain_queue_t *q = &drain_queues[source];
    if (q->count == 0) return;
    /* Free the host-side data buffer */
    if (q->msgs[q->head].data) {
        free(q->msgs[q->head].data);
        q->msgs[q->head].data = NULL;
    }
    q->head = (q->head + 1) % MAX_DRAIN_QUEUE;
    q->count--;
}

int drain_queue_empty(int source) {
    return drain_queues[source].count == 0;
}

void drain_queues_reset(void) {
    for (int i = 0; i < MAX_RANKS; i++) {
        /* Free any remaining drained data */
        while (!drain_queue_empty(i)) {
            drain_queue_pop(i);
        }
    }
    memset(drain_queues, 0, sizeof(drain_queues));
}
