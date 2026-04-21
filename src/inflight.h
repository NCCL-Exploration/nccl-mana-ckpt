#ifndef INFLIGHT_H
#define INFLIGHT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_RANKS       64
#define MAX_DRAIN_QUEUE 16

/* ── Local in-flight tracking ────────────────────────────────────── */

typedef struct {
    size_t bytes;  /* >0 = send posted but recv not yet matched; 0 = idle */
} pending_entry_t;

/* Process-local table: local_pending[src][dst].bytes */
extern pending_entry_t local_pending[MAX_RANKS][MAX_RANKS];

void   inflight_init(void);
void   inflight_mark_send(int src, int dst, size_t bytes);
void   inflight_mark_recv(int src, int dst);
int    inflight_has_pending(int src, int dst);
size_t inflight_pending_bytes(int src, int dst);

/* ── Per-source drain queue ──────────────────────────────────────── */

typedef struct {
    void  *data;   /* host-side copy of the drained message */
    size_t bytes;
    int    source;
} drain_msg_t;

typedef struct {
    drain_msg_t msgs[MAX_DRAIN_QUEUE];
    int head;
    int tail;
    int count;
} drain_queue_t;

/* One queue per possible source rank */
extern drain_queue_t drain_queues[MAX_RANKS];

int         drain_queue_push(int source, void *data, size_t bytes);
drain_msg_t *drain_queue_peek(int source);
void        drain_queue_pop(int source);
int         drain_queue_empty(int source);
void        drain_queues_reset(void);

#ifdef __cplusplus
}
#endif

#endif /* INFLIGHT_H */
