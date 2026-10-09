/*
 * The interface between netsplit's two halves in one process.
 *
 *   upper half  application, NCCL, CUDA, DMTCP and its plugins, and the net
 *               proxy (libnccl-net-netsplit.so); saved in the image.
 *   lower half  netsplit-lh: a kernel loader (from MANA) with its own ld.so
 *               and libc, NCCL's network back end (aws-ofi-nccl, libfabric)
 *               and the CUDA stub; never saved, started anew at restart.
 *
 * The lower half fills a NetsplitInfo and tells the upper half where it is:
 * at launch through NETSPLIT_LH_INFO_ADDR, at restart through the file
 * netsplit_lh_info_path().  Every function pointer here is lower-half code:
 * call it with a lower-half FS (TLS), see the comment of each, and never pass
 * an upper-half function pointer down (it would run on the wrong TLS).
 *
 * The first fields are those of MANA's LowerHalfInfo that the loader and the
 * memory arena use, in the same order.
 */
#ifndef NETSPLIT_API_H
#define NETSPLIT_API_H

#include <stddef.h>
#include <stdint.h>

#include "net_v6.h"

#ifdef __cplusplus
extern "C" {
#endif

#define NETSPLIT_API_VERSION 1
#define NETSPLIT_LH_INFO_ENV "NETSPLIT_LH_INFO_ADDR"

typedef char *VA;

typedef struct __MmapInfo {
    char *addr;
    size_t len;
} MmapInfo_t;

typedef struct NetsplitInfo {
    /* ---- MANA's LowerHalfInfo, minus MPI ---- */
    void *fsaddr; /* FS base of the lower half's main thread */
    int fsgsbase_enabled;
    /* The arena allocator for the upper half's memory (mem-wrapper.cpp);
     * called with the upper half's FS, they switch FS themselves. */
    void *mmap;
    void *munmap;
    /* std::vector<MmapInfo_t> &(*)(int *): the upper half's mappings. */
    void *mmap_list_fptr;
    /* Set by the upper half when it exits through exit(): see uhExit(). */
    int upper_half_exiting;
    char *uh_stack_start;
    char *uh_stack_end;
    char *uh_next_free_addr;
    /* Placeholder fds kept free for the upper half's fds at restart. */
    unsigned long reserved_fd_dev;
    unsigned long reserved_fd_ino;

    /* ---- netsplit ---- */
    unsigned int version; /* NETSPLIT_API_VERSION */

    /* Loads the CUDA stub and the back end (NETSPLIT_BACKEND, else
     * $NCCL_HOME/plugin/lib/libnccl-net.so) and fills 'net'.  0 on success;
     * on failure 'backend_error' says why.  Call with 'fsaddr', holding the
     * upper half's lock. */
    int (*backend_load)(void);
    const char *backend_error;
    /* The back end's ncclNet v6 table.  Call its functions on a borrowed
     * lower-half TLS (tls_acquire).  Its init() takes the lower half's own
     * logger: pass 'logger', never an upper-half function. */
    ncclNet_v6_t *net;
    ncclDebugLogger_t logger;

    /* The TLS of a parked lower-half thread, for one upper-half thread to
     * use while it runs lower-half code; returned by tls_release().  Call
     * both with 'fsaddr', holding the upper half's lock.  0 on success. */
    int (*tls_acquire)(unsigned long *fs);
    void (*tls_release)(unsigned long fs);
    int tls_parked; /* parked threads made so far */
    int tls_in_use;

    /* The CUDA stub's settings (stub/cudart_stub.c).  Call set_thread_device
     * on the borrowed TLS of the thread it is for; the others with
     * 'fsaddr'. */
    void (*stub_set_ndev)(int n);
    void (*stub_set_thread_device)(int dev);
    int (*stub_add_device_range)(void *ptr, size_t size, int dev);
    void (*stub_remove_device_range)(void *ptr);
} NetsplitInfo_t;

/* The file through which a restarted lower half publishes its NetsplitInfo:
 * /tmp/netsplit_lh_info_<hostname>_<pid>.  'buf' gets at most 'len' bytes. */
static inline void netsplit_lh_info_path(char *buf, size_t len, const char *host, long pid) {
    size_t n = 0;
    const char *prefix = "/tmp/netsplit_lh_info_";
    while (*prefix && n + 1 < len)
        buf[n++] = *prefix++;
    while (*host && n + 1 < len)
        buf[n++] = *host++;
    if (n + 1 < len)
        buf[n++] = '_';
    /* No snprintf: the upper half reads it while DMTCP starts up. */
    char digits[24];
    int nd = 0;
    if (pid == 0)
        digits[nd++] = '0';
    while (pid > 0 && nd < (int)sizeof(digits)) {
        digits[nd++] = (char)('0' + pid % 10);
        pid /= 10;
    }
    while (nd > 0 && n + 1 < len)
        buf[n++] = digits[--nd];
    buf[n] = '\0';
}

#ifdef __cplusplus
}
#endif
#endif /* NETSPLIT_API_H */
