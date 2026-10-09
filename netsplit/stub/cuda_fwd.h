/*
 * Forwarding a lower-half CUDA call to the upper half's driver (GPUDirect
 * RDMA, NETSPLIT_GDR=1).  Shared by the lower half's CUDA stand-ins:
 * cudart_stub.c (aws-ofi-nccl's libcudart.so.13) and cuda_fwd.c (libfabric's
 * libcuda.so.1, libcudart.so and libnvidia-ml.so.1).
 *
 * The back end runs on a lower-half TLS that the proxy borrowed for the
 * calling NCCL thread, and the proxy records that thread's own FS in
 * info->fs_map.  fwd_enter() switches to it; between fwd_enter() and
 * fwd_leave() only upper-half driver functions (info->cuda_fn) may be called:
 * no lower-half libc, no __thread variable, no stack protector (its canary
 * is in the TLS: build with -fno-stack-protector).
 */
#ifndef NETSPLIT_CUDA_FWD_H
#define NETSPLIT_CUDA_FWD_H

#include <asm/prctl.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/syscall.h>

#include "netsplit-api.h"

static NetsplitInfo_t *fwd_info;

/* Called by the lower half's backend_load() once the library is loaded. */
__attribute__((visibility("default"))) void netsplit_fwd_attach(NetsplitInfo_t *info) {
    fwd_info = info;
}

static inline int fwd_on(void) { return fwd_info != NULL && fwd_info->gdr; }

/* No libc: its syscall() can set errno, in whichever TLS is current. */
static inline long fwd_arch_prctl(long code, unsigned long arg) {
    long ret;
    __asm__ volatile("syscall"
                     : "=a"(ret)
                     : "0"((long)SYS_arch_prctl), "D"(code), "S"(arg)
                     : "rcx", "r11", "memory");
    return ret;
}

static inline unsigned long fwd_getfs(void) {
    unsigned long fs = 0;
    if (fwd_info->fsgsbase_enabled)
        __asm__ volatile("rex.W\n rdfsbase %0" : "=r"(fs)::"memory");
    else
        fwd_arch_prctl(ARCH_GET_FS, (unsigned long)&fs);
    return fs;
}

static inline void fwd_setfs(unsigned long fs) {
    if (fwd_info->fsgsbase_enabled)
        __asm__ volatile("rex.W\n wrfsbase %0" ::"r"(fs) : "memory");
    else
        fwd_arch_prctl(ARCH_SET_FS, fs);
}

/* Reports, once, a CUDA call from a thread that did not enter through the
 * proxy: it has no upper-half FS to run the driver on. */
static __attribute__((unused)) void fwd_orphan(const char *what) {
    static int reported;
    if (!__atomic_exchange_n(&reported, 1, __ATOMIC_RELAXED))
        fprintf(stderr, "[netsplit-lh] CUDA call %s from a thread that did not enter through "
                        "the net proxy: refused\n", what);
}

/* NETSPLIT_FWD_VERBOSE=1: the first forwarded calls, on stderr; 2: all. */
static __attribute__((unused)) void fwd_trace(const char *what, unsigned long lh) {
    static int verbose = -1, count;
    if (verbose < 0) {
        const char *e = getenv("NETSPLIT_FWD_VERBOSE");
        verbose = e == NULL ? 0 : e[0] == '2' ? 2 : e[0] == '1';
    }
    if (verbose == 2 || (verbose && __atomic_fetch_add(&count, 1, __ATOMIC_RELAXED) < 200))
        fprintf(stderr, "[netsplit-lh] forward %s (lower FS %#lx)\n", what, lh);
}

/* Switches to the calling thread's upper-half FS; returns the lower-half FS
 * to give fwd_leave(), or 0 (then nothing changed). */
static inline unsigned long fwd_enter(const char *what) {
    if (!fwd_on())
        return 0;
    unsigned long lh = fwd_getfs(), uh = 0;
    fwd_trace(what, lh);
    for (int i = 0; i < fwd_info->fs_map_len && uh == 0; i++)
        if (__atomic_load_n(&fwd_info->fs_map[i].lhfs, __ATOMIC_ACQUIRE) == lh)
            uh = fwd_info->fs_map[i].uhfs;
    if (uh == 0) {
        fwd_orphan(what);
        return 0;
    }
    fwd_setfs(uh);
    return lh;
}

static inline void fwd_leave(unsigned long lh) { fwd_setfs(lh); }

/* The upper half's driver function 'sym' (NS_CU_ 'idx'), typed by cuda.h. */
#define REAL(idx, sym) ((__typeof__(&sym))fwd_info->cuda_fn[NS_CU_##idx])

/* return (type) sym(args) in the upper half. */
#define FORWARD(type, idx, sym, ...)                                                       \
    do {                                                                                   \
        unsigned long lh_ = fwd_enter(#sym);                                               \
        if (lh_ == 0)                                                                      \
            return (type)CUDA_ERROR_NOT_INITIALIZED;                                       \
        CUresult r_ = REAL(idx, sym) != NULL ? REAL(idx, sym)(__VA_ARGS__)                 \
                                             : CUDA_ERROR_NOT_SUPPORTED;                   \
        fwd_leave(lh_);                                                                    \
        return (type)r_;                                                                   \
    } while (0)

#endif /* NETSPLIT_CUDA_FWD_H */
