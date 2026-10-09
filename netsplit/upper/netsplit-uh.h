/*
 * The upper half's side of netsplit, shared by the net proxy (netproxy.c)
 * and the DMTCP glue (netsplit_dmtcp.cpp), which are one library:
 * libnccl-net-netsplit.so.  DMTCP preloads it as a plugin, and NCCL loads it
 * as its net plugin (NCCL_NET_PLUGIN=netsplit); both get the same instance,
 * found by its SONAME.
 */
#ifndef NETSPLIT_UH_H
#define NETSPLIT_UH_H

#include <asm/prctl.h>
#include <stddef.h>
#include <sys/syscall.h>
#include <unistd.h>

#include "netsplit-api.h"

#ifdef __cplusplus
extern "C" {
#endif

/* The lower half's NetsplitInfo, or NULL when not started by netsplit-lh. */
extern NetsplitInfo_t *lh_info;

/* Finds lh_info: through NETSPLIT_LH_INFO_ADDR at launch, through the file
 * of netsplit-api.h at restart.  0 if found. */
int netsplit_lh_info_init(int restart);

/* The proxy's hooks for the DMTCP glue. */
void netproxy_restart(void);                     /* a new lower half */
long netproxy_live_handles(char *buf, size_t len); /* back-end handles alive */
/* The upper half's FS of the thread that runs on lower-half TLS 'lhfs', or 0:
 * a signal handler that interrupts lower-half code switches back with it. */
unsigned long netproxy_upper_fs_for(unsigned long lhfs);

static inline unsigned long ns_getfs(void) {
    unsigned long fs;
    if (lh_info != NULL && lh_info->fsgsbase_enabled)
        __asm__ volatile("rex.W\n rdfsbase %0" : "=r"(fs)::"memory");
    else
        syscall(SYS_arch_prctl, ARCH_GET_FS, &fs);
    return fs;
}

static inline void ns_setfs(unsigned long fs) {
    if (lh_info != NULL && lh_info->fsgsbase_enabled)
        __asm__ volatile("rex.W\n wrfsbase %0" ::"r"(fs) : "memory");
    else
        syscall(SYS_arch_prctl, ARCH_SET_FS, fs);
}

#ifdef __cplusplus
}
#endif
#endif /* NETSPLIT_UH_H */
