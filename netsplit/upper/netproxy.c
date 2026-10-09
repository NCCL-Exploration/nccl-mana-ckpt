/*
 * The net proxy: NCCL's net plugin (ncclNet v6) in the upper half, forwarding
 * every call to the network back end (aws-ofi-nccl) in netsplit's lower half.
 *
 * - Each call runs the back end on a lower-half TLS of its own: on its first
 *   call, a thread borrows the TLS of a parked lower-half thread
 *   (lh_info->tls_acquire) and keeps it until it exits (a pthread key
 *   destructor returns it).  So no two threads share a lower-half TLS, and
 *   the pool is reused.
 * - No upper-half function pointer goes down, except the CUDA driver's for
 *   GPUDirect RDMA (below): the back end logs through the lower half's own
 *   logger, and its CUDA calls go to the lower half's stub.
 * - Every back-end handle (comms, memory registrations, requests) belongs to
 *   a communicator, and the checkpoint plugin destroys every communicator
 *   before the checkpoint: the proxy counts them (netproxy_live_handles())
 *   and virtualizes none.
 * - An epoch counts lower halves: a restart starts a new one
 *   (netproxy_restart()), whose back end must be loaded and initialized
 *   again, and whose TLS each thread borrows again.
 * - Host staging (the default): getProperties reports host memory only, so
 *   NCCL never hands a GPU pointer to the network, and the back end needs no
 *   CUDA.
 * - GPUDirect RDMA (NETSPLIT_GDR=1): the back end's pointer support passes
 *   through, and libfabric gets CUDA through the lower half's forwarders
 *   (stub/cuda_fwd.c), which run this half's driver on the calling thread's
 *   own TLS.  The proxy gives them the driver's functions (lh_info->cuda_fn)
 *   and its table of borrowed TLS (lh_info->fs_map).
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "net_v6.h"
#include "netsplit-api.h"
#include "netsplit-uh.h"

static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
static ncclDebugLogger_t g_nccl_log; /* NCCL's logger: call it in the upper half only */

/* Lower halves seen, starting at 1.  A thread's TLS, the loaded back end and
 * its initialization are each tagged with the epoch they belong to. */
static volatile unsigned g_epoch = 1;
static unsigned g_loaded_epoch, g_inited_epoch;

static __thread unsigned long t_lhfs;
static __thread unsigned t_epoch;
static pthread_key_t g_tls_key;
static pthread_once_t g_key_once = PTHREAD_ONCE_INIT;

/* lower-half FS -> upper-half FS, for the signal wrappers and the lower
 * half's CUDA forwarders (no TLS needed). */
#define MAX_TLS_MAP 1024
static NetsplitFsPair g_tls_map[MAX_TLS_MAP];

/* Live back-end handles. */
static long g_n_listen, g_n_send, g_n_recv, g_n_mr, g_n_req;

/* getProperties strings, copied into the upper half (NCCL keeps the
 * pointers; the lower half's copies die with it at restart). */
#define MAX_DEVS 64
static char *g_dev_name[MAX_DEVS], *g_dev_pci[MAX_DEVS];

static char g_name_buf[64] = "AWS Libfabric";

#define LOG(level, ...)                                                                   \
    do {                                                                                  \
        if (g_nccl_log != NULL)                                                           \
            g_nccl_log((level), 16 /* NCCL_NET */, __FILE__, __LINE__, __VA_ARGS__);      \
        else if ((level) <= NCCL_LOG_WARN) {                                              \
            fprintf(stderr, "[netsplit] " __VA_ARGS__);                                   \
            fputc('\n', stderr);                                                          \
        }                                                                                 \
    } while (0)

/* ---- TLS borrowing ------------------------------------------------------ */

/* Thread-exit destructor: the value packs the epoch above the FS (x86-64
 * user addresses fit in 47 bits). */
#define PACK(epoch, fs) ((void *)(((uintptr_t)(epoch) << 48) | (uintptr_t)(fs)))
#define PACK_EPOCH(p) ((unsigned)((uintptr_t)(p) >> 48))
#define PACK_FS(p) ((unsigned long)((uintptr_t)(p) & ((1UL << 48) - 1)))

static void map_set(unsigned long lhfs, unsigned long uhfs) {
    for (int i = 0; i < MAX_TLS_MAP; i++) {
        if (g_tls_map[i].lhfs == 0 || g_tls_map[i].lhfs == lhfs) {
            g_tls_map[i].uhfs = uhfs;
            __atomic_store_n(&g_tls_map[i].lhfs, lhfs, __ATOMIC_RELEASE);
            return;
        }
    }
}

static void map_clear(unsigned long lhfs) {
    for (int i = 0; i < MAX_TLS_MAP; i++) {
        if (g_tls_map[i].lhfs == lhfs) {
            __atomic_store_n(&g_tls_map[i].lhfs, 0, __ATOMIC_RELEASE);
            return;
        }
    }
}

unsigned long netproxy_upper_fs_for(unsigned long lhfs) {
    for (int i = 0; i < MAX_TLS_MAP; i++)
        if (__atomic_load_n(&g_tls_map[i].lhfs, __ATOMIC_ACQUIRE) == lhfs)
            return g_tls_map[i].uhfs;
    return 0;
}

static void tls_dtor(void *p) {
    if (p == NULL || lh_info == NULL || PACK_EPOCH(p) != (g_epoch & 0xffff))
        return; /* borrowed from a lower half that is gone */
    unsigned long uh = ns_getfs();
    pthread_mutex_lock(&g_lock);
    ns_setfs((unsigned long)lh_info->fsaddr);
    lh_info->tls_release(PACK_FS(p));
    ns_setfs(uh);
    pthread_mutex_unlock(&g_lock);
    map_clear(PACK_FS(p));
}

static void make_key(void) { pthread_key_create(&g_tls_key, tls_dtor); }

/* The device of the calling thread's current CUDA context, through the
 * upper half's driver, or -1. */
static int current_device(void) {
    static int (*ctx_get_device)(int *) = NULL;
    static int tried = 0;
    if (!tried) {
        void *h = dlopen("libcuda.so.1", RTLD_NOW | RTLD_NOLOAD);
        if (h != NULL)
            *(void **)&ctx_get_device = dlsym(h, "cuCtxGetDevice");
        tried = 1;
    }
    int dev = -1;
    if (ctx_get_device == NULL || ctx_get_device(&dev) != 0)
        return -1;
    return dev;
}

/* The calling thread's lower-half TLS for this epoch, or 0. */
static unsigned long thread_lhfs(void) {
    if (t_lhfs != 0 && t_epoch == g_epoch)
        return t_lhfs;
    pthread_once(&g_key_once, make_key);
    unsigned long uh = ns_getfs(), fs = 0;
    pthread_mutex_lock(&g_lock);
    ns_setfs((unsigned long)lh_info->fsaddr);
    int rc = lh_info->tls_acquire(&fs);
    ns_setfs(uh);
    pthread_mutex_unlock(&g_lock);
    if (rc != 0 || fs == 0) {
        LOG(NCCL_LOG_WARN, "netsplit: no lower-half TLS left (%d parked, %d in use)",
            lh_info->tls_parked, lh_info->tls_in_use);
        return 0;
    }
    t_lhfs = fs;
    t_epoch = g_epoch;
    map_set(fs, uh);
    pthread_setspecific(g_tls_key, PACK(g_epoch & 0xffff, fs));
    /* The back end's cudaGetDevice() answers for this thread. */
    int dev = current_device();
    if (dev >= 0 && lh_info->stub_set_thread_device != NULL) {
        ns_setfs(fs);
        lh_info->stub_set_thread_device(dev);
        ns_setfs(uh);
    }
    return fs;
}

/* Runs 'expr' (a back-end call) on the thread's lower-half TLS. */
#define DOWN(expr)                                                                        \
    ({                                                                                    \
        ncclResult_t r_ = ncclSystemError;                                                \
        unsigned long lh_ = thread_lhfs();                                                \
        if (lh_ != 0) {                                                                   \
            unsigned long uh_ = ns_getfs();                                               \
            ns_setfs(lh_);                                                                \
            r_ = (expr);                                                                  \
            ns_setfs(uh_);                                                                \
        }                                                                                 \
        r_;                                                                               \
    })

/* ---- back end ------------------------------------------------------------ */

static int upper_device_count(void) {
    void *h = dlopen("libcuda.so.1", RTLD_NOW | RTLD_NOLOAD);
    if (h == NULL)
        h = dlopen("libcuda.so.1", RTLD_NOW);
    int (*cu_init)(unsigned) = h ? (int (*)(unsigned))dlsym(h, "cuInit") : NULL;
    int (*cu_count)(int *) = h ? (int (*)(int *))dlsym(h, "cuDeviceGetCount") : NULL;
    int n = -1;
    if (cu_init != NULL && cu_count != NULL && cu_init(0) == 0 && cu_count(&n) == 0)
        return n;
    return -1;
}

static int gdr_passthrough(void);

/* GPUDirect RDMA: what the lower half's CUDA forwarders need, before
 * backend_load().  0 on success. */
static int gdr_setup(void) {
    void *h = dlopen("libcuda.so.1", RTLD_NOW | RTLD_NOLOAD);
    if (h == NULL) {
        LOG(NCCL_LOG_WARN, "netsplit: NETSPLIT_GDR=1 but no CUDA driver in the upper half");
        return -1;
    }
#define NETSPLIT_CU_LOOKUP(idx, sym)                                                      \
    lh_info->cuda_fn[NS_CU_##idx] = dlsym(h, #sym);                                       \
    if (lh_info->cuda_fn[NS_CU_##idx] == NULL)                                            \
        LOG(NCCL_LOG_WARN, "netsplit: the CUDA driver has no %s", #sym);
    NETSPLIT_CUDA_FUNCTIONS(NETSPLIT_CU_LOOKUP)
#undef NETSPLIT_CU_LOOKUP
    lh_info->fs_map = g_tls_map;
    lh_info->fs_map_len = MAX_TLS_MAP;
    /* The forwarders' stream and pinned bounce buffer, for this thread's
     * context (NCCL's device), now: see netsplit-api.h.  Each restart makes
     * a new pair; the old one (4 MiB) stays allocated. */
    int (*ctx_get_current)(void **) = (int (*)(void **))lh_info->cuda_fn[NS_CU_CTX_GET_CURRENT];
    int (*stream_create)(void **, unsigned) =
        (int (*)(void **, unsigned))lh_info->cuda_fn[NS_CU_STREAM_CREATE];
    int (*host_alloc)(void **, size_t, unsigned) =
        (int (*)(void **, size_t, unsigned))lh_info->cuda_fn[NS_CU_MEM_HOST_ALLOC];
    void *ctx = NULL, *stream = NULL, *bounce = NULL;
    const size_t bounce_size = 4UL << 20;
    if (ctx_get_current != NULL && stream_create != NULL && host_alloc != NULL &&
        ctx_get_current(&ctx) == 0 && ctx != NULL &&
        stream_create(&stream, 1 /* CU_STREAM_NON_BLOCKING */) == 0 &&
        host_alloc(&bounce, bounce_size, 0) == 0) {
        lh_info->fwd_ctx = ctx;
        lh_info->fwd_stream = stream;
        lh_info->fwd_bounce = bounce;
        lh_info->fwd_bounce_size = bounce_size;
    } else {
        LOG(NCCL_LOG_WARN, "netsplit: no CUDA context yet for the forwarders' copies; "
                           "they will make their own");
    }
    lh_info->gdr = 1;
    return 0;
}

/* Loads and initializes the back end of the current epoch, once.  g_init_lock
 * serializes this; g_lock only guards the lower half's main TLS, which
 * backend_load() runs on and thread_lhfs() takes for tls_acquire(). */
static pthread_mutex_t g_init_lock = PTHREAD_MUTEX_INITIALIZER;

static ncclResult_t ensure_backend(void) {
    if (g_inited_epoch == g_epoch)
        return ncclSuccess;
    if (lh_info == NULL && netsplit_lh_info_init(0) != 0) {
        LOG(NCCL_LOG_WARN, "netsplit: not started by netsplit-lh (no %s)", NETSPLIT_LH_INFO_ENV);
        return ncclSystemError;
    }
    pthread_mutex_lock(&g_init_lock);
    if (g_loaded_epoch != g_epoch) {
        unsigned long uh = ns_getfs();
        int ndev = upper_device_count();
        if (gdr_passthrough() && gdr_setup() != 0) {
            pthread_mutex_unlock(&g_init_lock);
            return ncclSystemError;
        }
        pthread_mutex_lock(&g_lock);
        /* backend_load() runs on the lower half's main TLS: map it to this
         * thread for the forwarders while it does. */
        map_set((unsigned long)lh_info->fsaddr, uh);
        ns_setfs((unsigned long)lh_info->fsaddr);
        int rc = lh_info->backend_load();
        if (rc == 0 && ndev >= 0 && lh_info->stub_set_ndev != NULL)
            lh_info->stub_set_ndev(ndev);
        ns_setfs(uh);
        map_clear((unsigned long)lh_info->fsaddr);
        pthread_mutex_unlock(&g_lock);
        if (rc != 0) {
            pthread_mutex_unlock(&g_init_lock);
            LOG(NCCL_LOG_WARN, "netsplit: cannot load the network back end: %s",
                lh_info->backend_error ? lh_info->backend_error : "?");
            return ncclSystemError;
        }
        snprintf(g_name_buf, sizeof(g_name_buf), "%s", lh_info->net->name);
        g_loaded_epoch = g_epoch;
    }
    ncclResult_t r = ncclSuccess;
    if (g_inited_epoch != g_epoch) {
        r = DOWN(lh_info->net->init(lh_info->logger));
        if (r == ncclSuccess) {
            g_inited_epoch = g_epoch;
            LOG(NCCL_LOG_INFO, "netsplit: back end '%s' initialized in the lower half (epoch %u)",
                g_name_buf, g_epoch);
        }
    }
    pthread_mutex_unlock(&g_init_lock);
    return r;
}

#define READY()                                                                           \
    do {                                                                                  \
        ncclResult_t e_ = ensure_backend();                                               \
        if (e_ != ncclSuccess)                                                            \
            return e_;                                                                    \
    } while (0)

static int gdr_passthrough(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("NETSPLIT_GDR");
        v = e != NULL && atoi(e) != 0;
    }
    return v;
}

/* ---- ncclNet v6 ------------------------------------------------------------ */

static ncclResult_t ns_init(ncclDebugLogger_t logFunction) {
    g_nccl_log = logFunction;
    return ensure_backend();
}

static ncclResult_t ns_devices(int *ndev) {
    READY();
    return DOWN(lh_info->net->devices(ndev));
}

static ncclResult_t ns_getProperties(int dev, ncclNetProperties_v6_t *props) {
    READY();
    ncclResult_t r = DOWN(lh_info->net->getProperties(dev, props));
    if (r != ncclSuccess)
        return r;
    if (dev >= 0 && dev < MAX_DEVS) {
        pthread_mutex_lock(&g_lock);
        if (props->name != NULL &&
            (g_dev_name[dev] == NULL || strcmp(g_dev_name[dev], props->name) != 0)) {
            free(g_dev_name[dev]);
            g_dev_name[dev] = strdup(props->name);
        }
        if (props->pciPath != NULL &&
            (g_dev_pci[dev] == NULL || strcmp(g_dev_pci[dev], props->pciPath) != 0)) {
            free(g_dev_pci[dev]);
            g_dev_pci[dev] = strdup(props->pciPath);
        }
        props->name = g_dev_name[dev];
        props->pciPath = g_dev_pci[dev];
        pthread_mutex_unlock(&g_lock);
    }
    if (!gdr_passthrough())
        props->ptrSupport &= NCCL_PTR_HOST;
    return ncclSuccess;
}

static ncclResult_t ns_listen(int dev, void *handle, void **listenComm) {
    READY();
    ncclResult_t r = DOWN(lh_info->net->listen(dev, handle, listenComm));
    if (r == ncclSuccess && *listenComm != NULL)
        __atomic_add_fetch(&g_n_listen, 1, __ATOMIC_RELAXED);
    return r;
}

static ncclResult_t ns_connect(int dev, void *handle, void **sendComm) {
    READY();
    ncclResult_t r = DOWN(lh_info->net->connect(dev, handle, sendComm));
    if (r == ncclSuccess && *sendComm != NULL)
        __atomic_add_fetch(&g_n_send, 1, __ATOMIC_RELAXED);
    return r;
}

static ncclResult_t ns_accept(void *listenComm, void **recvComm) {
    READY();
    ncclResult_t r = DOWN(lh_info->net->accept(listenComm, recvComm));
    if (r == ncclSuccess && *recvComm != NULL)
        __atomic_add_fetch(&g_n_recv, 1, __ATOMIC_RELAXED);
    return r;
}

static ncclResult_t ns_regMr(void *comm, void *data, int size, int type, void **mhandle) {
    READY();
    if (type != NCCL_PTR_HOST && !gdr_passthrough()) {
        LOG(NCCL_LOG_WARN, "netsplit: regMr of non-host memory (type %d), which getProperties "
                           "does not offer", type);
        return ncclInternalError;
    }
    ncclResult_t r = DOWN(lh_info->net->regMr(comm, data, size, type, mhandle));
    if (r == ncclSuccess)
        __atomic_add_fetch(&g_n_mr, 1, __ATOMIC_RELAXED);
    return r;
}

static ncclResult_t ns_regMrDmaBuf(void *comm, void *data, size_t size, int type,
                                   uint64_t offset, int fd, void **mhandle) {
    READY();
    if (!gdr_passthrough() || lh_info->net->regMrDmaBuf == NULL)
        return ncclInternalError; /* not offered by getProperties */
    ncclResult_t r = DOWN(lh_info->net->regMrDmaBuf(comm, data, size, type, offset, fd, mhandle));
    if (r == ncclSuccess)
        __atomic_add_fetch(&g_n_mr, 1, __ATOMIC_RELAXED);
    return r;
}

static ncclResult_t ns_deregMr(void *comm, void *mhandle) {
    READY();
    ncclResult_t r = DOWN(lh_info->net->deregMr(comm, mhandle));
    if (r == ncclSuccess)
        __atomic_sub_fetch(&g_n_mr, 1, __ATOMIC_RELAXED);
    return r;
}

static ncclResult_t ns_isend(void *sendComm, void *data, int size, int tag, void *mhandle,
                             void **request) {
    ncclResult_t r = DOWN(lh_info->net->isend(sendComm, data, size, tag, mhandle, request));
    if (r == ncclSuccess && *request != NULL)
        __atomic_add_fetch(&g_n_req, 1, __ATOMIC_RELAXED);
    return r;
}

static ncclResult_t ns_irecv(void *recvComm, int n, void **data, int *sizes, int *tags,
                             void **mhandles, void **request) {
    ncclResult_t r = DOWN(lh_info->net->irecv(recvComm, n, data, sizes, tags, mhandles, request));
    if (r == ncclSuccess && *request != NULL)
        __atomic_add_fetch(&g_n_req, 1, __ATOMIC_RELAXED);
    return r;
}

static ncclResult_t ns_iflush(void *recvComm, int n, void **data, int *sizes, void **mhandles,
                              void **request) {
    ncclResult_t r = DOWN(lh_info->net->iflush(recvComm, n, data, sizes, mhandles, request));
    if (r == ncclSuccess && *request != NULL)
        __atomic_add_fetch(&g_n_req, 1, __ATOMIC_RELAXED);
    return r;
}

static ncclResult_t ns_test(void *request, int *done, int *sizes) {
    ncclResult_t r = DOWN(lh_info->net->test(request, done, sizes));
    if (r == ncclSuccess && *done)
        __atomic_sub_fetch(&g_n_req, 1, __ATOMIC_RELAXED);
    return r;
}

static ncclResult_t ns_closeSend(void *sendComm) {
    ncclResult_t r = DOWN(lh_info->net->closeSend(sendComm));
    if (r == ncclSuccess)
        __atomic_sub_fetch(&g_n_send, 1, __ATOMIC_RELAXED);
    return r;
}

static ncclResult_t ns_closeRecv(void *recvComm) {
    ncclResult_t r = DOWN(lh_info->net->closeRecv(recvComm));
    if (r == ncclSuccess)
        __atomic_sub_fetch(&g_n_recv, 1, __ATOMIC_RELAXED);
    return r;
}

static ncclResult_t ns_closeListen(void *listenComm) {
    ncclResult_t r = DOWN(lh_info->net->closeListen(listenComm));
    if (r == ncclSuccess)
        __atomic_sub_fetch(&g_n_listen, 1, __ATOMIC_RELAXED);
    return r;
}

__attribute__((visibility("default"))) ncclNet_v6_t ncclNetPlugin_v6 = {
    g_name_buf,   ns_init,  ns_devices, ns_getProperties, ns_listen,   ns_connect,
    ns_accept,    ns_regMr, ns_regMrDmaBuf, ns_deregMr,   ns_isend,    ns_irecv,
    ns_iflush,    ns_test,  ns_closeSend, ns_closeRecv,   ns_closeListen,
};

/* ---- hooks for the DMTCP glue ----------------------------------------------- */

long netproxy_live_handles(char *buf, size_t len) {
    long l = __atomic_load_n(&g_n_listen, __ATOMIC_RELAXED);
    long s = __atomic_load_n(&g_n_send, __ATOMIC_RELAXED);
    long rv = __atomic_load_n(&g_n_recv, __ATOMIC_RELAXED);
    long m = __atomic_load_n(&g_n_mr, __ATOMIC_RELAXED);
    long q = __atomic_load_n(&g_n_req, __ATOMIC_RELAXED);
    if (buf != NULL)
        snprintf(buf, len, "%ld listen, %ld send, %ld recv comms, %ld registrations, %ld requests",
                 l, s, rv, m, q);
    return l + s + rv + m + q;
}

void netproxy_restart(void) {
    /* Everything of the old lower half is gone: its back end, its TLS. */
    g_epoch++;
    g_loaded_epoch = 0;
    g_inited_epoch = 0;
    g_n_listen = g_n_send = g_n_recv = g_n_mr = g_n_req = 0;
    for (int i = 0; i < MAX_TLS_MAP; i++)
        g_tls_map[i].lhfs = 0;
}
