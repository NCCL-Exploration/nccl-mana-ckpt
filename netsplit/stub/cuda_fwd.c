/*
 * libfabric's CUDA in the lower half, for GPUDirect RDMA (NETSPLIT_GDR=1).
 *
 * libfabric 2.3.1 dlopen()s libcuda.so.1, libcudart.so and libnvidia-ml.so.1
 * for its CUDA memory support.  The real ones would start a second CUDA
 * driver in the process, one that knows nothing of the upper half's GPU
 * memory and that the CUDA checkpoint does not cover.  This file is built
 * three times, with those SONAMEs (and netsplit file names, so that the upper
 * half, which has netsplit/lib on its path, never loads them):
 *
 *   -DFWD_CUDA    libcuda.so.1       libfabric's 10 driver calls, forwarded
 *   -DFWD_CUDART  libcudart.so       its 14 runtime calls, made with the
 *                                    driver API (the upper half has no
 *                                    dynamic runtime: NCCL's is static)
 *   -DFWD_NVML    libnvidia-ml.so.1  init and device count
 *
 * backend_load() loads them before the back end, so libfabric's dlopen()
 * finds them by SONAME.  Every call runs the upper half's driver on the
 * calling thread's upper-half TLS (cuda_fwd.h).  Two runtime calls differ:
 *
 * - cudaMemcpy copies on a private non-blocking stream and waits for that
 *   stream only, and cudaDeviceSynchronize waits for that stream only.
 *   libfabric (without GDRCopy) copies small early messages into GPU memory
 *   with cudaMemcpy + cudaDeviceSynchronize; a device-wide sync waits for
 *   NCCL's kernel, which waits for the proxy thread doing the copy.
 * - cudaMemcpy moves host data through a pinned bounce buffer of its own.
 *   libfabric's buffers are pageable, and the driver serializes pageable
 *   copies through its staging buffers: an application thread in a pageable
 *   cudaMemcpy that waits for NCCL's kernel holds them, and the proxy
 *   thread's copy, which that kernel waits for, blocks behind it (seen in
 *   nccl_bench bw at a checkpoint).
 * - cudaHostRegister refuses.  libfabric would pin its own bounce buffers,
 *   which are lower-half memory: the CUDA restore after a restart would
 *   re-pin memory that the new lower half does not have.
 */
#define _GNU_SOURCE
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdlib.h>
#include <string.h>

#include "cuda_fwd.h"

#define EXPORT __attribute__((visibility("default")))

#ifdef FWD_CUDA
/* ---- libcuda.so.1: libfabric's driver calls, one to one ------------------- */

EXPORT CUresult cuDeviceGet(CUdevice *dev, int ordinal) {
    FORWARD(CUresult, DEVICE_GET, cuDeviceGet, dev, ordinal);
}

EXPORT CUresult cuDeviceGetAttribute(int *value, CUdevice_attribute attr, CUdevice dev) {
    FORWARD(CUresult, DEVICE_GET_ATTRIBUTE, cuDeviceGetAttribute, value, attr, dev);
}

EXPORT CUresult cuDeviceCanAccessPeer(int *can, CUdevice dev, CUdevice peer) {
    FORWARD(CUresult, DEVICE_CAN_ACCESS_PEER, cuDeviceCanAccessPeer, can, dev, peer);
}

EXPORT CUresult cuGetErrorName(CUresult error, const char **str) {
    FORWARD(CUresult, GET_ERROR_NAME, cuGetErrorName, error, str);
}

EXPORT CUresult cuGetErrorString(CUresult error, const char **str) {
    FORWARD(CUresult, GET_ERROR_STRING, cuGetErrorString, error, str);
}

EXPORT CUresult cuPointerGetAttribute(void *data, CUpointer_attribute attr, CUdeviceptr ptr) {
    FORWARD(CUresult, POINTER_GET_ATTRIBUTE, cuPointerGetAttribute, data, attr, ptr);
}

EXPORT CUresult cuPointerGetAttributes(unsigned int n, CUpointer_attribute *attrs, void **data,
                                       CUdeviceptr ptr) {
    FORWARD(CUresult, POINTER_GET_ATTRIBUTES, cuPointerGetAttributes, n, attrs, data, ptr);
}

EXPORT CUresult cuPointerSetAttribute(const void *value, CUpointer_attribute attr,
                                      CUdeviceptr ptr) {
    FORWARD(CUresult, POINTER_SET_ATTRIBUTE, cuPointerSetAttribute, value, attr, ptr);
}

/* cuda.h names it cuMemGetAddressRange; libfabric asks for the _v2 symbol. */
EXPORT CUresult cuMemGetAddressRange_v2(CUdeviceptr *base, size_t *size, CUdeviceptr ptr) {
    FORWARD(CUresult, MEM_GET_ADDRESS_RANGE, cuMemGetAddressRange_v2, base, size, ptr);
}

EXPORT CUresult cuMemGetHandleForAddressRange(void *handle, CUdeviceptr ptr, size_t size,
                                              CUmemRangeHandleType type,
                                              unsigned long long flags) {
    FORWARD(CUresult, MEM_GET_HANDLE_FOR_ADDRESS_RANGE, cuMemGetHandleForAddressRange, handle,
            ptr, size, type, flags);
}
#endif /* FWD_CUDA */

#ifdef FWD_CUDART
/* ---- libcudart.so: libfabric's runtime calls, on the driver API ------------ */

/* The driver's error codes match the runtime's for the errors that matter
 * here (success, invalid value, out of memory, not initialized, ...). */

#define MAX_DEVS 64
#define MAX_STREAMS 64

/* Private streams and pinned bounce buffers by context, and retained
 * primary contexts by device.  Upper-half CUDA objects in lower-half memory:
 * a restart's new lower half starts empty (the old ones stay behind in the
 * upper half); after a resume they are still valid. */
#define BOUNCE_SIZE (4UL << 20)
typedef struct {
    CUcontext ctx;
    CUstream stream;
    void *bounce;
    size_t bounce_size;
    volatile int busy; /* one copy at a time per bounce buffer */
} fwd_slot;
static fwd_slot g_streams[MAX_STREAMS];
static CUcontext g_primary[MAX_DEVS];
static volatile int g_lock;

static void lock(void) {
    while (__atomic_test_and_set(&g_lock, __ATOMIC_ACQUIRE))
        __builtin_ia32_pause();
}

static void unlock(void) { __atomic_clear(&g_lock, __ATOMIC_RELEASE); }

/* The current context's slot.  The proxy's stream and bounce buffer serve
 * its context; for another, 'create' makes them, which can block in the
 * driver if NCCL's traffic is running (netsplit-api.h).  Runs in the upper
 * half (between fwd_enter and fwd_leave). */
static fwd_slot *private_slot(int create) {
    CUcontext ctx = NULL;
    fwd_slot *slot = NULL;
    if (REAL(CTX_GET_CURRENT, cuCtxGetCurrent)(&ctx) != CUDA_SUCCESS || ctx == NULL)
        return NULL;
    lock();
    int free_slot = -1;
    for (int i = 0; i < MAX_STREAMS && slot == NULL; i++) {
        if (g_streams[i].ctx == ctx)
            slot = &g_streams[i];
        else if (g_streams[i].ctx == NULL && free_slot < 0)
            free_slot = i;
    }
    if (slot == NULL && free_slot >= 0 && fwd_info->fwd_ctx == (void *)ctx &&
        fwd_info->fwd_stream != NULL && fwd_info->fwd_bounce != NULL) {
        slot = &g_streams[free_slot];
        slot->ctx = ctx;
        slot->stream = (CUstream)fwd_info->fwd_stream;
        slot->bounce = fwd_info->fwd_bounce;
        slot->bounce_size = fwd_info->fwd_bounce_size;
    } else if (slot == NULL && create && free_slot >= 0) {
        CUstream s = NULL;
        void *bounce = NULL;
        if (REAL(STREAM_CREATE, cuStreamCreate)(&s, CU_STREAM_NON_BLOCKING) == CUDA_SUCCESS &&
            REAL(MEM_HOST_ALLOC, cuMemHostAlloc)(&bounce, BOUNCE_SIZE, 0) == CUDA_SUCCESS) {
            slot = &g_streams[free_slot];
            slot->ctx = ctx;
            slot->stream = s;
            slot->bounce = bounce;
            slot->bounce_size = BOUNCE_SIZE;
        }
    }
    unlock();
    return slot;
}

static int is_device(const void *p) {
    unsigned int type = 0;
    return REAL(POINTER_GET_ATTRIBUTE, cuPointerGetAttribute)(
               &type, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, (CUdeviceptr)p) == CUDA_SUCCESS &&
           (type == CU_MEMORYTYPE_DEVICE || type == CU_MEMORYTYPE_UNIFIED);
}

/* Host <-> device in pieces through the slot's pinned buffer; the host
 * side is a plain memcpy, on the lower half's FS. */
static CUresult bounce_copy(fwd_slot *slot, void *dst, const void *src, size_t count, int to_dev) {
    CUresult r = CUDA_SUCCESS;
    size_t piece = slot->bounce_size;
    for (size_t off = 0; off < count && r == CUDA_SUCCESS; off += piece) {
        size_t n = count - off < piece ? count - off : piece;
        if (to_dev)
            memcpy(slot->bounce, (const char *)src + off, n);
        unsigned long lh = fwd_enter("cudaMemcpy");
        if (lh == 0)
            return CUDA_ERROR_NOT_INITIALIZED;
        if (to_dev)
            r = REAL(MEMCPY_HTOD_ASYNC, cuMemcpyHtoDAsync_v2)((CUdeviceptr)dst + off, slot->bounce,
                                                              n, slot->stream);
        else
            r = REAL(MEMCPY_DTOH_ASYNC, cuMemcpyDtoHAsync_v2)(
                slot->bounce, (CUdeviceptr)src + off, n, slot->stream);
        if (r == CUDA_SUCCESS)
            r = REAL(STREAM_SYNCHRONIZE, cuStreamSynchronize)(slot->stream);
        fwd_leave(lh);
        if (!to_dev && r == CUDA_SUCCESS)
            memcpy((char *)dst + off, slot->bounce, n);
    }
    return r;
}

EXPORT cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind) {
    if (kind == cudaMemcpyHostToHost) {
        memcpy(dst, src, count);
        return cudaSuccess;
    }
    unsigned long lh = fwd_enter("cudaMemcpy");
    if (lh == 0)
        return cudaErrorInitializationError;
    fwd_slot *slot = private_slot(1);
    int src_dev = kind == cudaMemcpyDeviceToHost || kind == cudaMemcpyDeviceToDevice ||
                  (kind == cudaMemcpyDefault && is_device(src));
    int dst_dev = kind == cudaMemcpyHostToDevice || kind == cudaMemcpyDeviceToDevice ||
                  (kind == cudaMemcpyDefault && is_device(dst));
    CUresult r = CUDA_ERROR_INVALID_CONTEXT;
    if (slot != NULL && src_dev && dst_dev) {
        r = REAL(MEMCPY_ASYNC, cuMemcpyAsync)((CUdeviceptr)dst, (CUdeviceptr)src, count,
                                              slot->stream);
        if (r == CUDA_SUCCESS)
            r = REAL(STREAM_SYNCHRONIZE, cuStreamSynchronize)(slot->stream);
    }
    fwd_leave(lh);
    if (slot == NULL || (src_dev && dst_dev))
        return (cudaError_t)r;
    if (!src_dev && !dst_dev) {
        memcpy(dst, src, count);
        return cudaSuccess;
    }
    while (__atomic_test_and_set(&slot->busy, __ATOMIC_ACQUIRE))
        __builtin_ia32_pause();
    r = bounce_copy(slot, dst, src, count, dst_dev);
    __atomic_clear(&slot->busy, __ATOMIC_RELEASE);
    return (cudaError_t)r;
}

/* libfabric syncs after its own copies, which have already waited. */
EXPORT cudaError_t cudaDeviceSynchronize(void) {
    unsigned long lh = fwd_enter("cudaDeviceSynchronize");
    if (lh == 0)
        return cudaErrorInitializationError;
    CUresult r = CUDA_SUCCESS;
    fwd_slot *slot = private_slot(0);
    if (slot != NULL)
        r = REAL(STREAM_SYNCHRONIZE, cuStreamSynchronize)(slot->stream);
    fwd_leave(lh);
    return (cudaError_t)r;
}

EXPORT cudaError_t cudaGetDeviceCount(int *count) {
    FORWARD(cudaError_t, DEVICE_GET_COUNT, cuDeviceGetCount, count);
}

/* The current context's device; 0 without one, as the runtime reports. */
EXPORT cudaError_t cudaGetDevice(int *device) {
    unsigned long lh = fwd_enter("cudaGetDevice");
    if (lh == 0)
        return cudaErrorInitializationError;
    CUdevice dev = 0;
    if (REAL(CTX_GET_DEVICE, cuCtxGetDevice)(&dev) != CUDA_SUCCESS)
        dev = 0;
    fwd_leave(lh);
    *device = (int)dev;
    return cudaSuccess;
}

/* Makes the device's primary context current on the calling NCCL thread,
 * which normally has it current already. */
EXPORT cudaError_t cudaSetDevice(int device) {
    if (device < 0 || device >= MAX_DEVS)
        return cudaErrorInvalidDevice;
    unsigned long lh = fwd_enter("cudaSetDevice");
    if (lh == 0)
        return cudaErrorInitializationError;
    CUdevice cur, dev;
    CUresult r = CUDA_SUCCESS;
    if (REAL(CTX_GET_DEVICE, cuCtxGetDevice)(&cur) != CUDA_SUCCESS || cur != device) {
        lock();
        CUcontext ctx = g_primary[device];
        unlock();
        r = REAL(DEVICE_GET, cuDeviceGet)(&dev, device);
        if (r == CUDA_SUCCESS && ctx == NULL &&
            (r = REAL(DEVICE_PRIMARY_CTX_RETAIN, cuDevicePrimaryCtxRetain)(&ctx, dev)) ==
                CUDA_SUCCESS) {
            lock();
            g_primary[device] = ctx;
            unlock();
        }
        if (r == CUDA_SUCCESS)
            r = REAL(CTX_SET_CURRENT, cuCtxSetCurrent)(ctx);
    }
    fwd_leave(lh);
    return (cudaError_t)r;
}

EXPORT cudaError_t cudaMalloc(void **ptr, size_t size) {
    FORWARD(cudaError_t, MEM_ALLOC, cuMemAlloc_v2, (CUdeviceptr *)ptr, size);
}

EXPORT cudaError_t cudaFree(void *ptr) {
    if (ptr == NULL)
        return cudaSuccess;
    FORWARD(cudaError_t, MEM_FREE, cuMemFree_v2, (CUdeviceptr)ptr);
}

/* Refused: see the top of the file. */
EXPORT cudaError_t cudaHostRegister(void *ptr, size_t size, unsigned int flags) {
    (void)ptr;
    (void)size;
    (void)flags;
    return cudaErrorNotSupported;
}

EXPORT cudaError_t cudaHostUnregister(void *ptr) {
    (void)ptr;
    return cudaErrorHostMemoryNotRegistered;
}

EXPORT cudaError_t cudaIpcGetMemHandle(cudaIpcMemHandle_t *handle, void *ptr) {
    FORWARD(cudaError_t, IPC_GET_MEM_HANDLE, cuIpcGetMemHandle, (CUipcMemHandle *)handle,
            (CUdeviceptr)ptr);
}

EXPORT cudaError_t cudaIpcOpenMemHandle(void **ptr, cudaIpcMemHandle_t handle,
                                        unsigned int flags) {
    CUipcMemHandle h;
    memcpy(&h, &handle, sizeof(h));
    FORWARD(cudaError_t, IPC_OPEN_MEM_HANDLE, cuIpcOpenMemHandle_v2, (CUdeviceptr *)ptr, h,
            flags);
}

EXPORT cudaError_t cudaIpcCloseMemHandle(void *ptr) {
    FORWARD(cudaError_t, IPC_CLOSE_MEM_HANDLE, cuIpcCloseMemHandle, (CUdeviceptr)ptr);
}

static const char *error_text(cudaError_t error, int name) {
    const char *s = NULL;
    unsigned long lh = fwd_enter(name ? "cudaGetErrorName" : "cudaGetErrorString");
    if (lh != 0) {
        if (name)
            REAL(GET_ERROR_NAME, cuGetErrorName)((CUresult)error, &s);
        else
            REAL(GET_ERROR_STRING, cuGetErrorString)((CUresult)error, &s);
        fwd_leave(lh);
    }
    return s != NULL ? s : "unknown CUDA error";
}

EXPORT const char *cudaGetErrorName(cudaError_t error) { return error_text(error, 1); }

EXPORT const char *cudaGetErrorString(cudaError_t error) { return error_text(error, 0); }
#endif /* FWD_CUDART */

#ifdef FWD_NVML
/* ---- libnvidia-ml.so.1: libfabric only checks that NVIDIA devices exist ---- */

typedef int nvmlReturn_t; /* NVML_SUCCESS = 0, NVML_ERROR_UNKNOWN = 999 */

EXPORT nvmlReturn_t nvmlInit_v2(void) { return 0; }

EXPORT nvmlReturn_t nvmlShutdown(void) { return 0; }

EXPORT nvmlReturn_t nvmlDeviceGetCount_v2(unsigned int *count) {
    unsigned long lh = fwd_enter("nvmlDeviceGetCount_v2");
    if (lh == 0)
        return 999;
    int n = 0;
    CUresult r = REAL(DEVICE_GET_COUNT, cuDeviceGetCount)(&n);
    fwd_leave(lh);
    if (r != CUDA_SUCCESS)
        return 999;
    *count = (unsigned int)n;
    return 0;
}
#endif /* FWD_NVML */
