/*
 * The lower half's libcudart.so.13: answers the four CUDA runtime calls of
 * aws-ofi-nccl 1.6.0 without a CUDA driver, so that no live CUDA call crosses
 * the split (netsplit design decision D3).
 *
 *   cudaGetDeviceCount   the count the upper half reported
 *   cudaGetDevice        the calling NCCL thread's device
 *   cudaPointerGetAttributes  host memory, unless the proxy registered the
 *                        range as device memory (phase 2)
 *   cudaDeviceFlushGPUDirectRDMAWrites  no-op while GPUDirect RDMA is off
 *
 * The proxy sets the values through netsplit_stub_set_*() (lower half, via
 * the call table).  Until then, NETSPLIT_NDEV and NETSPLIT_DEV (or
 * SLURM_LOCALID) give them, which is what the standalone probe uses.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef int cudaError_t; /* cudaSuccess = 0, cudaErrorInvalidValue = 1 */

/* driver_types.h of CUDA 13.2 */
struct cudaPointerAttributes {
    int type; /* cudaMemoryTypeUnregistered = 0, Host = 1, Device = 2 */
    int device;
    void *devicePointer;
    void *hostPointer;
    long reserved[8];
};

static int g_ndev = -1;
static __thread int t_dev = -1;

/* Device memory ranges registered by the proxy (phase 2). */
#define MAX_RANGES 256
static struct {
    uintptr_t lo, hi;
    int dev;
} g_ranges[MAX_RANGES];
static int g_nranges;

static int env_int(const char *name, int dflt) {
    const char *v = getenv(name);
    return (v != NULL && *v != '\0') ? atoi(v) : dflt;
}

void netsplit_stub_set_ndev(int n) { g_ndev = n; }
void netsplit_stub_set_thread_device(int dev) { t_dev = dev; }

int netsplit_stub_add_device_range(void *ptr, size_t size, int dev) {
    if (g_nranges >= MAX_RANGES)
        return -1;
    g_ranges[g_nranges].lo = (uintptr_t)ptr;
    g_ranges[g_nranges].hi = (uintptr_t)ptr + size;
    g_ranges[g_nranges].dev = dev;
    g_nranges++;
    return 0;
}

void netsplit_stub_remove_device_range(void *ptr) {
    for (int i = 0; i < g_nranges; i++) {
        if (g_ranges[i].lo == (uintptr_t)ptr) {
            g_ranges[i] = g_ranges[--g_nranges];
            return;
        }
    }
}

cudaError_t cudaGetDeviceCount(int *count) {
    if (count == NULL)
        return 1;
    *count = g_ndev >= 0 ? g_ndev : env_int("NETSPLIT_NDEV", 4);
    return 0;
}

cudaError_t cudaGetDevice(int *device) {
    if (device == NULL)
        return 1;
    if (t_dev >= 0)
        *device = t_dev;
    else
        *device = env_int("NETSPLIT_DEV", env_int("SLURM_LOCALID", 0));
    return 0;
}

cudaError_t cudaPointerGetAttributes(struct cudaPointerAttributes *attr, const void *ptr) {
    if (attr == NULL)
        return 1;
    memset(attr, 0, sizeof(*attr));
    for (int i = 0; i < g_nranges; i++) {
        if ((uintptr_t)ptr >= g_ranges[i].lo && (uintptr_t)ptr < g_ranges[i].hi) {
            attr->type = 2;
            attr->device = g_ranges[i].dev;
            attr->devicePointer = (void *)ptr;
            return 0;
        }
    }
    attr->type = 0;    /* unregistered host memory */
    attr->device = -2; /* what the runtime reports for host memory */
    attr->hostPointer = (void *)ptr;
    return 0;
}

cudaError_t cudaDeviceFlushGPUDirectRDMAWrites(int target, int scope) {
    (void)target;
    (void)scope;
    return 0;
}
