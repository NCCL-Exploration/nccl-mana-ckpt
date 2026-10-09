/*
 * fi_cudart_shim: the CUDA runtime as libfabric sees it, without GDRCopy.
 *
 * Without GDRCopy, libfabric copies small messages into GPU memory with
 * cudaMemcpy followed by cudaDeviceSynchronize (hmem_cuda.c), which waits for
 * NCCL's running kernel while that kernel waits for the very proxy thread
 * doing the copy: a deadlock.  For callers inside libfabric this library
 * copies on a private non-blocking stream and waits for that stream only.
 * Everyone else gets the real functions.
 *
 * Preloaded (LD_PRELOAD): libfabric's dlopen("libcudart.so") then finds this
 * library by its SONAME.  Under DMTCP, launch with --disable-dl-plugin: the
 * dl plugin's dlopen() wrapper opens "<dir>/libcudart.so" for each directory
 * of LD_LIBRARY_PATH, which finds the real runtime instead.
 */
#define _GNU_SOURCE
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* FI_SHIM_VERBOSE=1: report loading and the first calls from libfabric. */
static int verbose(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("FI_SHIM_VERBOSE");
        v = e != NULL && e[0] == '1';
    }
    return v;
}

static void note(const char *what, size_t n) {
    static int count;
    if (verbose() && count++ < 20)
        fprintf(stderr, "[fi-shim %d] %s %zu\n", (int)getpid(), what, n);
}

__attribute__((constructor)) static void loaded(void) {
    if (verbose())
        fprintf(stderr, "[fi-shim %d] loaded\n", (int)getpid());
}

static void *real(const char *name) {
    static void *h;
    if (h == NULL)
        h = dlopen("libcudart.so.13", RTLD_NOW | RTLD_LOCAL);
    return h != NULL ? dlsym(h, name) : NULL;
}

static int from_libfabric(void *ret) {
    Dl_info di;
    return dladdr(ret, &di) != 0 && di.dli_fname != NULL && strstr(di.dli_fname, "libfabric") != NULL;
}

#define FWD(ret_t, name, params, args)                                             \
    ret_t name params {                                                            \
        static ret_t(*f) params;                                                   \
        if (f == NULL)                                                             \
            *(void **)&f = real(#name);                                            \
        return f args;                                                             \
    }

FWD(cudaError_t, cudaFree, (void *p), (p))
FWD(cudaError_t, cudaGetDevice, (int *d), (d))
FWD(cudaError_t, cudaGetDeviceCount, (int *n), (n))
FWD(const char *, cudaGetErrorName, (cudaError_t e), (e))
FWD(const char *, cudaGetErrorString, (cudaError_t e), (e))
FWD(cudaError_t, cudaHostRegister, (void *p, size_t n, unsigned int fl), (p, n, fl))
FWD(cudaError_t, cudaHostUnregister, (void *p), (p))
FWD(cudaError_t, cudaIpcCloseMemHandle, (void *p), (p))
FWD(cudaError_t, cudaIpcGetMemHandle, (cudaIpcMemHandle_t *h, void *p), (h, p))
FWD(cudaError_t, cudaIpcOpenMemHandle, (void **p, cudaIpcMemHandle_t h, unsigned int fl),
    (p, h, fl))
FWD(cudaError_t, cudaMalloc, (void **p, size_t n), (p, n))
FWD(cudaError_t, cudaSetDevice, (int d), (d))

#define MAX_DEV 16
static __thread cudaStream_t t_stream[MAX_DEV];

static cudaStream_t private_stream(void) {
    static cudaError_t (*get)(int *), (*create)(cudaStream_t *, unsigned int);
    int d = 0;
    if (get == NULL) {
        *(void **)&get = real("cudaGetDevice");
        *(void **)&create = real("cudaStreamCreateWithFlags");
    }
    get(&d);
    if (d < 0 || d >= MAX_DEV)
        d = 0;
    if (t_stream[d] == NULL)
        create(&t_stream[d], cudaStreamNonBlocking);
    return t_stream[d];
}

cudaError_t cudaMemcpy(void *dst, const void *src, size_t n, enum cudaMemcpyKind kind) {
    static cudaError_t (*f)(void *, const void *, size_t, enum cudaMemcpyKind);
    static cudaError_t (*async)(void *, const void *, size_t, enum cudaMemcpyKind, cudaStream_t);
    static cudaError_t (*sync)(cudaStream_t);
    if (f == NULL) {
        *(void **)&f = real("cudaMemcpy");
        *(void **)&async = real("cudaMemcpyAsync");
        *(void **)&sync = real("cudaStreamSynchronize");
    }
    if (!from_libfabric(__builtin_return_address(0)))
        return f(dst, src, n, kind);
    note("cudaMemcpy", n);
    cudaStream_t s = private_stream();
    cudaError_t e = async(dst, src, n, kind, s);
    return e != cudaSuccess ? e : sync(s);
}

cudaError_t cudaDeviceSynchronize(void) {
    static cudaError_t (*f)(void), (*sync)(cudaStream_t);
    if (f == NULL) {
        *(void **)&f = real("cudaDeviceSynchronize");
        *(void **)&sync = real("cudaStreamSynchronize");
    }
    /* libfabric syncs after its own copies, which already waited. */
    if (from_libfabric(__builtin_return_address(0))) {
        note("cudaDeviceSynchronize", 0);
        return sync(private_stream());
    }
    return f();
}
