/*
 * NCCL's network plugin API, version 6 (ncclNet_v6_t), as NCCL 2.29 loads it
 * from aws-ofi-nccl 1.6.0 ("Loaded net plugin AWS Libfabric (v6)").  The
 * layout follows NCCL's ext-net/example/nccl/net_v6.h; it must match that
 * header exactly, since both NCCL and the back end use it.
 *
 * Self-contained on purpose: the lower half and the probe include it without
 * CUDA or NCCL headers.
 */
#ifndef NETSPLIT_NET_V6_H
#define NETSPLIT_NET_V6_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef NCCL_H_
/* Same values as nccl.h's ncclResult_t. */
typedef enum {
    ncclSuccess = 0,
    ncclUnhandledCudaError = 1,
    ncclSystemError = 2,
    ncclInternalError = 3,
    ncclInvalidArgument = 4,
    ncclInvalidUsage = 5,
    ncclRemoteError = 6,
    ncclInProgress = 7,
    ncclNumResults = 8
} ncclResult_t;
#endif

#define NCCL_NET_HANDLE_MAXSIZE 128

#define NCCL_PTR_HOST 0x1
#define NCCL_PTR_CUDA 0x2
#define NCCL_PTR_DMABUF 0x4

typedef enum {
    NCCL_LOG_NONE = 0,
    NCCL_LOG_VERSION = 1,
    NCCL_LOG_WARN = 2,
    NCCL_LOG_INFO = 3,
    NCCL_LOG_ABORT = 4,
    NCCL_LOG_TRACE = 5
} ncclDebugLogLevel;

typedef void (*ncclDebugLogger_t)(ncclDebugLogLevel level, unsigned long flags, const char *file,
                                  int line, const char *fmt, ...);

typedef struct {
    char *name;     /* Used mostly for logging. */
    char *pciPath;  /* Path to the PCI device in /sys. */
    uint64_t guid;  /* Unique identifier for the NIC chip. */
    int ptrSupport; /* NCCL_PTR_HOST | NCCL_PTR_CUDA | NCCL_PTR_DMABUF */
    int speed;      /* Port speed in Mbps. */
    int port;       /* Port number. */
    float latency;  /* Network latency */
    int maxComms;   /* Maximum number of comms we can create */
    int maxRecvs;   /* Maximum number of grouped receives. */
} ncclNetProperties_v6_t;

typedef struct {
    const char *name;
    ncclResult_t (*init)(ncclDebugLogger_t logFunction);
    ncclResult_t (*devices)(int *ndev);
    ncclResult_t (*getProperties)(int dev, ncclNetProperties_v6_t *props);
    /* listen/connect/accept never block: connect and accept return a NULL
     * comm until the connection is up, and are called again. */
    ncclResult_t (*listen)(int dev, void *handle, void **listenComm);
    ncclResult_t (*connect)(int dev, void *handle, void **sendComm);
    ncclResult_t (*accept)(void *listenComm, void **recvComm);
    ncclResult_t (*regMr)(void *comm, void *data, int size, int type, void **mhandle);
    ncclResult_t (*regMrDmaBuf)(void *comm, void *data, size_t size, int type, uint64_t offset,
                                int fd, void **mhandle);
    ncclResult_t (*deregMr)(void *comm, void *mhandle);
    /* isend/irecv may return a NULL request: try again later. */
    ncclResult_t (*isend)(void *sendComm, void *data, int size, int tag, void *mhandle,
                          void **request);
    ncclResult_t (*irecv)(void *recvComm, int n, void **data, int *sizes, int *tags,
                          void **mhandles, void **request);
    ncclResult_t (*iflush)(void *recvComm, int n, void **data, int *sizes, void **mhandles,
                           void **request);
    ncclResult_t (*test)(void *request, int *done, int *sizes);
    ncclResult_t (*closeSend)(void *sendComm);
    ncclResult_t (*closeRecv)(void *recvComm);
    ncclResult_t (*closeListen)(void *listenComm);
} ncclNet_v6_t;

#ifdef __cplusplus
}
#endif
#endif /* NETSPLIT_NET_V6_H */
