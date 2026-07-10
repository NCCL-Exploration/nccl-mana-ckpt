/*
 * drain_recv.c
 *
 * 2 MPI processes on 2 GPUs.
 * Rank 0: ncclSend → sleep(30)
 * Rank 1: sleep(10) → ncclRecv
 *
 * Build & Run:
 *   Run `make drain_recv_check` in parent directory
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <assert.h>
#include <mpi.h>
#include <nccl.h>
#include <cuda_runtime.h>

#define NELEMS 1024

#define CUDACHECK(cmd) do {                                 \
    cudaError_t e = cmd;                                    \
    if (e != cudaSuccess) {                                 \
        fprintf(stderr, "CUDA error %d: %s (%s:%d)\n",     \
                e, cudaGetErrorString(e),__FILE__,__LINE__);\
        exit(1);                                            \
    }                                                       \
} while(0)

#define NCCLCHECK(cmd) do {                                 \
    ncclResult_t r = cmd;                                   \
    if (r != ncclSuccess) {                                 \
        fprintf(stderr, "NCCL error %d: %s (%s:%d)\n",     \
                r, ncclGetErrorString(r),__FILE__,__LINE__);\
        exit(1);                                            \
    }                                                       \
} while(0)

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);

    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    assert(nranks == 2);

    CUDACHECK(cudaSetDevice(rank));

    ncclUniqueId id;
    if (rank == 0) NCCLCHECK(ncclGetUniqueId(&id));
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

    ncclComm_t comm;
    NCCLCHECK(ncclCommInitRank(&comm, 2, id, rank));

    cudaStream_t stream;
    CUDACHECK(cudaStreamCreate(&stream));

    float *d_buf;
    CUDACHECK(cudaMalloc(&d_buf, NELEMS * sizeof(float)));

    if (rank == 0) {
        float *h_buf = (float *)malloc(NELEMS * sizeof(float));
        for (int i = 0; i < NELEMS; i++) h_buf[i] = 42.0f;
        CUDACHECK(cudaMemcpy(d_buf, h_buf, NELEMS * sizeof(float),
                  cudaMemcpyHostToDevice));
        free(h_buf);

        printf("[rank 0] Sending...\n");
        NCCLCHECK(ncclSend(d_buf, NELEMS, ncclFloat, 1, comm, stream));
        CUDACHECK(cudaStreamSynchronize(stream));
        printf("[rank 0] Send complete. Sleeping 10s...\n");
        sleep(10);

    } else {
        printf("[rank 1] Sleeping 10s... [CHECKPOINT WINDOW]\n");
        sleep(10);

        printf("[rank 1] Receiving...\n");
        NCCLCHECK(ncclRecv(d_buf, NELEMS, ncclFloat, 0, comm, stream));
        CUDACHECK(cudaStreamSynchronize(stream));

        float *h_buf = (float *)malloc(NELEMS * sizeof(float));
        CUDACHECK(cudaMemcpy(h_buf, d_buf, NELEMS * sizeof(float),
                  cudaMemcpyDeviceToHost));
        int ok = 1;
        for (int i = 0; i < NELEMS; i++) {
            if (h_buf[i] != 42.0f) {
                fprintf(stderr, "[rank 1] MISMATCH [%d]: %.0f != 42\n",
                        i, h_buf[i]);
                ok = 0;
                break;
            }
        }
        if (ok) printf("[rank 1] PASS — all elements == 42\n");
        free(h_buf);
    }

    CUDACHECK(cudaFree(d_buf));
    CUDACHECK(cudaStreamDestroy(stream));
    NCCLCHECK(ncclCommDestroy(comm));
    MPI_Finalize();
    return 0;
}
