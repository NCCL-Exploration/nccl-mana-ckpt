/*
 * cc_simple.c
 *
 * 2 MPI processes on 2 GPUs.
 * Rank 0: ncclSend → sleep(30)
 * Rank 1: sleep(10) → ncclRecv
 *
 * Build & Run:
 *   Run `make drain_check` in parent directory
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <assert.h>
#include <mpi.h>
#include <nccl.h>
#include <cuda_runtime.h>

#define NELEMS 1024
#define NITERS 100

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
    float *h_buf;
    h_buf = (float *)malloc(NELEMS * sizeof(float));

    for (int i = 0; i < NITERS; i++) {
        for (int j = 0; j < NELEMS; j++) h_buf[j] = (float)rank;
        CUDACHECK(cudaMemcpy(d_buf, h_buf, NELEMS * sizeof(float),
                             cudaMemcpyHostToDevice));
        NCCLCHECK(ncclAllReduce(d_buf, d_buf, NELEMS,
                                ncclFloat, ncclSum, comm, stream));
        CUDACHECK(cudaStreamSynchronize(stream));

        CUDACHECK(cudaMemcpy(h_buf, d_buf, NELEMS * sizeof(float),
                             cudaMemcpyDeviceToHost));
        for (int j = 0; j < NELEMS; j++) {
            if (h_buf[j] != (float)(nranks - 1) * nranks / 2) {
                fprintf(stderr, "[rank %d] MISMATCH iter %d\n", rank, i);
                exit(1);
            }
        }
        if (i == 10){
	    printf("[rank %d] iter 10 — CHECKPOINT NOW: dmtcp_command -c\n", rank);
	    sleep(10);
	}
    }

    CUDACHECK(cudaFree(d_buf));
    free(h_buf);
    CUDACHECK(cudaStreamDestroy(stream));
    NCCLCHECK(ncclCommDestroy(comm));
    printf("[rank %d] PASS\n", rank);
    MPI_Finalize();
    return 0;
}
