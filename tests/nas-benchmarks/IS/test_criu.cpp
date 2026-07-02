#include <stdio.h>
#include <unistd.h>
#include <cuda.h>
#include <mpi.h>
#include "nccl.h"

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    cuInit(0);
    int pid = getpid();
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    ncclUniqueId id;
    if (rank == 0) ncclGetUniqueId(&id);
    MPI_Bcast((void*)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    
    ncclComm_t comm;
    ncclCommInitRank(&comm, size, id, rank);
    
    cudaStream_t s;
    cudaStreamCreate(&s);
    
    void* d_ptr;
    cudaMalloc(&d_ptr, 1024);
    cudaMemsetAsync(d_ptr, 0, 1024, s); // Do not sync!
    
    ncclAllReduce(d_ptr, d_ptr, 256, ncclFloat, ncclSum, comm, s);
    cudaStreamSynchronize(s);
    
    CUcheckpointLockArgs lock_args = {0};
    CUresult err = cuCheckpointProcessLock(pid, &lock_args);
    if (err != CUDA_SUCCESS) {
        printf("Lock failed: %d\n", err);
        return 1;
    }
    printf("Lock succeeded.\n");

    CUcheckpointCheckpointArgs checkpoint_args = {0};
    err = cuCheckpointProcessCheckpoint(pid, &checkpoint_args);
    if (err != CUDA_SUCCESS) {
        printf("Checkpoint failed: %d\n", err);
        return 1;
    }
    printf("Checkpoint succeeded.\n");
    return 0;
}
