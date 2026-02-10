#include <stdio.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <stdlib.h>

#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",       \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",       \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

int main(int argc, char* argv[]) {
  int nRanks = 1;
  int myRank = 0;

  ncclUniqueId id;
  ncclComm_t comm;
  cudaStream_t stream;

  printf("Initializing NCCL...\n");
  NCCLCHECK(ncclGetUniqueId(&id));
  NCCLCHECK(ncclCommInitRank(&comm, nRanks, id, myRank));
  CUDACHECK(cudaStreamCreate(&stream));

  size_t count = 1024 * 1024;
  float *sendbuff, *recvbuff;
  CUDACHECK(cudaMalloc(&sendbuff, count * sizeof(float)));
  CUDACHECK(cudaMalloc(&recvbuff, count * sizeof(float)));

  // Initialize data
  CUDACHECK(cudaMemset(sendbuff, 1, count * sizeof(float)));
  CUDACHECK(cudaMemset(recvbuff, 0, count * sizeof(float)));

  printf("Starting loop...\n");
  for (int i = 0; i < 1000; i++) {
    if (i % 5 == 0) {
      printf("Iteration %d\n", i);
      fflush(stdout); // flush for log visibility
    }
    
    NCCLCHECK(ncclAllReduce((const void*)sendbuff, (void*)recvbuff, count, ncclFloat, ncclSum, comm, stream));
    CUDACHECK(cudaStreamSynchronize(stream));
    
    sleep(1);
  }

  CUDACHECK(cudaFree(sendbuff));
  CUDACHECK(cudaFree(recvbuff));
  NCCLCHECK(ncclCommDestroy(comm));
  CUDACHECK(cudaStreamDestroy(stream));

  printf("Test Finished\n");
  return 0;
}
