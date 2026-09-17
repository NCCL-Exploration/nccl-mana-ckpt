#include "test_collectives_common.h"
#include <string.h>


typedef struct {
	void* sendbuff;
	void* recvbuff;
	size_t count;
	ncclComm_t comm;
} allreduce_data_t;

static void call_allreduce(void* call_data, cudaStream_t stream) {
	allreduce_data_t* data = ((allreduce_data_t*) call_data);
	NCCLCHECK(ncclAllReduce(data->sendbuff, data->recvbuff, data->count,
				ncclInt, ncclSum, data->comm, stream));
}

int main(int argc, char* argv[]) {
	if (argc < 3) {
		fprintf(stderr, "Proper usage is: %s <rank> <number of ranks>\n", argv[0]);
		return 1;
	}
	int rank = atoi(argv[1]);
	int num_ranks = atoi(argv[2]);
	// Initialize CUDA device, NCCL unique ID, and communicator.
	CUDACHECK(cudaSetDevice(rank));
	ncclUniqueId Id;
	getUniqueIdHelp(&Id, rank);
	ncclComm_t comm;
	NCCLCHECK(ncclCommInitRank(&comm, num_ranks, Id, rank));
	// Initialize the buffers.
	int* sendbuff;
	int* recvbuff;
	const size_t count = 1 << 20; // about a million.
	CUDACHECK(cudaMalloc(&sendbuff, count * sizeof(int)));
	CUDACHECK(cudaMalloc(&recvbuff, count * sizeof(int)));
	int* hostbuff = malloc(count * sizeof(int));
	for (size_t i = 0; i < count; i++) hostbuff[i] = rank * 1000 + ((int) i);
	CUDACHECK(cudaMemcpy(sendbuff, hostbuff, count * sizeof(int), cudaMemcpyHostToDevice));
	// Initialize the CUDA stream.
	cudaStream_t cuStream;
	CUDACHECK(cudaStreamCreate(&cuStream));
	// Initialize context and call reduce.
	collective_ctx_t cctx;
	collective_init(&cctx, rank, num_ranks, cuStream);
	allreduce_data_t data = { sendbuff, recvbuff, count, comm };
	run_collective(&cctx, &data, call_allreduce);	
	// Retrieve the result of ncclAllReduce.
	int* result = malloc(count * sizeof(int));
	CUDACHECK(cudaMemcpy(result, recvbuff, count * sizeof(int), cudaMemcpyDeviceToHost));
	// Check that ncclAllReduce result and expected result match.
	int rank_sum = 0;
	for (int r = 0; r < num_ranks; r++) {
		rank_sum += r * 1000;
	}
	int mismatch = 0;
	for (size_t i = 0; i < count; i++) {
		int expected = rank_sum + (((int) i) * num_ranks);
		if (result[i] != expected) {
			fprintf(stderr, "MISMATCH: at index %zu; expected %d but result was %d\n",
					i, expected, result[i]);
			mismatch = 1;
			break;
		}
	}
	// Free memory.
	free(hostbuff);
	free(result);
	cudaFree(sendbuff);
	cudaFree(recvbuff);
	cudaStreamDestroy(cuStream);
	ncclCommDestroy(comm);
	// Return mismatch result.
	printf(mismatch ? "TEST FILE FAIL\n" : "TEST FILE PASS\n"); fflush(stdout);
	return mismatch;
}	
