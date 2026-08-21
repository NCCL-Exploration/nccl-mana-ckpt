#include "test_collective_common.h"
#include <string.h>


int main(int argc, char* argv[]) {
	if (argc < 3) {
		fprintf(stderr, "Proper usage is: %s <rank> <number of ranks>\n", argv[0]);
		return 1;
	}
	int rank = atoi(argv[1]);
	int num_ranks = atoi(argv[2]);
	const char* test_scenario = getenv("COLLECTIVE_TEST_SCENARIO");
	if (!test_scenario) test_scenario = "POST_COLLECTIVE";
	const char* ckpt_fired_flag_path = getenv("COLLECTIVE_TEST_CKPT_FIRED_FLAG_FILE");
	if (!ckpt_fired_flag_path) ckpt_fired_flag_path = "/tpm/collective_test_ckpt_fired.flag";
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
	// Sequentially check all scenarios.
	if (strcmp(test_scenario, "PRE_COLLECTIVE") == 0) {
		printf("MARKER:PRE_COLLECTIVE\n"); fflush(stdout);
		while (access(ckpt_fired_flag_path, F_OK) != 0) usleep(1000);
		NCCLCHECK(ncclAllReduce(sendbuff, recvbuff, count, ncclInt, ncclSum, comm, cuStream));
		CUDACHECK(cudaStreamSynchronize(cuStream));
	} else if (strcmp(test_scenario, "STRAGGLING_RANK") == 0) {
		if (rank == (num_ranks - 1)) {
			printf("MARKER:STRAGGLING_RANK\n"); fflush(stdout);
			while (access(ckpt_fired_flag_path, F_OK) != 0) usleep(1000);
		}
		NCCLCHECK(ncclAllReduce(sendbuff, recvbuff, count, ncclInt, ncclSum, comm, cuStream));
		if (rank != (num_ranks - 1)) {
			printf("MARKER:STRAGGLING_RANK\n"); fflush(stdout);
		}
		CUDACHECK(cudaStreamSynchronize(cuStream));
	} else if (strcmp(test_scenario, "INFLIGHT") == 0) {
		NCCLCHECK(ncclAllReduce(sendbuff, recvbuff, count, ncclInt, ncclSum, comm, cuStream));
		if (cudaStreamQuery(cuStream) == cudaErrorNotReady) {
			printf("MARKER:INFLIGHT\n"); fflush(stdout);
			while (access(ckpt_fired_flag_path, F_OK) != 0) usleep(1000);
		}
		CUDACHECK(cudaStreamSynchronize(cuStream));
	} else { // post collective case.
		NCCLCHECK(ncclAllReduce(sendbuff, recvbuff, count, ncclInt, ncclSum, comm, cuStream));
		CUDACHECK(cudaStreamSynchronize(cuStream));
		printf("MARKER:POST_COLLECTIVE\n"); fflush(stdout);
		while (access(ckpt_fired_flag_path, F_OK) != 0) usleep(1000);
	}
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
	printf(mismatch ? "TEST FILE FAIL\n" : "TEST FILE PASS\n");
	// Free memory.
	free(hostbuff);
	free(result);
	cudaFree(sendbuff);
	cudaFree(recvbuff);
	cudaStreamDestroy(cuStream);
	ncclCommDestroy(comm);
	// Return mismatch result.
	return mismatch;
}	
