#ifndef DMTCP_NCCL_TEST_COLLECTIVE_COMMON_H
#define DMTCP_NCCL_TEST_COLLECTIVE_COMMON_H


#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <string.h>
#include <stdbool.h>


#define CUDACHECK(cudacmd) do { \
	cudaError_t e = (cudacmd); \
	if (e != cudaSuccess) { \
		fprintf(stderr, "CUDA error in %s:%d, error %s\n", \
			       	__FILE__, __LINE__, cudaGetErrorString(e)); \
		exit(EXIT_FAILURE); \
	} \
} while (0)

#define NCCLCHECK(ncclcmd) do { \
	ncclResult_t e = (ncclcmd); \
	if (e != ncclSuccess) { \
		fprintf(stderr, "NCCL error in %s:%d, error %s\n", \
				__FILE__, __LINE__, ncclGetErrorString(e)); \
		exit(EXIT_FAILURE); \
	} \
} while (0)

/*
 * Passes back the unique ID to the given NCCL rank.
 * @param id is a pointer to the unique ID of the calling rank process.
 * @param rank is the rank number.
 * @throws an error if cannot generate a unique ID.
 * @throws on file errors.
 */
static inline void getUniqueIdHelp(ncclUniqueId* id, int rank) {
	const char* filepath = getenv("COLLECTIVE_TEST_ID_FILE");
	if (!filepath) filepath = "/tmp/collective_test_id.bin";
	if (rank == 0) {
		NCCLCHECK(ncclGetUniqueId(id));
		FILE* fd = fopen(filepath, "wb");
		if (!fd) { perror("fopen for writing id."); exit(EXIT_FAILURE); }
		fwrite(id, sizeof(*id), 1, fd);
		fclose(fd);
	} else {
		FILE* fd;
		while ((fd = fopen(filepath, "rb")) == NULL) usleep(1000);
		fread(id, sizeof(*id), 1, fd);
		fclose(fd);
	}
}

typedef void (*collective_fnc)(void* call_data, cudaStream_t stream);

typedef struct {
	int rank;
	int num_ranks;
	cudaStream_t stream;
	const char* ckpt_fired_flag_path;
} collective_ctx_t;

/**
 * Given a collective context, the rank, and the amount of ranks, and the stream,
 * initializes that context with those parameters.
 * @param cctx is a pointer to the context.
 * @param rank is the rank of the process running the collective call.
 * @param num_ranks is the total amount of ranks in that collective call.
 * @param stream is the cuda stream for that process.
 */
static inline void collective_init(collective_ctx_t* cctx, int rank, int num_ranks,
	       	cudaStream_t stream) {
	cctx->rank = rank;
	cctx->num_ranks = num_ranks;
	cctx->stream = stream;
	cctx->ckpt_fired_flag_path = getenv("COLLECTIVE_EXIT_FLAG_FILE");
	if (!cctx->ckpt_fired_flag_path) {
		cctx->ckpt_fired_flag_path = "/tmp/collective_test_ckpt_fired.flag";
	}
}

/**
 * Given a collective context, blocks the process until the checkpoint has been fired.
 * @param cctx is a pointer to the context.
 */
static inline void wait_for_checkpoint(collective_ctx_t* cctx) {
	const char* exit_flag = cctx->ckpt_fired_flag_path;
	while (access(exit_flag, F_OK) != 0) {
		usleep(1000);
	}
}

/**
 * Given a collective context, the call data, and a function,
 * calls that function with the given call data, the manner of which
 * is dependent on the context.
 * @param cctx is a pointer to the context.
 * @param call_data is a pointer to the collective call data.
 * @param collective_fnx is a pointer to the collective function to call.
 */
static inline void run_collective(collective_ctx_t* cctx, void* call_data,
		collective_fnc collective_fnc) {
	const char* queue_size_env = getenv("COLLECTIVE_TEST_QUEUE_SIZE");
	int min_queue_size = queue_size_env ? atoi(queue_size_env) : 500;
	printf("TEST SETUP COMPLETE\n"); fflush(stdout);
	for (int i = 0; i < min_queue_size; i++) {
		collective_fnc(call_data, cctx->stream);
	}
	// Wait for the checkpoint, and re-run the collective operation after.
	wait_for_checkpoint(cctx);
	collective_fnc(call_data, cctx->stream);
	CUDACHECK(cudaStreamSynchronize(cctx->stream));
}


#endif
