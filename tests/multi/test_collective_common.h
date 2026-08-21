#ifndef DMTCP_NCCL_TEST_COLLECTIVE_COMMON_H
#define DMTCP_NCCL_TEST_COLLECTIVE_COMMON_H


#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <nccl.h>


#define CUDACHECK(cudacmd) do {
	cudaError_t e = (cudacmd);
	if (e != cudaSuccess) {
		fprintf(stderr, "CUDA error in %s:%d, error %s\n",
			       	__FILE__, __LINE__, cudaGetErrorString(e));
		exit(EXIT_FAILURE);
	}
} while (0)

#define NCCLCHECK(ncclcmd) do {
	ncclResult_t e = (ncclcmd);
	if (e != ncclSuccess) {
		fprintf(stderr, "NCCL error in %s:%d, error %s\n",
				__FILE__, __LINE__, ncclGetErrorString(e));
		exit(EXIT_FAILURE);
	}
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
		while (fd = fopen(filepath, "rb") == NULL) usleep(1000);
		fread(id, sizeof(*id), 1, fd);
		fclose(fd);
	}
}


#endif
