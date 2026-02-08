#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "nccl_wrappers.h"
#include "dmtcp_kvdb_c.h"

ncclResult_t ncclIprobe(int source, ncclComm_t comm, int* flag, size_t* count) {
  if (flag == NULL || count == NULL) return ncclInvalidArgument;

  // determine rank
  int myRank;
  ncclCommUserRank(comm, &myRank);

  // format key and query
  char key[64];
  snprintf(key, sizeof(key), "%d|%d", source, myRank);
  
  // Use our wrapper to get the count
  int64_t msgCount = dmtcp_kvdb_get(key);

  if (msgCount > 0) {
      *flag = 1;
      *count = (size_t)msgCount;
      printf("[NCCL-MANA] Probe: Found %ld messages from %d to %d\n", msgCount, source, myRank);
  } else {
      *flag = 0;
      *count = 0;
  }

  return ncclSuccess;
}
