#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "nccl_wrappers.h"
#include "dmtcp.h"

size_t get_nccl_type_size(ncclDataType_t type) {
  switch (type) {
    case ncclInt8: case ncclUint8: return 1;
    case ncclFloat16: case ncclBfloat16: return 2;
    case ncclInt32: caseUint32: case ncclFloat32: return 4;
    case ncclInt64: case ncclUint64: case ncclFloat64: return 8;
    default: return 4; // just default to 4-byte float 
  }
}

ncclResult_t ncclIprobe(int source, ncclComm_t comm, int* flag, size_t* count) {
  if (flag == NULL || count == NULL) return ncclInvalidArgument;

  // determine rank
  int myRank;
  ncclCommUserRank(comm, &myRank);

  // format key and query
  char key[64];
  snprintf(key, sizeof(key), "%d|%d", source, myRank);
  const char* valueStr = dmtcp_kv_get(key);

  if (valueStr != NULL) {
    size_t bytesPending = (size_t)strtoul(valueStr, NULL, 10);

    if (bytesPending > 0) {
      *flag = 1;
      *count = bytesPending;
      printf("[NCCL-MANA] Probe: Found %zu bytes from %d to %d\n", bytesPending, source, myRank);
    } else {
      *flag = 0;
      *count = 0;
    }
  } else {
    // key does not exist
    *flag = 0;
    *count = 0;
  }

  return ncclSuccess;
}
