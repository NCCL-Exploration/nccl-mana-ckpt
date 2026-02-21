#include "dmtcp_kvdb_c.h"
#include "kvdb.h"
#include <string>

using dmtcp::kvdb::KVDBRequest;
using dmtcp::kvdb::KVDBResponse;

static const char *NCCL_DB = "/plugin/MANA/nccl-pending";

extern "C" int dmtcp_kvdb_incr(const char *key, int64_t val) {
    if (!key) return 0;
    // NCCL_DB path must match what was used in original nccl_wrappers.cpp
    // Original used "/plugin/MANA/nccl-pending"
    return (dmtcp::kvdb::request64(KVDBRequest::INCRBY, NCCL_DB, key, val) == KVDBResponse::SUCCESS) ? 1 : 0;
}

extern "C" int64_t dmtcp_kvdb_get(const char *key) {
    if (!key) return 0;
    int64_t val = 0;
    
    if (dmtcp::kvdb::get64(NCCL_DB, key, &val) == KVDBResponse::SUCCESS) {
        return val;
    }
    return 0;
}
