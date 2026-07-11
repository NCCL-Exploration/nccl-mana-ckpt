#include "dmtcp_kvdb_c.h"
#include "dmtcp.h"
#include "kvdb.h"
#include <string>

// Distributed key-value store using DMTCP Coordinator's Name Service.
// This replaces the local in-process map to support multi-node synchronization.

using namespace dmtcp::kvdb;

extern "C" int dmtcp_kvdb_incr(const char *key, int64_t val) {
    if (!key) return 0;
    dmtcp_disable_ckpt();
    // Using "nccl-mana" as the database ID for the coordinator's name service.
    KVDBResponse res = request64(KVDBRequest::INCRBY, "nccl-mana", key, val);
    dmtcp_enable_ckpt();
    return (res == KVDBResponse::SUCCESS);
}

extern "C" int dmtcp_kvdb_set(const char *key, int64_t val) {
    if (!key) return 0;
    dmtcp_disable_ckpt();
    KVDBResponse res = set64("nccl-mana", key, val);
    dmtcp_enable_ckpt();
    return (res == KVDBResponse::SUCCESS);
}

extern "C" int64_t dmtcp_kvdb_get(const char *key) {
    if (!key) return 0;
    dmtcp_disable_ckpt();
    int64_t oldVal = 0;
    KVDBResponse res = get64("nccl-mana", key, &oldVal);
    dmtcp_enable_ckpt();
    if (res == KVDBResponse::SUCCESS) {
        return oldVal;
    }
    return 0;
}
