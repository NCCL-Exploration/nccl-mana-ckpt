#include "dmtcp_kvdb_c.h"
#include <map>
#include <string>

// Local in-process key-value store.
// Replaces DMTCP's coordinator-based KVDB, which is not available in CRAC.
// This is functionally equivalent for single-node / single-process tests.
static std::map<std::string, int64_t> g_kvdb;

extern "C" int dmtcp_kvdb_incr(const char *key, int64_t val) {
    if (!key) return 0;
    g_kvdb[key] += val;
    return 1;
}

extern "C" int64_t dmtcp_kvdb_get(const char *key) {
    if (!key) return 0;
    auto it = g_kvdb.find(key);
    if (it != g_kvdb.end()) {
        return it->second;
    }
    return 0;
}
