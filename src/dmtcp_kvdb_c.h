#ifndef DMTCP_KVDB_C_H
#define DMTCP_KVDB_C_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Increments the value at key by val. Returns 1 for success, 0 otherwise.
int dmtcp_kvdb_incr(const char *key, int64_t val);

// Sets the value at key. Returns 1 for success, 0 otherwise.
int dmtcp_kvdb_set(const char *key, int64_t val);

// Gets the value at key. Returns the value (parsed as int64_t), or 0 if not found/error.
int64_t dmtcp_kvdb_get(const char *key);

#ifdef __cplusplus
}
#endif

#endif // DMTCP_KVDB_C_H
