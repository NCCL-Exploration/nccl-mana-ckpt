/*
 * The lower half's view of the netsplit interface (../include/netsplit-api.h).
 *
 * MANA's loader sources (mem-wrapper.cpp, switch-context.cpp, ...) name the
 * shared structure LowerHalfInfo_t and reach it through lh_info; netsplit's
 * structure keeps MANA's first fields in the same order, so those sources
 * compile unchanged against it.  There is no MPI here: no function table, no
 * MPI constants.
 */
#ifndef _LOWER_HALF_API_H
#define _LOWER_HALF_API_H

#include <stddef.h>
#include <stdint.h>
#include <vector>

#include "netsplit-api.h"
#include "switch-context.h"

typedef NetsplitInfo_t LowerHalfInfo_t;

extern LowerHalfInfo_t *lh_info;

// The FS to enter the lower half with (see JUMP_TO_LOWER_HALF()).  MANA gave
// its checkpoint thread a TLS of its own here; in netsplit the lower half's
// own code always runs on its main TLS (the upper half's net proxy borrows a
// parked thread's TLS itself), so this is the identity.
static inline unsigned long
lower_half_fs(unsigned long lh_fs, unsigned long uh_fs)
{
  (void)uh_fs;
  return lh_fs;
}

std::vector<MmapInfo_t> &get_mmapped_list(int *num);
typedef std::vector<MmapInfo_t>& (*get_mmapped_list_fptr_t)(int *num);

#endif // ifndef _LOWER_HALF_API_H
