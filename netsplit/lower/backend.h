/*
 * The lower half's side of netsplit (backend.cpp): NCCL's network back end
 * and the CUDA stub it links with, the back end's logger, and the pool of
 * parked threads whose TLS the upper half's net proxy borrows.
 */
#ifndef NETSPLIT_BACKEND_H
#define NETSPLIT_BACKEND_H

#include "lower-half-api.h"

// Fills in the back end's part of 'info' (netsplit-api.h).  The back end
// itself is loaded only when the upper half calls info->backend_load().
void netsplit_init_backend_info(LowerHalfInfo_t *info);

#endif // ifndef NETSPLIT_BACKEND_H
