#!/bin/bash
# Runs "$@" with the rank variables PyTorch reads, from Slurm's (no MPI, no
# torchrun): RANK, WORLD_SIZE, LOCAL_RANK.  Values already set are kept.
export RANK=${RANK:-$SLURM_PROCID} WORLD_SIZE=${WORLD_SIZE:-$SLURM_NTASKS} \
    LOCAL_RANK=${LOCAL_RANK:-$SLURM_LOCALID}
exec "$@"
