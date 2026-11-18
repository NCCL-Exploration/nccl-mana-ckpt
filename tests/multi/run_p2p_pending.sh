#!/bin/bash
# Checkpoints taken while a point-to-point operation lacks its counterpart
# (tests/multi/p2p_pending.cu), with resume and restart: experiment E4 of
# experiments/README.md. Runs inside an allocation; no MPI:
#
#   salloc -A m4128 -q interactive -t 01:00:00 -N 2 -C gpu --gpus-per-node=4
#   tests/multi/run_p2p_pending.sh
#
# Build first: experiments/build.sh plugin p2ppending.
#
# The scenarios and layouts are P2P_CASES in experiments/plan.sh; cases that
# need more nodes than the allocation has are left out. A case passes when the
# run verifies (every rank received its sender's data, before and after the
# checkpoint) and no "[NCCL Drain]" line appears: the checkpoint waited for
# the missing operation instead of completing the other side with a dummy
# message. Results: $SCRATCH/nccl-ckpt-paper/p2p-pending-<job>/; tables with
# experiments/collect.sh p2p-pending-<job> (p2p.csv, summary.md).
#
# DMTCP_PLUGINS=<libncclplugin.so>:<libdmtcp_cuda.so> runs with another build
# of the plugin (absolute paths), e.g. the previous one, to see what this
# test catches. tests/multi/run_p2p_policy.sh covers the fallback when the
# counterpart never comes (NCCL_PLUGIN_P2P_WAIT).

set -uo pipefail
EXP=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../experiments" && pwd)
: "${CAMPAIGN:=p2p-pending-${SLURM_JOB_ID:?run inside an allocation}${TAG:+-$TAG}}"
export CAMPAIGN
exec "$EXP/submit.sh" --here p2p
