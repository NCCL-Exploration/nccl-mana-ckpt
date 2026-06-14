#!/bin/bash
#SBATCH -A m4128
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 00:15:00
#SBATCH -n 2
#SBATCH -c 16
#SBATCH --gpus-per-task=1
#SBATCH -J nas-is-nccl-ckpt
#SBATCH -o nas_is_nccl_ckpt_%j.out
#SBATCH -e nas_is_nccl_ckpt_%j.err

export SLURM_CPU_BIND="cores"

# ─────────────────────────────────────────────────────────────────────
#  NAS IS Benchmark (NCCL port) — DMTCP Checkpoint test on Perlmutter
#
#  This script:
#    1. Runs IS under DMTCP with both NCCL + CRAC plugins
#    2. Takes a checkpoint mid-run
#    3. Restarts from that checkpoint and verifies completion
# ─────────────────────────────────────────────────────────────────────

set -euo pipefail

module load cudatoolkit 2>/dev/null || true

# Paths — adjust if your layout differs
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/../../.." && pwd)

DMTCP_ROOT=${DMTCP_ROOT:-$ROOT_DIR/dmtcp}
CRAC_ROOT=${CRAC_ROOT:-$ROOT_DIR/crac}

DMTCP_LAUNCH=$DMTCP_ROOT/bin/dmtcp_launch
DMTCP_CMD=$DMTCP_ROOT/bin/dmtcp_command
DMTCP_RESTART=$DMTCP_ROOT/bin/dmtcp_restart

NCCL_PLUGIN=$ROOT_DIR/src/libncclplugin.so
CRAC_PLUGIN=$CRAC_ROOT/libdmtcp_crac.so
PLUGIN="$NCCL_PLUGIN:$CRAC_PLUGIN"

PORT=7783
CKPT_DELAY=5   # seconds before triggering checkpoint

export NCCL_DEBUG=WARN
export NCCL_NET_GDR_LEVEL=PHB

cd $SCRIPT_DIR

echo "============================================="
echo "  NAS IS Benchmark — DMTCP Checkpoint Test"
echo "  DMTCP_ROOT: $DMTCP_ROOT"
echo "  Plugins:    $PLUGIN"
echo "  Date:       $(date)"
echo "============================================="

# Clean up any stale coordinator
$DMTCP_CMD --coord-port $PORT --quit --quiet 2>/dev/null || true
rm -f ckpt_*.dmtcp

# Phase 1: Launch under DMTCP
echo ""
echo "==> Phase 1: Initial run with checkpoint"
$DMTCP_LAUNCH --coord-port $PORT --interval 0 \
  --with-plugin $PLUGIN \
  srun ./is_nccl_exe &
PID=$!

# Wait and trigger checkpoint
sleep $CKPT_DELAY
echo "Triggering checkpoint..."
$DMTCP_CMD --coord-port $PORT --checkpoint || true
wait $PID
RESULT=$?

if [ $RESULT -eq 0 ]; then
    echo "PASS: Initial run completed successfully"
else
    echo "FAIL: Initial run exited with code $RESULT"
    exit 1
fi

# Phase 2: Restart from checkpoint
CKPT=$(ls -t ckpt_*.dmtcp 2>/dev/null | head -n 1)
if [ -n "$CKPT" ]; then
    echo ""
    echo "==> Phase 2: Restarting from checkpoint: $CKPT"
    $DMTCP_RESTART --coord-port $PORT $CKPT
    if [ $? -eq 0 ]; then
        echo "PASS: Restart completed successfully"
    else
        echo "FAIL: Restart failed"
        exit 1
    fi
else
    echo "No checkpoint file found — skipping restart phase"
fi

echo ""
echo "Job completed at $(date)"
