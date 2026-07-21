#!/bin/bash
#SBATCH -A m4128
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:15:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-node=4
#SBATCH --job-name=pytorch_ckpt_test
#SBATCH -o pytorch_ckpt_%j.out
#SBATCH -e pytorch_ckpt_%j.err

export SLURM_CPU_BIND="cores"

# ─────────────────────────────────────────────────────────────────────
#  PyTorch DDP Benchmark — DMTCP Checkpoint test on Perlmutter
#
#  This script:
#    1. Runs PyTorch DDP under DMTCP with both NCCL + CUDA plugins
#    2. Takes a checkpoint mid-training
#    3. Measures CUDA overhead, NCCL overhead, Disk Write, and Restart time
#    4. Displays the standardized CHECKPOINT PROFILING SUMMARY
# ─────────────────────────────────────────────────────────────────────

set -euo pipefail
ulimit -c unlimited

# Disable Darshan to prevent conflicts with DMTCP syscall wrappers
export DARSHAN_DISABLE=1
module unload darshan 2>/dev/null || true

module load cudatoolkit 2>/dev/null || true
module load nccl 2>/dev/null || true
module load pytorch 2>/dev/null || true

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
SCRIPT_DIR=${SCRIPT_DIR//\/global\/u1//global/homes}
ROOT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
ROOT_DIR=${ROOT_DIR//\/global\/u1//global/homes}

DMTCP_ROOT=$ROOT_DIR/dmtcp

DMTCP_LAUNCH=$DMTCP_ROOT/bin/dmtcp_launch
DMTCP_CMD=$DMTCP_ROOT/bin/dmtcp_command
DMTCP_RESTART=$DMTCP_ROOT/bin/dmtcp_restart

NCCL_PLUGIN=$ROOT_DIR/bin/libncclplugin.so
CUDA_PLUGIN=$ROOT_DIR/dmtcp/plugin/cuda/libdmtcp_cuda.so
PLUGIN="$NCCL_PLUGIN:$CUDA_PLUGIN"

PORT=7783

export NCCL_DEBUG=WARN
export NCCL_NET_GDR_LEVEL=PHB
export DMTCP_SLEEP_ON_FAILURE=1

# Standard NCCL tuning flags for DMTCP compatibility
export NCCL_P2P_DISABLE=1
export NCCL_CUMEM_ENABLE=0
export NCCL_DMABUF_ENABLE=0
export NCCL_NVLS_ENABLE=0
export NCCL_SHM_DISABLE=1
export NCCL_NET_GDR_LEVEL=0
export MPICH_GPU_SUPPORT_ENABLED=0

cd $SCRIPT_DIR

JOB_START=$(date +%s%N)

# Clean up any stale coordinator or processes
$DMTCP_CMD --coord-port $PORT --quit --quiet 2>/dev/null || true
killall -9 dmtcp_coordinator 2>/dev/null || true
sleep 0.5
rm -rf ckpt_*.dmtcp ckpt_rank_* ckpt_*_files/ dmtcp_restart_script*.sh
rm -f srun_launch.log restart_launch.log

# ─────────────────────────────────────────────────────────────────────
# Phase 1: Launch PyTorch DDP under DMTCP & Checkpoint
# ─────────────────────────────────────────────────────────────────────
echo ""
echo "==> Phase 1: Initial PyTorch DDP run with checkpoint"
PHASE1_START=$(date +%s%N)

echo "Starting DMTCP coordinator on port $PORT..."
$DMTCP_ROOT/bin/dmtcp_coordinator --daemon --coord-port $PORT --exit-on-last
sleep 1

if ! $DMTCP_CMD --coord-port $PORT --status >/dev/null 2>&1; then
  echo "ERROR: Failed to start DMTCP coordinator"
  exit 1
fi
echo "Coordinator is running."

export WORLD_SIZE=2
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT=29500

(
  export RANK=0
  export LOCAL_RANK=0
  $DMTCP_LAUNCH -j --coord-port $PORT --no-gzip \
    --with-plugin $NCCL_PLUGIN:$CUDA_PLUGIN \
    python3 pytorch_ddp_test.py --epochs -1 &

  export RANK=1
  export LOCAL_RANK=1
  $DMTCP_LAUNCH -j --coord-port $PORT --no-gzip \
    --with-plugin $NCCL_PLUGIN:$CUDA_PLUGIN \
    python3 pytorch_ddp_test.py --epochs -1 &

  wait
) >srun_launch.log 2>&1 &
PID=$!

echo "Waiting for PyTorch processes to connect to coordinator..."
while true; do
  PEERS=$($DMTCP_CMD --coord-port $PORT --status 2>/dev/null | grep NUM_PEERS | awk -F= '{print $2}' || echo "0")
  if [ "$PEERS" != "" ] && [ "$PEERS" -ge 2 ]; then
    echo "Found $PEERS connected processes."
    break
  fi
  if ! kill -0 $PID 2>/dev/null; then
    echo "ERROR: PyTorch process died unexpectedly. Log output:"
    cat srun_launch.log
    exit 1
  fi
  sleep 0.5
done

echo "Waiting for PyTorch training iterations to start..."
while true; do
  if grep -q "Epoch" srun_launch.log 2>/dev/null; then
    echo "PyTorch training active."
    break
  fi
  if ! kill -0 $PID 2>/dev/null; then
    echo "ERROR: Process exited before training started. Log output:"
    cat srun_launch.log
    exit 1
  fi
  sleep 0.5
done
sleep 3

echo "Triggering checkpoint..."
CKPT_START=$(date +%s%N)
$DMTCP_CMD --coord-port $PORT --checkpoint
CKPT_TRIGGER=$(date +%s%N)
echo "Checkpoint command sent. Waiting for checkpoint files to write..."

# Wait for checkpoint files to complete writing
while true; do
  if ls ckpt_*.dmtcp 1>/dev/null 2>&1; then
    sleep 2 # Allow final write flush
    break
  fi
  sleep 0.5
done
CKPT_DONE=$(date +%s%N)

# Terminate Phase 1 processes cleanly after checkpointing
$DMTCP_CMD --coord-port $PORT --quit --quiet 2>/dev/null || true
kill -9 $PID 2>/dev/null || true

CKPT_TRIGGER_MS=$(((CKPT_TRIGGER - CKPT_START) / 1000000))
CKPT_TOTAL_MS=$(((CKPT_DONE - CKPT_START) / 1000000))

# ─────────────────────────────────────────────────────────────────────
#  Profiling Calculations
# ─────────────────────────────────────────────────────────────────────
set +eu

CKPT_COUNT=0
CKPT_SIZE_MB="N/A"
CKPT_DIR_SIZE_MB="N/A"
if ls ckpt_*.dmtcp 1>/dev/null 2>&1; then
  CKPT_COUNT=$(ls -1 ckpt_*.dmtcp | wc -l)
  CKPT_SIZE_BYTES=$(du -csb ckpt_*.dmtcp | tail -1 | awk '{print $1}')
  if [ -n "$CKPT_SIZE_BYTES" ]; then
    CKPT_SIZE_MB=$(awk "BEGIN {printf \"%.2f\", $CKPT_SIZE_BYTES / 1048576}")
  fi
fi

PHASE1_MS=$(((CKPT_DONE - PHASE1_START) / 1000000))

# Extract plugin timing statistics from log
NCCL_TIME_S=$(grep "\[NCCL PLUGIN\] Checkpoint overhead:" srun_launch.log | awk '{print $5}' | sort -n | tail -1 || echo "0")
CUDA_TIME_S=$(grep "\[CUDA PLUGIN\] Checkpoint overhead:" srun_launch.log | awk '{print $5}' | sort -n | tail -1 || echo "0")

if [ -z "$NCCL_TIME_S" ]; then NCCL_TIME_S=0; fi
if [ -z "$CUDA_TIME_S" ]; then CUDA_TIME_S=0; fi

CKPT_TOTAL_S=$(awk "BEGIN {print $CKPT_TOTAL_MS / 1000.0}")
DISK_TIME_S=$(awk "BEGIN {print $CKPT_TOTAL_S - $NCCL_TIME_S - $CUDA_TIME_S}")
DISK_TIME_S=$(awk "BEGIN {if ($DISK_TIME_S < 0) print 0; else printf \"%.3f\", $DISK_TIME_S}")

# ─────────────────────────────────────────────────────────────────────
# Phase 2: Restart from checkpoint
# ─────────────────────────────────────────────────────────────────────
RESTART_MS="N/A"
if ls ckpt_*.dmtcp 1>/dev/null 2>&1; then
  echo ""
  echo "==> Phase 2: Restarting from checkpoint via dmtcp_restart"

  echo "Re-launching coordinator for restart phase..."
  $DMTCP_ROOT/bin/dmtcp_coordinator --daemon --coord-port $PORT --exit-on-last
  sleep 1

  RESTART_START=$(date +%s%N)

  RESTART_SCRIPT=$(ls -t dmtcp_restart_script*.sh | head -n 1)
  if [ -n "$RESTART_SCRIPT" ]; then
    ./$RESTART_SCRIPT --coord-port $PORT >restart_launch.log 2>&1 &
    REST_PID=$!
  else
    echo "No restart script found!"
    exit 1
  fi

  echo "Waiting for restarted process to resume training..."
  while true; do
    if grep -q "Epoch" restart_launch.log 2>/dev/null; then
      echo "Restarted process successfully resumed training."
      break
    fi
    if ! kill -0 $REST_PID 2>/dev/null; then
      echo "WARN: Restart process exited."
      break
    fi
    sleep 0.5
  done
  RESTART_DONE=$(date +%s%N)
  RESTART_MS=$(((RESTART_DONE - RESTART_START) / 1000000))

  # Clean up restarted processes
  $DMTCP_CMD --coord-port $PORT --quit --quiet 2>/dev/null || true
  kill -9 $REST_PID 2>/dev/null || true
else
  echo "No checkpoint directories found — skipping restart phase"
fi

JOB_DONE=$(date +%s%N)
TOTAL_MS=$(((JOB_DONE - JOB_START) / 1000000))

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║              CHECKPOINT PROFILING SUMMARY                   ║"
echo "╠══════════════════════════════════════════════════════════════╣"
printf "║  %-40s %15s  ║\n" "Phase 1 (run + checkpoint):" "${PHASE1_MS} ms"
printf "║  %-40s %15s  ║\n" "  ├─ Checkpoint trigger (cmd):" "${CKPT_TRIGGER_MS} ms"
printf "║  %-40s %15s  ║\n" "  └─ Checkpoint completion (drain+write):" "${CKPT_TOTAL_MS} ms"
printf "║  %-40s %15s  ║\n" "      ├─ CUDA Overhead:" "${CUDA_TIME_S} s"
printf "║  %-40s %15s  ║\n" "      ├─ NCCL Overhead:" "${NCCL_TIME_S} s"
printf "║  %-40s %15s  ║\n" "      └─ Disk Write (approx):" "${DISK_TIME_S} s"
echo "║──────────────────────────────────────────────────────────────║"
printf "║  %-40s %15s  ║\n" "Checkpoint files:" "${CKPT_COUNT} file(s)"
printf "║  %-40s %15s  ║\n" "Checkpoint .dmtcp size:" "${CKPT_SIZE_MB} MB"
printf "║  %-40s %15s  ║\n" "Checkpoint dir size:" "${CKPT_DIR_SIZE_MB} MB"
echo "║──────────────────────────────────────────────────────────────║"
printf "║  %-40s %15s  ║\n" "Phase 2 (restart):" "${RESTART_MS} ms"
echo "║──────────────────────────────────────────────────────────────║"
printf "║  %-40s %15s  ║\n" "Total wall time:" "${TOTAL_MS} ms"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Job completed at $(date)"
