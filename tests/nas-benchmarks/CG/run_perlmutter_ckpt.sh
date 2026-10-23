#!/bin/bash
#SBATCH -A m4128
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 00:15:00
#SBATCH -G 2
#SBATCH -J nas-cg-nccl-ckpt
#SBATCH -o nas_cg_nccl_ckpt_%j.out
#SBATCH -e nas_cg_nccl_ckpt_%j.err

export SLURM_CPU_BIND="cores"

# ─────────────────────────────────────────────────────────────────────
#  NAS CG Benchmark (NCCL port) — DMTCP Checkpoint test on Perlmutter
#
#  This script:
#    1. Runs CG under DMTCP with both NCCL + CUDA plugins
#    2. Takes a checkpoint mid-run
#    3. Restarts from that checkpoint and verifies completion
# ─────────────────────────────────────────────────────────────────────

set -euo pipefail
ulimit -c unlimited

# Darshan (Perlmutter I/O profiler) wraps dup2/open/close via LD_PRELOAD.
# DMTCP also wraps these calls. The two conflict, causing infinite recursion
# in dup2 → stack overflow → SIGSEGV. Disable Darshan when using DMTCP.
export DARSHAN_DISABLE=1
module unload darshan 2>/dev/null || true

module load cudatoolkit 2>/dev/null || true
module load nccl 2>/dev/null || true

echo "=== GPU Environment (nvidia-smi) ==="
nvidia-smi || true
echo "====================================="

# Paths — adjust if your layout differs
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
SCRIPT_DIR=${SCRIPT_DIR//\/global\/u1//global/homes}
ROOT_DIR=$(cd "$SCRIPT_DIR/../../.." && pwd)
ROOT_DIR=${ROOT_DIR//\/global\/u1//global/homes}

DMTCP_ROOT=$ROOT_DIR/dmtcp

DMTCP_LAUNCH=$DMTCP_ROOT/bin/dmtcp_launch
DMTCP_CMD=$DMTCP_ROOT/bin/dmtcp_command
DMTCP_RESTART=$DMTCP_ROOT/bin/dmtcp_restart

NCCL_PLUGIN=$ROOT_DIR/bin/libncclplugin.so
CUDA_PLUGIN=$ROOT_DIR/dmtcp/plugin/cuda/libdmtcp_cuda.so
PLUGIN="$NCCL_PLUGIN:$CUDA_PLUGIN"

PORT=7783
CKPT_DELAY=8 # seconds before triggering checkpoint

export NCCL_DEBUG=WARN
export NCCL_NET_GDR_LEVEL=PHB
export DMTCP_SLEEP_ON_FAILURE=1

cd $SCRIPT_DIR

# Clean up any stale coordinator or processes
$DMTCP_CMD --coord-port $PORT --quit --quiet 2>/dev/null || true
killall -9 dmtcp_coordinator 2>/dev/null || true
sleep 0.5

# Ensure the benchmark is compiled with CLASS=A for longer execution in the source dir
echo "Compiling benchmark (CLASS=A)..."
make clean && make cg_nccl CLASS=A

# Create and switch to node-local NVMe /tmp directory to eliminate I/O bottleneck
TMP_RUN_DIR="/tmp/dmtcp_job_${SLURM_JOB_ID:-local}"
rm -rf "$TMP_RUN_DIR"
mkdir -p "$TMP_RUN_DIR"
cp cg_nccl_exe "$TMP_RUN_DIR/"
cd "$TMP_RUN_DIR"

JOB_START=$(date +%s%N)

# Phase 1: Launch under DMTCP
echo ""
echo "==> Phase 1: Initial run with checkpoint"
PHASE1_START=$(date +%s%N)

# Step 1: Start a standalone DMTCP coordinator
echo "Starting DMTCP coordinator on port $PORT..."
$DMTCP_ROOT/bin/dmtcp_coordinator --daemon --coord-port $PORT --exit-on-last
sleep 1

# Verify coordinator is running
if ! $DMTCP_CMD --coord-port $PORT --status >/dev/null 2>&1; then
  echo "ERROR: Failed to start DMTCP coordinator"
  exit 1
fi
echo "Coordinator is running."

# Step 2: Launch each task under dmtcp_launch
# Each cg_nccl_exe task is individually DMTCP-managed
export NCCL_P2P_DISABLE=1
export NCCL_CUMEM_ENABLE=0
export NCCL_DMABUF_ENABLE=0
export NCCL_NVLS_ENABLE=0
export NCCL_SHM_DISABLE=1
export NCCL_NET_GDR_LEVEL=0
export MPICH_GPU_SUPPORT_ENABLED=0
(
  $DMTCP_LAUNCH -j --coord-port $PORT --no-gzip \
    --with-plugin $NCCL_PLUGIN:$CUDA_PLUGIN \
    ./cg_nccl_exe 0 2 &
  $DMTCP_LAUNCH -j --coord-port $PORT --no-gzip \
    --with-plugin $NCCL_PLUGIN:$CUDA_PLUGIN \
    ./cg_nccl_exe 1 2 &
  wait
) >srun_launch.log 2>&1 &
PID=$!

# Wait for both processes to connect to coordinator
echo "Waiting for processes to connect to coordinator..."
while true; do
  PEERS=$($DMTCP_CMD --coord-port $PORT --status 2>/dev/null | grep NUM_PEERS | awk -F= '{print $2}' || echo "0")
  if [ "$PEERS" != "" ] && [ "$PEERS" -ge 2 ]; then
    echo "Found $PEERS connected processes."
    break
  fi
  if ! kill -0 $PID 2>/dev/null; then
    echo "ERROR: srun process died unexpectedly. Log output:"
    cat srun_launch.log
    exit 1
  fi
  sleep 0.5
done

# Wait for NCCL initialization and a few iterations to run
echo "Waiting for NCCL to initialize and computation to start..."
while true; do
  if grep -q "iteration" srun_launch.log 2>/dev/null; then
    echo "NCCL initialized, computation running."
    break
  fi
  if ! kill -0 $PID 2>/dev/null; then
    echo "ERROR: process exited before computation started. Log output:"
    cat srun_launch.log
    exit 1
  fi
  sleep 0.5
done
sleep 2

echo "Triggering checkpoint..."
CKPT_START=$(date +%s%N)
$DMTCP_CMD --coord-port $PORT --checkpoint
CKPT_TRIGGER=$(date +%s%N)
echo "Checkpoint triggered. Waiting for process to complete..."
wait $PID || true
CKPT_DONE=$(date +%s%N)
RESULT=${PIPESTATUS[0]:-$?}

CKPT_TRIGGER_MS=$(((CKPT_TRIGGER - CKPT_START) / 1000000))
CKPT_TOTAL_MS=$(((CKPT_DONE - CKPT_START) / 1000000))

if [ $RESULT -eq 0 ]; then
  echo "PASS: Initial run completed successfully"
else
  echo "WARN: Initial run exited with code $RESULT (may be expected after checkpoint)"
fi

# ─────────────────────────────────────────────────────────────────────
#  Profiling — disable strict mode so stats commands don't kill script
# ─────────────────────────────────────────────────────────────────────
set +eu

# Checkpoint file stats
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

# Extract plugin checkpoint overheads from log
NCCL_TIME_S=$(grep "\[NCCL PLUGIN\] Checkpoint overhead:" srun_launch.log | awk '{print $5}' | sort -n | tail -1 || echo "0")
CUDA_TIME_S=$(grep "\[CUDA PLUGIN\] Checkpoint overhead:" srun_launch.log | awk '{print $5}' | sort -n | tail -1 || echo "0")

if [ -z "$NCCL_TIME_S" ]; then NCCL_TIME_S=0; fi
if [ -z "$CUDA_TIME_S" ]; then CUDA_TIME_S=0; fi

CKPT_TOTAL_S=$(awk "BEGIN {print $CKPT_TOTAL_MS / 1000.0}")
DISK_TIME_S=$(awk "BEGIN {print $CKPT_TOTAL_S - $NCCL_TIME_S - $CUDA_TIME_S}")
DISK_TIME_S=$(awk "BEGIN {if ($DISK_TIME_S < 0) print 0; else printf \"%.3f\", $DISK_TIME_S}")

DISK_SPEED_MB_S="N/A"
if [ "$CKPT_SIZE_MB" != "N/A" ] && [ "$DISK_TIME_S" != "0" ] && [ "$DISK_TIME_S" != "0.000" ]; then
  DISK_SPEED_MB_S=$(awk "BEGIN {printf \"%.2f\", $CKPT_SIZE_MB / $DISK_TIME_S}")
fi

# Phase 2: Restart from checkpoint
RESTART_MS="N/A"
if ls ckpt_*.dmtcp 1>/dev/null 2>&1; then
  echo ""
  echo "==> Phase 2: Restarting from checkpoint via dmtcp_restart"

  # The coordinator was started with --exit-on-last, so it dies when
  # Phase 1 processes disconnect after checkpoint.  Start a fresh one.
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

  echo "Waiting for restarted process to complete..."
  wait $REST_PID
  RESTART_RC=$?

  RESTART_DONE=$(date +%s%N)
  RESTART_MS=$(((RESTART_DONE - RESTART_START) / 1000000))
  if [ $RESTART_RC -eq 0 ]; then
    echo "PASS: Restart completed successfully"
  else
    echo "FAIL: Restart failed (exit code $RESTART_RC)"
    echo "--- restart_launch.log ---"
    cat restart_launch.log
  fi
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
printf "║  %-40s %15s  ║\n" "Checkpoint write speed:" "${DISK_SPEED_MB_S} MB/s"
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
