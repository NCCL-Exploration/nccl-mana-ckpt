#!/bin/bash
#SBATCH -A m4128
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:10:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-node=4
#SBATCH --job-name=pytorch_ckpt_test
#SBATCH -o pytorch_ckpt_%j.out
#SBATCH -e pytorch_ckpt_%j.err

# ─────────────────────────────────────────────────────────────────────
#  PyTorch DDP (NCCL-CR) — DMTCP Checkpoint/Restart Profiling on Perlmutter
#
#  This script:
#    1. Launches PyTorch DDP under DMTCP with NCCL + CUDA plugins
#    2. Measures startup latency to active distributed training
#    3. Triggers a mid-training checkpoint with active collectives
#    4. Measures NCCL coordination, CUDA freeze, and storage write times
#    5. Restarts from checkpoint and verifies training continuity
#    6. Computes comprehensive metrics and exports structured JSON report
# ─────────────────────────────────────────────────────────────────────

set -euo pipefail
ulimit -c unlimited

export SLURM_CPU_BIND="cores"
export DARSHAN_DISABLE=1
module unload darshan 2>/dev/null || true

module load cudatoolkit 2>/dev/null || true
module load nccl 2>/dev/null || true
module load pytorch 2>/dev/null || true

echo "=== GPU Environment (nvidia-smi) ==="
nvidia-smi || true
echo "====================================="

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
SCRIPT_DIR=${SCRIPT_DIR//\/global\/u1//global/homes}
ROOT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
ROOT_DIR=${ROOT_DIR//\/global\/u1//global/homes}

DMTCP_ROOT=$ROOT_DIR/dmtcp
DMTCP_LAUNCH=$DMTCP_ROOT/bin/dmtcp_launch
DMTCP_CMD=$DMTCP_ROOT/bin/dmtcp_command
DMTCP_RESTART=$DMTCP_ROOT/bin/dmtcp_restart

NCCL_PLUGIN=$ROOT_DIR/bin/libncclplugin.so
CUDA_PLUGIN=$DMTCP_ROOT/plugin/cuda/libdmtcp_cuda.so
PLUGIN="$NCCL_PLUGIN:$CUDA_PLUGIN"

PORT=7783
PT_GPUS=${PT_GPUS:-4}
PT_RANKS=${PT_RANKS:-4}
PT_CKPT_DELAY=${PT_CKPT_DELAY:-10}
PT_STARTUP_TIMEOUT=${PT_STARTUP_TIMEOUT:-600}
PT_CKPT_TIMEOUT=${PT_CKPT_TIMEOUT:-400}
PT_RESTART_RUN=${PT_RESTART_RUN:-45}
PT_STORE_FILE=${PT_STORE_FILE:-}

PT_PY=$(command -v python3 || command -v python)
export NCCL_DEBUG=WARN
export NCCL_MULTI_RANK_GPU_ENABLE=1
export DMTCP_SLEEP_ON_FAILURE=1

cd "$SCRIPT_DIR"

# Source DMTCP-PyTorch runtime environment and shim
PT_ENV_DIR=$SCRIPT_DIR . "$SCRIPT_DIR/pytorch_dmtcp_env.sh"

# Clean up any stale coordinator or processes
$DMTCP_CMD --coord-port $PORT --quit --quiet 2>/dev/null || true
killall -9 dmtcp_coordinator 2>/dev/null || true
sleep 0.5

# Switch to node-local NVMe /tmp directory if available, or stay in local test dir
TMP_RUN_DIR="/tmp/pytorch_dmtcp_${SLURM_JOB_ID:-local}"
rm -rf "$TMP_RUN_DIR"
mkdir -p "$TMP_RUN_DIR"
cp "$SCRIPT_DIR/pytorch_minimal_test.py" "$SCRIPT_DIR/report_metrics.py" "$TMP_RUN_DIR/"
cp -r "$SCRIPT_DIR/pyshim" "$TMP_RUN_DIR/" 2>/dev/null || true
cd "$TMP_RUN_DIR"

JOB_START=$(date +%s%N)

# ─────────────────────────────────────────────────────────────────────
# Phase 1: Launch under DMTCP & Trigger Checkpoint
# ─────────────────────────────────────────────────────────────────────
echo ""
echo "==> Phase 1: Initial PyTorch DDP run with mid-training checkpoint"
PHASE1_START=$(date +%s%N)

# Start standalone coordinator
echo "Starting DMTCP coordinator on port $PORT..."
$DMTCP_ROOT/bin/dmtcp_coordinator --daemon -p $PORT >/dev/null 2>&1
sleep 1

if ! $DMTCP_CMD --coord-port $PORT --status >/dev/null 2>&1; then
  echo "ERROR: Failed to start DMTCP coordinator"
  exit 1
fi
echo "Coordinator is running."

# FileStore flock verification if enabled
if [ -n "$PT_STORE_FILE" ] && command -v flock >/dev/null 2>&1; then
  flock -n "$PT_STORE_FILE" true 2>/dev/null || {
    echo "FAIL: $PT_STORE_FILE does not support flock()."
    exit 1
  }
fi

PT_ENV="WORLD_SIZE=$PT_RANKS MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 PT_STORE_FILE=$PT_STORE_FILE NCCL_MULTI_RANK_GPU_ENABLE=1 NCCL_DEBUG=WARN"

PIDS=""
for r in $(seq 0 $((PT_RANKS - 1))); do
  setsid env $PT_ENV RANK=$r LOCAL_RANK=$((r % PT_GPUS)) \
    $DMTCP_LAUNCH --coord-port $PORT --interval 0 --no-gzip \
    --with-plugin $PLUGIN $PT_PY pytorch_minimal_test.py --steps -1 \
    > pt_rank$r.log 2>&1 &
  PIDS="$PIDS $!"
done

echo "Waiting up to ${PT_STARTUP_TIMEOUT}s for $PT_RANKS rank(s) to start training..."
ready=0
for i in $(seq 1 $PT_STARTUP_TIMEOUT); do
  st=$($DMTCP_CMD --coord-port $PORT --status 2>/dev/null || true)
  np=$(echo "$st" | grep -o 'NUM_PEERS=[0-9]*' | cut -d= -f2 || echo "0")
  rn=$(echo "$st" | grep -o 'RUNNING=[a-z]*' | cut -d= -f2 || echo "no")
  trained=$(grep -l "all_reduce ok" pt_rank*.log 2>/dev/null | wc -l || echo "0")
  if [ "$np" = "$PT_RANKS" ] && [ "$rn" = "yes" ] && [ "$trained" -ge "$PT_RANKS" ]; then
    ready=1
    break
  fi
  if grep -alE "Traceback \(most recent call last\)|^FAIL " pt_rank*.log >/dev/null 2>&1; then
    echo "ERROR: A rank failed during startup:"
    break
  fi
  alive=0
  for q in $PIDS; do kill -0 $q 2>/dev/null && alive=$((alive + 1)) || true; done
  if [ "$alive" -eq 0 ] && [ "$i" -gt 5 ]; then
    echo "ERROR: All launcher processes exited before training started."
    break
  fi
  sleep 1
done

if [ "$ready" -ne 1 ]; then
  echo "FAIL: Ranks never reached training (waited ${i}s, max ${PT_STARTUP_TIMEOUT}s)"
  for f in pt_rank*.log; do echo "--- $f ---"; tail -n 12 "$f"; done
  $DMTCP_CMD --coord-port $PORT --quit --quiet 2>/dev/null || true
  for q in $PIDS; do kill -9 -$q 2>/dev/null || kill -9 $q 2>/dev/null || true; done
  exit 1
fi

READY_TIME=$(date +%s%N)
STARTUP_MS=$(((READY_TIME - PHASE1_START) / 1000000))
echo "All $PT_RANKS rank(s) training (startup took ${STARTUP_MS}ms). Checkpointing in ${PT_CKPT_DELAY}s..."
sleep "$PT_CKPT_DELAY"

echo "Triggering checkpoint mid-training..."
CKPT_START=$(date +%s%N)
timeout "$PT_CKPT_TIMEOUT" $DMTCP_CMD --coord-port $PORT --checkpoint
CKPT_TRIGGER=$(date +%s%N)
CKPT_TRIGGER_MS=$(((CKPT_TRIGGER - CKPT_START) / 1000000))

# Wait for checkpoint files to be completely written
for i in $(seq 1 40); do
  [ $(ls ckpt_*.dmtcp 2>/dev/null | wc -l) -ge "$PT_RANKS" ] && break
  sleep 5
done
CKPT_DONE=$(date +%s%N)
CKPT_TOTAL_MS=$(((CKPT_DONE - CKPT_START) / 1000000))

grep -aE "^\[pt\]|NCCL Plugin|NCCL PLUGIN" pt_rank0.log | tail -20 || true
nimg=$(ls ckpt_*.dmtcp 2>/dev/null | wc -l || echo "0")
PRE_STEPS=$(grep -a "all_reduce ok" pt_rank0.log 2>/dev/null | tail -1 | grep -o 'step [0-9]*' | awk '{print $2}' || echo "0")

# Terminate initial run processes
$DMTCP_CMD --coord-port $PORT --quit --quiet 2>/dev/null || true
for q in $PIDS; do kill -9 -$q 2>/dev/null || kill -9 $q 2>/dev/null || true; done

if ! grep -aq "sequence consensus converged" pt_rank0.log; then
  echo "FAIL: Plugin never reached a consistent cut across communicators"
  exit 1
fi

if [ "$nimg" -lt "$PT_RANKS" ]; then
  echo "FAIL: $nimg checkpoint images found, expected $PT_RANKS"
  exit 1
fi

# ─────────────────────────────────────────────────────────────────────
# Phase 2: Restart from Checkpoint
# ─────────────────────────────────────────────────────────────────────
latest=$(ls -t ckpt_*.dmtcp | head -n 1)
echo ""
echo "==> Phase 2: Restarting from checkpoint: $latest"

RESTART_START=$(date +%s%N)
$DMTCP_RESTART --coord-port $PORT ckpt_*.dmtcp > pt_restart.log 2>&1 &
RPID=$!

FIRST_STEP_MS="N/A"
for i in $(seq 1 $PT_RESTART_RUN); do
  if [ "$FIRST_STEP_MS" = "N/A" ] && grep -aq "all_reduce ok" pt_restart.log 2>/dev/null; then
    NOW=$(date +%s%N)
    FIRST_STEP_MS=$(((NOW - RESTART_START) / 1000000))
  fi
  sleep 1
done

kill -9 $RPID 2>/dev/null || true
$DMTCP_CMD --coord-port $PORT --quit --quiet 2>/dev/null || true

RESTART_DONE=$(date +%s%N)
RESTART_MS=$(((RESTART_DONE - RESTART_START) / 1000000))
JOB_DONE=$(date +%s%N)
JOB_TOTAL_MS=$(((JOB_DONE - JOB_START) / 1000000))

grep -aE "^\[pt\]|NCCL Plugin|rebuild" pt_restart.log | tail -12 || true

if grep -aqE "Error\]|JASSERT|Terminating|Segmentation fault|Killed" pt_restart.log; then
  echo "FAIL: dmtcp_restart encountered errors:"
  grep -aE "Error\]|JASSERT|Terminating|Segmentation fault|Killed" pt_restart.log | head -5
  exit 1
fi

nreb=$(grep -ac "rebuild: comm" pt_restart.log || echo "0")
if [ "$nreb" -lt "$PT_RANKS" ]; then
  echo "FAIL: $nreb communicators rebuilt, expected $PT_RANKS"
  exit 1
fi

steps=$(grep -ac "all_reduce ok" pt_restart.log || echo "0")
if [ "$steps" -lt 3 ]; then
  echo "FAIL: Restarted run made no progress"
  exit 1
fi

# ─────────────────────────────────────────────────────────────────────
# Metrics Extraction and Reporting
# ─────────────────────────────────────────────────────────────────────
set +e
CKPT_BYTES=$(du -csb ckpt_*.dmtcp 2>/dev/null | tail -1 | awk '{print $1}' || echo "0")
NCCL_TIME_S=$(grep -a "\[NCCL PLUGIN\] Checkpoint overhead:" pt_rank*.log 2>/dev/null | awk '{print $5}' | sort -n | tail -1 || echo "0")
CUDA_TIME_S=$(grep -a "\[CUDA PLUGIN\] Checkpoint overhead:" pt_rank*.log 2>/dev/null | awk '{print $5}' | sort -n | tail -1 || echo "0")

# Run structured report metrics generator
$PT_PY report_metrics.py \
  --startup-ms "$STARTUP_MS" \
  --ckpt-trig-ms "$CKPT_TRIGGER_MS" \
  --ckpt-total-ms "$CKPT_TOTAL_MS" \
  --restart-total-ms "$RESTART_MS" \
  --first-step-ms "$FIRST_STEP_MS" \
  --job-total-ms "$JOB_TOTAL_MS" \
  --ckpt-bytes "$CKPT_BYTES" \
  --nccl-overhead-sec "${NCCL_TIME_S:-0}" \
  --cuda-overhead-sec "${CUDA_TIME_S:-0}" \
  --ckpt-file-count "$nimg" \
  --rebuilt-comms "$nreb" \
  --pre-steps "${PRE_STEPS:-0}" \
  --post-steps "$steps" \
  --ranks "$PT_RANKS" \
  --gpus "$PT_GPUS" \
  --store "$(if [ -n "$PT_STORE_FILE" ]; then echo "FileStore ($PT_STORE_FILE)"; else echo "TCPStore (env://)"; fi)" \
  --rank0-log "pt_rank0.log" \
  --restart-log "pt_restart.log" \
  --json-out "$SCRIPT_DIR/pt_ckpt_metrics.json"

echo "PASS: PyTorch DDP checkpointed, restarted, $nreb comm(s) rebuilt, training resumed."
echo "Job completed at $(date)"
