#!/bin/bash
#SBATCH -A m4128
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:10:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-node=4
#SBATCH --job-name=gemma_finetune_ckpt
#SBATCH -o gemma_finetune_%j.out
#SBATCH -e gemma_finetune_%j.err

export SLURM_SUBMIT_DIR=$(pwd)
cd $SLURM_SUBMIT_DIR

set -euo pipefail

module load cudatoolkit 2>/dev/null || true
module load nccl 2>/dev/null || true

# DMTCP & Plugin Paths
DMTCP_ROOT=${SLURM_SUBMIT_DIR}/../../dmtcp
DMTCP_LAUNCH=${DMTCP_ROOT}/bin/dmtcp_launch
DMTCP_CMD=${DMTCP_ROOT}/bin/dmtcp_command
PLUGIN=${SLURM_SUBMIT_DIR}/../../bin/libncclplugin.so:${DMTCP_ROOT}/plugin/cuda/libdmtcp_cuda.so

# NCCL / MANA Environment
export NCCL_P2P_DISABLE=0
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_DEBUG=WARN
export DMTCP_COORD_HOST=$(hostname)
PORT=7789

echo "=============================================================="
echo "==> Phase 0: Cleaning old checkpoint artifacts"
echo "=============================================================="
rm -rf ckpt_*.dmtcp ckpt_rank_* ckpt_*_files/ dmtcp_restart_script*.sh

echo "==> Launching DMTCP Coordinator on port $PORT"
${DMTCP_ROOT}/bin/dmtcp_coordinator --daemon -p $PORT

echo "=============================================================="
echo "==> Phase 1: Launching Gemma Fine-Tuning Workload under DMTCP"
echo "=============================================================="
export WORLD_SIZE=2
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT=29510

PHASE1_START=$(date +%s%N)

(
  export RANK=0
  export LOCAL_RANK=0
  $DMTCP_LAUNCH -j --coord-port $PORT --interval 0 --with-plugin $PLUGIN \
      uv run python3 gemma_finetune_test.py --epochs -1 &
      
  export RANK=1
  export LOCAL_RANK=1
  $DMTCP_LAUNCH -j --coord-port $PORT --interval 0 --with-plugin $PLUGIN \
      uv run python3 gemma_finetune_test.py --epochs -1 &
      
  wait
) &

APP_PID=$!

echo "Allowing Gemma fine-tuning to run for 25 seconds..."
sleep 25

CKPT_START=$(date +%s%N)
echo "Triggering checkpoint..."
$DMTCP_CMD --coord-port $PORT --checkpoint

echo "Waiting for checkpoint to complete..."
sleep 5
CKPT_DONE=$(date +%s%N)
CKPT_MS=$(((CKPT_DONE - CKPT_START) / 1000000))

echo "Terminating initial training process..."
$DMTCP_CMD --coord-port $PORT --quit --quiet 2>/dev/null || true
kill -9 $APP_PID 2>/dev/null || true

echo "=============================================================="
echo "==> Checkpoint complete. Generated files:"
ls -lh ckpt_*.dmtcp
echo "=============================================================="

echo "=============================================================="
echo "==> Phase 2: Restarting Gemma Fine-Tuning from Checkpoint"
echo "=============================================================="
RESTART_SCRIPT=$(ls -t dmtcp_restart_script*.sh | head -n 1)

REST_START=$(date +%s%N)
if [ -n "$RESTART_SCRIPT" ]; then
    echo "Restarting execution via $RESTART_SCRIPT..."
    ./$RESTART_SCRIPT --coord-port $PORT &
    REST_PID=$!
    
    echo "Allowing restarted training process to run for 15 seconds..."
    sleep 15
    REST_DONE=$(date +%s%N)
    REST_MS=$(((REST_DONE - REST_START) / 1000000))
    
    echo "Terminating restarted process..."
    $DMTCP_CMD --coord-port $PORT --quit --quiet 2>/dev/null || true
    kill -9 $REST_PID 2>/dev/null || true
else
    echo "Error: Restart script not found!"
    exit 1
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          GEMMA FINE-TUNING C/R PROFILING SUMMARY            ║"
echo "╠══════════════════════════════════════════════════════════════╣"
printf "║  %-40s %15s  ║\n" "Checkpoint Completion Latency:" "${CKPT_MS} ms"
printf "║  %-40s %15s  ║\n" "Restart Latency:" "${REST_MS} ms"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Gemma Fine-Tuning Checkpoint Benchmark completed cleanly at $(date)"
