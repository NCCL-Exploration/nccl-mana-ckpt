#!/bin/bash
#SBATCH -A m4128
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:10:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-node=4
#SBATCH --job-name=pytorch_ckpt_test

export SLURM_SUBMIT_DIR=$(pwd)
cd $SLURM_SUBMIT_DIR

module load pytorch

# DMTCP Paths
DMTCP_ROOT=${SLURM_SUBMIT_DIR}/../../dmtcp
DMTCP_LAUNCH=${DMTCP_ROOT}/bin/dmtcp_launch
DMTCP_CMD=${DMTCP_ROOT}/bin/dmtcp_command
PLUGIN=${SLURM_SUBMIT_DIR}/../../bin/libncclplugin.so:${DMTCP_ROOT}/plugin/cuda/libdmtcp_cuda.so

# NCCL / MANA Environment
export NCCL_P2P_DISABLE=0
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_DEBUG=WARN
export DMTCP_COORD_HOST=$(hostname)
PORT=7783

echo "==> Cleaning old checkpoint files"
rm -rf ckpt_*.dmtcp ckpt_rank_* ckpt_*_files/ dmtcp_restart_script*.sh

echo "==> Launching Coordinator in daemon mode"
${DMTCP_ROOT}/bin/dmtcp_coordinator --daemon -p $PORT

echo "==> Phase 1: Initial PyTorch run with checkpoint"
# Run infinite epochs (-1) so we have plenty of time to checkpoint
export WORLD_SIZE=2
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT=29500

TEST_SCRIPT=${1:-pytorch_multicomm_test.py}
echo "==> Selected PyTorch Test: $TEST_SCRIPT"

(
  export RANK=0
  export LOCAL_RANK=0
  $DMTCP_LAUNCH -j --coord-port $PORT --interval 0 --with-plugin $PLUGIN \
      python3 $TEST_SCRIPT --epochs -1 &
      
  export RANK=1
  export LOCAL_RANK=1
  $DMTCP_LAUNCH -j --coord-port $PORT --interval 0 --with-plugin $PLUGIN \
      python3 $TEST_SCRIPT --epochs -1 &
      
  wait
) &

APP_PID=$!

echo "Waiting 25 seconds for PyTorch to initialize and start training..."
sleep 25

echo "Triggering checkpoint..."
$DMTCP_CMD --coord-port $PORT --checkpoint

echo "Waiting 5 seconds for checkpoint to complete..."
sleep 5

echo "Killing initial run..."
$DMTCP_CMD --coord-port $PORT --quit --quiet
kill -9 $APP_PID 2>/dev/null

echo "==> Checkpoint complete. Verifying files exist:"
ls -lh ckpt_*.dmtcp

echo "==> Phase 2: Restarting from checkpoint"
RESTART_SCRIPT=$(ls -t dmtcp_restart_script*.sh | head -n 1)

if [ -n "$RESTART_SCRIPT" ]; then
    echo "Restarting using $RESTART_SCRIPT"
    ./$RESTART_SCRIPT --coord-port $PORT &
    REST_PID=$!
    
    echo "Letting restarted process run for 15 seconds..."
    sleep 15
    
    echo "Killing restarted process..."
    $DMTCP_CMD --coord-port $PORT --quit --quiet
    kill -9 $REST_PID 2>/dev/null
else
    echo "Error: No restart script found!"
fi

echo "==> PyTorch MANA Test Complete"
