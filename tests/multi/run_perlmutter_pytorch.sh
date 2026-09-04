#!/bin/bash
#SBATCH -A m4128
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:10:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-node=4
#SBATCH --job-name=pytorch_ckpt_test

# Resolve directories robustly
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${SCRIPT_DIR}"

# ─────────────────────────────────────────────────────────────────────
# Perlmutter Darshan Fix:
# Darshan (I/O profiler) wraps dup2/open/close via LD_PRELOAD, conflicting
# with DMTCP and causing infinite recursion / immediate SIGSEGV.
# ─────────────────────────────────────────────────────────────────────
export DARSHAN_DISABLE=1
module unload darshan 2>/dev/null || true
module load pytorch 2>/dev/null || true
module load cudatoolkit 2>/dev/null || true
module load nccl 2>/dev/null || true

# DMTCP Paths
DMTCP_ROOT="${REPO_ROOT}/dmtcp"
DMTCP_LAUNCH="${DMTCP_ROOT}/bin/dmtcp_launch"
DMTCP_CMD="${DMTCP_ROOT}/bin/dmtcp_command"
NCCL_PLUGIN="${REPO_ROOT}/bin/libncclplugin.so"
CUDA_PLUGIN="${DMTCP_ROOT}/plugin/cuda/libdmtcp_cuda.so"
PLUGIN="${NCCL_PLUGIN}:${CUDA_PLUGIN}"

# Verify plugins exist
if [ ! -f "${NCCL_PLUGIN}" ]; then
    echo "ERROR: NCCL plugin not found at ${NCCL_PLUGIN}. Run 'make' in repo root."
    exit 1
fi
if [ ! -f "${CUDA_PLUGIN}" ]; then
    echo "ERROR: CUDA plugin not found at ${CUDA_PLUGIN}. Run 'cd dmtcp/plugin/cuda && make libdmtcp_cuda.so'."
    exit 1
fi

# NCCL / DMTCP Environment
export NCCL_P2P_DISABLE=0
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_DEBUG=WARN
export DMTCP_COORD_HOST="127.0.0.1"
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT=29500
export WORLD_SIZE=2
export PYTHONUNBUFFERED=1
PORT=7783

TEST_SCRIPT="${1:-pytorch_multicomm_test.py}"
echo "==> Selected PyTorch Test: ${TEST_SCRIPT}"

echo "==> Cleaning old checkpoint files"
$DMTCP_CMD --coord-port $PORT --quit --quiet 2>/dev/null || true
rm -rf ckpt_*.dmtcp ckpt_rank_* ckpt_*_files/ dmtcp_restart_script*.sh rank_*.log

echo "==> Launching Coordinator in daemon mode"
${DMTCP_ROOT}/bin/dmtcp_coordinator --daemon -p $PORT
sleep 1

echo "==> Phase 1: Initial PyTorch run with checkpoint"
declare -a PIDS=()

for rank in 0 1; do
    echo "    Starting Rank ${rank}..."
    RANK=${rank} LOCAL_RANK=${rank} ${DMTCP_LAUNCH} -j --coord-port $PORT --interval 0 --with-plugin $PLUGIN \
        python3 "${TEST_SCRIPT}" --epochs -1 > "rank_${rank}.log" 2>&1 &
    PIDS+=($!)
done

echo "Waiting for PyTorch processes to connect to coordinator..."
for i in {1..30}; do
    PEERS=$(${DMTCP_CMD} --coord-port $PORT --status 2>/dev/null | grep NUM_PEERS | awk -F= '{print $2}' || echo "0")
    if [ "$PEERS" != "" ] && [ "$PEERS" -ge 2 ]; then
        echo "Found $PEERS connected processes."
        break
    fi
    sleep 1
done

echo "Letting PyTorch train for 20 seconds..."
sleep 20

echo "Current Rank 0 Log:"
cat rank_0.log | tail -n 15 || true

echo "Triggering checkpoint..."
${DMTCP_CMD} --coord-port $PORT --checkpoint

# Wait for checkpoint image creation
echo "Waiting for checkpoint files to appear..."
for i in {1..20}; do
    if ls ckpt_*.dmtcp >/dev/null 2>&1; then
        echo "Checkpoint images written."
        break
    fi
    sleep 1
done

echo "Killing initial run..."
${DMTCP_CMD} --coord-port $PORT --quit --quiet 2>/dev/null || true
for pid in "${PIDS[@]}"; do
    kill -9 "$pid" 2>/dev/null || true
done
sleep 2

echo "==> Checkpoint complete. Verifying files exist:"
ls -lh ckpt_*.dmtcp || {
    echo "ERROR: Checkpoint files were not created! Rank logs:"
    echo "--- rank_0.log ---"
    cat rank_0.log
    echo "--- rank_1.log ---"
    cat rank_1.log
    exit 1
}

echo "==> Phase 2: Restarting from checkpoint"
RESTART_SCRIPT=$(ls -t dmtcp_restart_script*.sh | head -n 1)

if [ -n "$RESTART_SCRIPT" ]; then
    echo "Restarting using $RESTART_SCRIPT"
    ./$RESTART_SCRIPT --coord-port $PORT > restart.log 2>&1 &
    REST_PID=$!
    
    echo "Letting restarted process run for 15 seconds..."
    sleep 15
    
    echo "Post-restart log output:"
    cat rank_0.log | tail -n 15 || true
    
    echo "Killing restarted process..."
    ${DMTCP_CMD} --coord-port $PORT --quit --quiet 2>/dev/null || true
    kill -9 $REST_PID 2>/dev/null || true
else
    echo "Error: No restart script found!"
    exit 1
fi

echo "==> PyTorch MANA Test Complete"
