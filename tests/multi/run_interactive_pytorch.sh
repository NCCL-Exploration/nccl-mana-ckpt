#!/usr/bin/env bash
# ==============================================================================
# Interactive PyTorch DMTCP / ShadowNCCL Checkpoint & Restart Runner
# ==============================================================================
# Usage:
#   ./tests/multi/run_interactive_pytorch.sh [TEST_SCRIPT] [NUM_GPUS] [PORT]
#
# Examples:
#   ./tests/multi/run_interactive_pytorch.sh
#   ./tests/multi/run_interactive_pytorch.sh pytorch_multicomm_test.py 2
#   ./tests/multi/run_interactive_pytorch.sh pytorch_ddp_test.py 4
# ==============================================================================

set -e

# Resolve repository root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Arguments with defaults
TEST_SCRIPT="${1:-pytorch_multicomm_test.py}"
NUM_GPUS="${2:-2}"
PORT="${3:-7783}"
WAIT_BEFORE_CKPT="${4:-20}"
WAIT_AFTER_RESTART="${5:-15}"

# Paths
DMTCP_ROOT="${REPO_ROOT}/dmtcp"
DMTCP_LAUNCH="${DMTCP_ROOT}/bin/dmtcp_launch"
DMTCP_COORD="${DMTCP_ROOT}/bin/dmtcp_coordinator"
DMTCP_CMD="${DMTCP_ROOT}/bin/dmtcp_command"
NCCL_PLUGIN="${REPO_ROOT}/bin/libncclplugin.so"
CUDA_PLUGIN="${DMTCP_ROOT}/plugin/cuda/libdmtcp_cuda.so"
PLUGIN="${NCCL_PLUGIN}:${CUDA_PLUGIN}"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================================================${NC}"
echo -e "${BLUE}  PyTorch Multi-Communicator Checkpoint & Restart Interactive Runner${NC}"
echo -e "${BLUE}================================================================${NC}"
echo -e "  Repo Root:           ${REPO_ROOT}"
echo -e "  Test Script:         ${SCRIPT_DIR}/${TEST_SCRIPT}"
echo -e "  Number of GPUs:      ${NUM_GPUS}"
echo -e "  Coordinator Port:    ${PORT}"
echo -e "  Pre-ckpt Wait:       ${WAIT_BEFORE_CKPT}s"
echo -e "  Post-restart Wait:   ${WAIT_AFTER_RESTART}s"
echo -e "${BLUE}================================================================${NC}"

# Verification of required binaries
if [ ! -f "${NCCL_PLUGIN}" ]; then
    echo -e "${RED}[ERROR] NCCL Plugin not found at ${NCCL_PLUGIN}.${NC}"
    echo -e "${YELLOW}        Run 'make' in repo root to build it.${NC}"
    exit 1
fi

if [ ! -f "${CUDA_PLUGIN}" ]; then
    echo -e "${RED}[ERROR] CUDA Plugin not found at ${CUDA_PLUGIN}.${NC}"
    echo -e "${YELLOW}        Run 'cd dmtcp/plugin/cuda && make libdmtcp_cuda.so' to build it.${NC}"
    exit 1
fi

if [ ! -f "${SCRIPT_DIR}/${TEST_SCRIPT}" ]; then
    echo -e "${RED}[ERROR] Test script ${SCRIPT_DIR}/${TEST_SCRIPT} does not exist.${NC}"
    exit 1
fi

# Change directory to tests/multi so checkpoints & logs stay contained
cd "${SCRIPT_DIR}"

# Cleanup previous coordinator and checkpoint artifacts
echo -e "\n${YELLOW}==> Cleaning up old coordinator and checkpoint files...${NC}"
${DMTCP_CMD} --coord-port ${PORT} --quit --quiet 2>/dev/null || true
rm -rf ckpt_*.dmtcp ckpt_rank_* ckpt_*_files/ dmtcp_restart_script*.sh

# Export environment variables for PyTorch & NCCL
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT=$((PORT + 20000))
export WORLD_SIZE=${NUM_GPUS}
export DMTCP_COORD_HOST="127.0.0.1"
export NCCL_P2P_DISABLE=0
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_DEBUG=WARN
export PYTHONUNBUFFERED=1

# 1. Launch Coordinator
echo -e "\n${GREEN}==> Launching DMTCP coordinator on port ${PORT}...${NC}"
${DMTCP_COORD} --daemon -p ${PORT}
sleep 1

# 2. Launch Ranks
echo -e "\n${GREEN}==> Launching ${NUM_GPUS} PyTorch ranks under DMTCP...${NC}"
declare -a PIDS=()

for ((rank=0; rank<NUM_GPUS; rank++)); do
    echo "    Starting Rank ${rank} (LOCAL_RANK=${rank})..."
    RANK=${rank} LOCAL_RANK=${rank} ${DMTCP_LAUNCH} -j --coord-port ${PORT} --interval 0 --with-plugin ${PLUGIN} \
        python3 "${TEST_SCRIPT}" --epochs -1 > "rank_${rank}.log" 2>&1 &
    PIDS+=($!)
done

echo -e "\n${BLUE}==> Training is running. Streaming Rank 0 log output for ${WAIT_BEFORE_CKPT}s...${NC}"
tail -f rank_0.log &
TAIL_PID=$!

sleep ${WAIT_BEFORE_CKPT}
kill ${TAIL_PID} 2>/dev/null || true

# 3. Trigger Checkpoint
echo -e "\n${YELLOW}==> Triggering checkpoint via DMTCP coordinator...${NC}"
${DMTCP_CMD} --coord-port ${PORT} --checkpoint
sleep 5

# 4. Stop Initial Execution
echo -e "\n${YELLOW}==> Stopping initial PyTorch processes...${NC}"
${DMTCP_CMD} --coord-port ${PORT} --quit --quiet 2>/dev/null || true
for pid in "${PIDS[@]}"; do
    kill -9 "${pid}" 2>/dev/null || true
done
sleep 2

# 5. Check Checkpoint Files
echo -e "\n${GREEN}==> Verifying generated checkpoint images:${NC}"
ls -lh ckpt_*.dmtcp || {
    echo -e "${RED}[ERROR] No checkpoint files were created! Check rank_*.log for errors.${NC}"
    exit 1
}

# 6. Restart from Checkpoint
RESTART_SCRIPT=$(ls -t dmtcp_restart_script*.sh 2>/dev/null | head -n 1)
if [ -z "${RESTART_SCRIPT}" ]; then
    echo -e "${RED}[ERROR] No restart script found!${NC}"
    exit 1
fi

echo -e "\n${GREEN}==> Phase 2: Restarting from ${RESTART_SCRIPT}...${NC}"
./"${RESTART_SCRIPT}" --coord-port ${PORT} &
REST_PID=$!

echo -e "\n${BLUE}==> Restarted processes running. Monitoring output for ${WAIT_AFTER_RESTART}s...${NC}"
tail -f rank_0.log &
TAIL_PID=$!

sleep ${WAIT_AFTER_RESTART}
kill ${TAIL_PID} 2>/dev/null || true

# 7. Cleanup & Completion
echo -e "\n${YELLOW}==> Shutting down restarted processes...${NC}"
${DMTCP_CMD} --coord-port ${PORT} --quit --quiet 2>/dev/null || true
kill -9 ${REST_PID} 2>/dev/null || true

echo -e "\n${GREEN}================================================================${NC}"
echo -e "${GREEN}  ✓ Checkpoint and Restart Test Completed Successfully!${NC}"
echo -e "${GREEN}================================================================${NC}"
