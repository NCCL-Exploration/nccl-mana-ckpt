#!/bin/bash
#SBATCH -A p2026_0048_neu  
#SBATCH --partition=b200-batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus=3
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH -J collective_nccl_ckpt
#SBATCH -o nas_is_nccl_ckpt_%j.out
#SBATCH -e nas_is_nccl_ckpt_%j.err

export SLURM_CPU_BIND="cores"
export UCX_MEM_EVENTS=n

# Usage: sbatch test_aicr_ckpt.sh <test_binary> <marker_type> <timout_s>
TEST_BIN=$1
MARKER_TYPE=$2
TIMEOUT_S=${3:-60}
NRANKS=3

set -euo pipefail
ulimit -c unlimited

# Darshan (Perlmutter I/O profiler) wraps dup2/open/close via LD_PRELOAD.
# DMTCP also wraps these calls. The two conflict, causing infinite recursion
# in dup2 → stack overflow → SIGSEGV. Disable Darshan when using DMTCP.
export DARSHAN_DISABLE=1
module unload darshan 2>/dev/null || true

module load cuda/13.1.1 2>/dev/null || true
module load yale/StdEnv 2>/dev/null || true
module load OpenMPI/5.0.3-GCC-13.3.0 2>/dev/null || true
export NCCL_HOME=$HOME/nccl-2.29.2/build
export LD_LIBRARY_PATH=$NCCL_HOME/lib:$LD_LIBRARY_PATH
export CPATH=$NCCL_HOME/include:$CPATH

echo "=== GPU Environment (nvidia-smi) ==="
nvidia-smi || true
echo "====================================="

# Ensure NCCL and CUDA libs are on LD_LIBRARY_PATH for srun-launched processes
# export LD_LIBRARY_PATH=${NCCL_DIR}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Paths — adjust if your layout differs
SCRIPT_DIR=$SLURM_SUBMIT_DIR
ROOT_DIR=$(cd "$SCRIPT_DIR/../../.." && pwd)

DMTCP_ROOT=$ROOT_DIR/dmtcp

DMTCP_LAUNCH=$DMTCP_ROOT/bin/dmtcp_launch
DMTCP_CMD=$DMTCP_ROOT/bin/dmtcp_command
DMTCP_RESTART=$DMTCP_ROOT/bin/dmtcp_restart

NCCL_PLUGIN=$ROOT_DIR/bin/libncclplugin.so
CUDA_PLUGIN=$ROOT_DIR/dmtcp/plugin/cuda/libdmtcp_cuda.so
PLUGIN="$NCCL_PLUGIN:$CUDA_PLUGIN"

PORT=7783

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET
export NCCL_NET_GDR_LEVEL=PHB
export DMTCP_SLEEP_ON_FAILURE=1

cd $SCRIPT_DIR

# Clean up any stale coordinator or processes
$DMTCP_CMD --coord-port $PORT --quit --quiet 2>/dev/null || true
killall -9 dmtcp_coordinator 2>/dev/null || true
sleep 0.5

# Create and switch to node-local NVMe /tmp directory to eliminate I/O bottleneck
TMP_RUN_DIR="/tmp/dmtcp_job_${SLURM_JOB_ID:-local}"
rm -rf "$TMP_RUN_DIR"
mkdir -p "$TMP_RUN_DIR"
cd "$TMP_RUN_DIR"
FLAG_FILE="$TMP_RUN_DIR/collective_ckpt_fired_${SLURM_JOB_ID}.flag"

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

# Step 2: Launch each task under dmtcp_launch via srun
export NCCL_SHM_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_LEVEL=0
export MPICH_GPU_SUPPORT_ENABLED=0
PIDS=()
for ((i=0; i<NRANKS; i++)); do
	COLLECTIVE_TEST_SCENARIO="$MARKER_TYPE" \
	COLLECTIVE_TEST_CKPT_FIRED_FLAG_FILE="$FLAG_FILE" \
	$DMTCP_LAUNCH -j --coord-port $PORT --no-gzip \
	--with-plugin $NCCL_PLUGIN:$CUDA_PLUGIN \
	"$SCRIPT_DIR/$TEST_BIN" $i $NRANKS > "pre_ckpt_rank${i}.log" 2>&1 &
	PIDS+=("$!")
done	

wait_with_timeout() {
	local description="$1"; shift
	local time_elapsed=0
	while ! "$@"; do
		sleep 0.05
		time_elapsed=$((time_elapsed + 1))
		if [ $((time_elapsed / 20)) -ge "$TIMEOUT_S" ]; then
			echo "FAIL: timed out waiting for $description"
			cat pre_ckpt_rank*.log
			exit 1
		fi
	done
}

# Wait for both processes to connect to coordinator
check_peers() {
  local PEERS=$($DMTCP_CMD --coord-port $PORT --status 2>/dev/null | grep NUM_PEERS | awk -F= '{print $2}' || echo "0")
  [ "$PEERS" != "" ] && [ "$PEERS" -ge 3 ]
}
wait_with_timeout "all ranks to connect to coordinator" check_peers

# Wait for ranks to hit marker before checkpointing
check_all_markers() {
	for ((i=0; i<NRANKS; i++)); do 
		grep -q "MARKER:$MARKER_TYPE" "pre_ckpt_rank${i}.log" || return 1
	done
}
wait_with_timeout "marker:$MARKER_TYPE on all $NRANKS ranks" check_all_markers

echo "Triggering checkpoint..."
$DMTCP_CMD --coord-port $PORT --checkpoint
echo "Checkpoint triggered. Waiting for process to complete..."
touch "$FLAG_FILE"
for pid in "${PIDS[@]}"; do wait pid || true; done

# Phase 2: Restart from checkpoint
# Restarts MUST go through "srun dmtcp_restart".
$DMTCP_ROOT/bin/dmtcp_coordinator --daemon --coord-port $PORT --exit-on-last
sleep 1
RESTART_PIDS=()
for ((i=0; i<NRANKS; i++)); do
  RESTART_SCRIPT=$(ls -t dmtcp_restart_script_${i}*.sh 2>/dev/null | head -1)
  [ -z "$RESTART_SCRIPT" ] && { echo "FAIL: no restart script for rank $i"; exit 1; }
  ./"$RESTART_SCRIPT" --coord-port $PORT > "post_ckpt_rank${i}.log" 2>&1 &
  RESTART_PIDS+=($!)
done
for pid in "${RESTART_PIDS[@]}"; do wait "$pid" || true; done

# Phase 3: Verify that all of the tests passed for each rank
ALL_PASS=1
for ((i=0; i<NRANKS; i++)); do
	grep -q "PASS" "post_ckpt_rank{$i}" || {
		echo "TEST FAIL: rank {i} did not pass post-restart."
		ALL_PASS=0
	}
done

if ["$ALL_PASS" -eq 1]; then
	echo "TEST PASSED: tested $TEST_BIN $MARKER_TYPE"
	exit 0
else
	exit 1
fi
