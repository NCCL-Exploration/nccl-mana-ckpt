#!/bin/bash
#SBATCH --partition=b200-batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus=2
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH -J collective-nccl-ckpt
#SBATCH -o collective_nccl_ckpt_%j.out
#SBATCH -e collective_nccl_ckpt_%j.err
export SLURM_CPU_BIND="cores"
export UCX_MEM_EVENTS=n

# Usage: sbatch test_aicr_ckpt.sh <test_binary> <n_cycles> <timeout_s> <num_ranks> 
TEST_BIN=$1
N_CYCLES=${2:-5}
TIMEOUT_S=${3:-60}
NRANKS=${4:-2}

set -euo pipefail
ulimit -c unlimited

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
ROOT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
DMTCP_ROOT=$ROOT_DIR/dmtcp
DMTCP_LAUNCH=$DMTCP_ROOT/bin/dmtcp_launch
DMTCP_CMD=$DMTCP_ROOT/bin/dmtcp_command
DMTCP_RESTART=$DMTCP_ROOT/bin/dmtcp_restart
NCCL_PLUGIN=$ROOT_DIR/bin/libncclplugin.so
CUDA_PLUGIN=$ROOT_DIR/dmtcp/plugin/cuda/libdmtcp_cuda.so
# Set up DMTCP + NCCL environment variables.
PORT=7783
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET
export NCCL_NET_GDR_LEVEL=PHB
export DMTCP_SLEEP_ON_FAILURE=1
export NCCL_SHM_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_LEVEL=0
export MPICH_GPU_SUPPORT_ENABLED=0
# Set up directories.
cd $SCRIPT_DIR
TMP_RUN_DIR="/tmp/dmtcp_job_${SLURM_JOB_ID:-local}"
rm -rf "$TMP_RUN_DIR"
mkdir -p "$TMP_RUN_DIR"
cd "$TMP_RUN_DIR"
RESULTS_DIR="$SCRIPT_DIR/results_${SLURM_JOB_ID}"
mkdir -p "$RESULTS_DIR"

copy_logs_back() {
        cp -v "$TMP_RUN_DIR"/*.log "$RESULTS_DIR/" 2>/dev/null || true
        cp -v "$TMP_RUN_DIR"/dmtcp_restart_script*.sh "$RESULTS_DIR/" 2>/dev/null || true
}
trap copy_logs_back EXIT

wait_with_timeout() {
        local description="$1"; shift
        local time_elapsed=0
        while ! "$@"; do
                sleep 0.05
                time_elapsed=$((time_elapsed + 1))
                if [ $((time_elapsed / 20)) -ge "$TIMEOUT_S" ]; then
                        echo "FAIL: timed out waiting for $description"
                        exit 1
                fi
        done
}

check_coordinator_running() {
  local STATUS
  STATUS=$($DMTCP_CMD --coord-port $PORT --status 2>/dev/null)
  local PEERS=$(echo "$STATUS" | grep NUM_PEERS | awk -F= '{print $2}')
  local RUNNING=$(echo "$STATUS" | grep 'RUNNING' | awk -F= '{print $2}')
  [ "$PEERS" != "" ] && [ "$PEERS" -ge "$NRANKS" ] && [ "$RUNNING" = "yes" ]
}

# Repeatedly checkpoint and restart.
for ((cycle=1; cycle<=N_CYCLES; cycle++)); do
	# Set up file flags and id files.
	echo "==> Cycle $cycle/$N_CYCLES"
	FLAG_FILE="$TMP_RUN_DIR/ckpt_fired_flag_cycle${cycle}.flag"
	ID_FILE="$TMP_RUN_DIR/collective_id_cycle${cycle}.flag"
	rm -f "$FLAG_FILE" "$ID_FILE"
	# Set up the coordinator.
	"$DMTCP_CMD" --coord-port "$PORT" --quit --quiet 2>/dev/null || true
	killall -9 dmtcp_coordinator 2>/dev/null || true
	sleep 0.5
	"$DMTCP_ROOT/bin/dmtcp_coordinator" --daemon --coord-port "$PORT" --exit-on-last
	sleep 1
	# Start up the test binary.
	for ((i=0; i<NRANKS; i++)); do
		COLLECTIVE_TEST_ID_FILE="$ID_FILE" \
		COLLECTIVE_TEST_CKPT_FIRED_FLAG_FILE="$FLAG_FILE" \
		"$DMTCP_LAUNCH" -j --coord-port "$PORT" --no-gzip \
			--with-plugin "$NCCL_PLUGIN:$CUDA_PLUGIN" \
			"$SCRIPT_DIR/$TEST_BIN" "$i" "$NRANKS" > "rank${i}_c${cycle}.log" 2>&1 &
	done
	wait_with_timeout "ranks connect (cycle $cycle)" check_coordinator_running
	wait_with_timeout "loop marker (cycle $cycle)" bash -c "grep -q 'TEST MARKER: LOOP' rank0_c${cycle}.log"
	# Checkpoint.
	"$DMTCP_CMD" --coord-port "$PORT" --bcheckpoint || { echo "FAIL: checkpoint failed on cycle $cycle"; exit 1; }
	wait_with_timeout "restart script (cycle $cycle)" bash -c 'ls -t dmtcp_restart_script*.sh >/dev/null 2>&1'
	RESTART_SCRIPT=$(ls -t dmtcp_restart_script*.sh | head -1)
	# Allow the test binary to continue.
	touch "$FLAG_FILE"
	# Kill the coordinator and the test processes.
	"$DMTCP_CMD" --coord-port "$PORT" --quit --quiet 2>/dev/null || true
	killall -9 dmtcp_coordinator 2>/dev/null || true
	sleep 0.5
	"$DMTCP_ROOT/bin/dmtcp_coordinator" --daemon --coord-port "$PORT" --exit-on-last
	sleep 1
	# Restart the test.
	./"$RESTART_SCRIPT" --coord-port "$PORT" > "restart_c${cycle}.log" 2>&1
	RC=$?
	# Check that the restart completed successfully.
	if [ $RC -ne 0 ]; then
		echo "FAIL: cycle $cycle restart: nonzero return code ($RC)."
		cat "restart_c${cycle}.log"
		exit 1
	fi
	# Check that the test files under restart pass.
	PASS_COUNT=$(grep -c "TEST FILE PASS" "restart_c${cycle}.log" || echo 0)
	if [ "$PASS_COUNT" -ne "$NRANKS" ]; then
		echo "FAIL: cycle $cycle had only $PASS_COUNT/$NRANKS ranks pass"
		cat "restart_c${cycle}.log"
		exit 1
	fi
	echo "Cycle $cycle PASSED ($PASS_COUNT/$NRANKS)."
done
echo "TEST PASSED: all $N_CYCLES cycles independently verified."
