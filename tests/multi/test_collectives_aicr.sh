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

# Usage: sbatch test_aicr_ckpt.sh <test_binary> <n_cycles> <nranks> <timeout_s>
TEST_BIN=$1
N_CYCLES=${2:-5}
NRANKS=${3:-2}
TIMEOUT_S=${4:-60}

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
FLAG_FILE="$TMP_RUN_DIR/collective_ckpt_fired_${SLURM_JOB_ID}.flag"
ID_FILE="$TMP_RUN_DIR/collective_id_${SLURM_JOB_ID}.bin"
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

# Start up coordinator.
"$DMTCP_CMD" --coord-port "$PORT" --quit --quiet 2>/dev/null || true
killall -9 dmtcp_coordinator 2>/dev/null || true
sleep 0.5
echo "Starting DMTCP coordinator on port $PORT..."
"$DMTCP_ROOT/bin/dmtcp_coordinator" --daemon --coord-port "$PORT" --exit-on-last
sleep 1
"$DMTCP_CMD" --coord-port "$PORT" --status >/dev/null 2>&1
# Launch each rank under DMTCP.
PIDS=()
for ((i=0; i<NRANKS; i++)); do
        COLLECTIVE_TEST_ID_FILE="$ID_FILE" \
        COLLECTIVE_TEST_CKPT_FIRED_FLAG_FILE="$FLAG_FILE" \
        "$DMTCP_LAUNCH" -j --coord-port "$PORT" --no-gzip \
        --with-plugin "$NCCL_PLUGIN:$CUDA_PLUGIN" \
        "$SCRIPT_DIR/$TEST_BIN" "$i" "$NRANKS" > "rank${i}.log" 2>&1 &
        PIDS+=("$!")
done
wait_with_timeout "all ranks to connect to coordinator" check_coordinator_running
wait_with_timeout "loop marker on rank 0" bash -c 'grep -q "TEST MARKER: LOOP" rank0.log'
# Repeatedly checkpoint and restart.
RESTART_PID=""
for ((cycle=1; cycle<=N_CYCLES; cycle++)); do
	# Checkpoint.
        echo "==> Cycle $cycle/$N_CYCLES: checkpointing..."
        if ! "$DMTCP_CMD" --coord-port "$PORT" --bcheckpoint; then
		echo "FAIL: checkpoint command failed on cycle $cycle"
		exit 1
	fi
	# Get the restart script.
	wait_with_timeout "restart script to appear" bash -c 'ls -t dmtcp_restart_script*.sh >/dev/null 2>&1'
	RESTART_SCRIPT=$(ls -t dmtcp_restart_script*.sh | head -1)
	if ! ls dmtcp_restart_script*.sh >/dev/null 2>&1; then
		echo "FAIL: checkpoint reported success but no restart script was written"
		ls -la "$TMP_RUN_DIR" >&2
		exit 1
	fi	
	# Reture the current coordinator and computation running.
	"$DMTCP_CMD" --coord-port "$PORT" --quit --quiet 2>/dev/null || true
	killall -9 dmtcp_coordinator 2>/dev/null || true
	sleep 0.5
	# Restart the coordinator and run the checkpointed processes.
        "$DMTCP_ROOT/bin/dmtcp_coordinator" --daemon --coord-port "$PORT" --exit-on-last
        sleep 1
       ./"$RESTART_SCRIPT" --coord-port "$PORT" > "cycle${cycle}.log" 2>&1 &
        RESTART_PID=$!
        wait_with_timeout "cycle $cycle to reconnect" check_coordinator_running
        echo "Cycle $cycle complete."
done
# Check the final cycle's result.
touch "$FLAG_FILE"
wait "$RESTART_PID"
RESTART_RC=$?
# Check that final cycle exited fine.
if [ $RESTART_RC -ne 0 ]; then
        echo "FAIL: final generation exited nonzero ($RESTART_RC)"
        cat "cycle${N_CYCLES}.log"
        exit 1
fi
# Check that all files passed.
PASS_COUNT=$(grep -c "TEST FILE PASS" "cycle${N_CYCLES}.log" 2>/dev/null || echo 0)
if [ "$PASS_COUNT" -eq "$NRANKS" ]; then
        echo "TEST PASSED: $TEST_BIN ($PASS_COUNT/$NRANKS ranks passed)."
        exit 0
else
        echo "TEST FAILED: $TEST_BIN (only $PASS_COUNT/$NRANKS ranks passed)."
        cat "cycle${N_CYCLES}.log"
        exit 1
fi
