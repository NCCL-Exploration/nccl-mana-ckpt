#!/bin/bash
#SBATCH -A m4128
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 00:10:00
#SBATCH -J nas-is-nccl
#SBATCH -o nas_is_nccl_%j.out
#SBATCH -e nas_is_nccl_%j.err

export SLURM_CPU_BIND="cores"

set -euo pipefail

# Parse command line flags
NUM_GPUS=${SLURM_NTASKS:-${SLURM_GPUS_ON_NODE:-2}}
CLASS="A"

while [[ $# -gt 0 ]]; do
  case $1 in
    --num-gpus|-n|-g)
      NUM_GPUS="$2"
      shift 2
      ;;
    --num-gpus=*)
      NUM_GPUS="${1#*=}"
      shift 1
      ;;
    --class|-c)
      CLASS="$2"
      shift 2
      ;;
    --class=*)
      CLASS="${1#*=}"
      shift 1
      ;;
    -h|--help)
      echo "Usage: $0 [--num-gpus N] [--class {A,B,C,D}]"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: $0 [--num-gpus N] [--class {A,B,C,D}]"
      exit 1
      ;;
  esac
done

CLASS=$(echo "$CLASS" | tr 'a-z' 'A-Z')

echo "Running NAS-IS benchmark with NUM_GPUS=$NUM_GPUS, CLASS=$CLASS"

# Disable Darshan to match checkpoint run environment
export DARSHAN_DISABLE=1
module unload darshan 2>/dev/null || true

module load cudatoolkit 2>/dev/null || true
module load nccl 2>/dev/null || true

echo "=== GPU Environment (nvidia-smi) ==="
nvidia-smi || true
echo "====================================="

# Paths
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
SCRIPT_DIR=${SCRIPT_DIR//\/global\/u1//global/homes}
cd $SCRIPT_DIR

# Set NCCL ID file path for multi-node compatibility
export NCCL_ID_FILE="${SCRIPT_DIR}/.is_nccl_id_${SLURM_JOB_ID:-$$}.bin"

# Compile benchmark with requested CLASS
echo "Compiling benchmark (CLASS=$CLASS)..."
make clean && make is_nccl CLASS="$CLASS"

echo "Starting baseline run (no checkpoint)..."
RUN_START=$(date +%s%N)

export NCCL_P2P_DISABLE=1
export NCCL_CUMEM_ENABLE=0
export NCCL_DMABUF_ENABLE=0
export NCCL_NVLS_ENABLE=0
export NCCL_SHM_DISABLE=1
export NCCL_NET_GDR_LEVEL=0
export MPICH_GPU_SUPPORT_ENABLED=0

# Dynamically launch N worker processes across single and multi-node Slurm allocations
if [ -n "${SLURM_JOB_ID:-}" ] && command -v srun >/dev/null 2>&1; then
    echo "Launching $NUM_GPUS process(es) via srun across Slurm allocation..."
    srun -n "$NUM_GPUS" ./is_nccl_exe --num-gpus "$NUM_GPUS" --class "$CLASS"
else
    echo "Launching $NUM_GPUS process(es) locally in background..."
    for (( rank=0; rank<NUM_GPUS; rank++ )); do
        ./is_nccl_exe "$rank" "$NUM_GPUS" --num-gpus "$NUM_GPUS" --class "$CLASS" &
    done
    wait
fi

RUN_DONE=$(date +%s%N)
RUN_MS=$(((RUN_DONE - RUN_START) / 1000000))

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                BASELINE PROFILING SUMMARY                    ║"
echo "╠══════════════════════════════════════════════════════════════╣"
printf "║  %-40s %15s  ║\n" "Baseline execution time:" "${RUN_MS} ms"
printf "║  %-40s %15s  ║\n" "GPUs / Class:" "${NUM_GPUS} / ${CLASS}"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Job completed at $(date)"
