#!/bin/bash
#SBATCH -A m4128
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 00:10:00
#SBATCH -G 2
#SBATCH -J nas-mg-nccl
#SBATCH -o nas_mg_nccl_%j.out
#SBATCH -e nas_mg_nccl_%j.err

export SLURM_CPU_BIND="cores"

set -euo pipefail

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

# Compile benchmark with CLASS=A to match checkpoint run
echo "Compiling benchmark (CLASS=A)..."
make clean && make mg_nccl CLASS=A

echo "Starting baseline run (no checkpoint)..."
RUN_START=$(date +%s%N)

# Launch exactly like the checkpoint version (background processes) to isolate overheads
export NCCL_P2P_DISABLE=1
export NCCL_CUMEM_ENABLE=0
export NCCL_DMABUF_ENABLE=0
export NCCL_NVLS_ENABLE=0
export NCCL_SHM_DISABLE=1
export NCCL_NET_GDR_LEVEL=0
export MPICH_GPU_SUPPORT_ENABLED=0
export MG_REPEAT=3

./mg_nccl_exe 0 2 &
./mg_nccl_exe 1 2 &
wait

RUN_DONE=$(date +%s%N)
RUN_MS=$(((RUN_DONE - RUN_START) / 1000000))

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                BASELINE PROFILING SUMMARY                    ║"
echo "╠══════════════════════════════════════════════════════════════╣"
printf "║  %-40s %15s  ║\n" "Baseline execution time:" "${RUN_MS} ms"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Job completed at $(date)"
