#!/bin/bash
#SBATCH -A <ACCOUNT>          # ← Replace with your NERSC allocation (e.g., m1234)
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 00:10:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none
#SBATCH -J nas-is-nccl
#SBATCH -o nas_is_nccl_%j.out
#SBATCH -e nas_is_nccl_%j.err

# ─────────────────────────────────────────────────────────────────────
#  NAS IS Benchmark (NCCL port) — Perlmutter batch script
#
#  Build first:
#    module load cudatoolkit
#    cd tests/nas-benchmarks/IS
#    make is_nccl CLASS=S      # or CLASS=A, CLASS=W, etc.
#
#  Submit:
#    sbatch run_perlmutter.sh
# ─────────────────────────────────────────────────────────────────────

set -euo pipefail

module load cudatoolkit 2>/dev/null || true

# NCCL tuning for Perlmutter (single-node, NVLink)
export NCCL_DEBUG=WARN
export NCCL_NET_GDR_LEVEL=PHB

echo "============================================="
echo "  NAS IS Benchmark — NCCL port"
echo "  Nodes:  $SLURM_NNODES"
echo "  Tasks:  $SLURM_NTASKS"
echo "  GPUs:   $SLURM_GPUS_ON_NODE"
echo "  Date:   $(date)"
echo "============================================="

cd $SLURM_SUBMIT_DIR

srun ./is_nccl_exe

echo ""
echo "Job completed at $(date)"
