#!/bin/bash
#SBATCH -A m4128
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 00:10:00
#SBATCH -G 2
#SBATCH -J nas-is-nccl
#SBATCH -o nas_is_nccl_%j.out
#SBATCH -e nas_is_nccl_%j.err

export SLURM_CPU_BIND="cores"

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

cd $SLURM_SUBMIT_DIR

srun -n 2 -c 16 ./is_nccl_exe

echo ""
echo "Job completed at $(date)"
