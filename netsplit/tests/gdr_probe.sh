#!/bin/bash
# Which GPU memory can the NIC still use after a CUDA checkpoint and restore
# (tests/gdr_probe.c)?  Runs as a one-GPU step inside an allocation:
#
#   salloc -A m4128 -q interactive -t 01:00:00 -N 1 -C gpu --gpus-per-node=4 --no-shell
#   srun --jobid=J -N 1 -n 1 --gpus-per-node=4 --overlap bash netsplit/tests/gdr_probe.sh
#
# Two cases: libfabric with its CUDA support (FI_HMEM default), and without it
# (FI_HMEM=system, as in netsplit's lower half).  In each, cuda-checkpoint
# checkpoints and restores the probe in place; its helper is left alone.
# Output: $OUT/summary.txt and one log per case.

set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
REPO=$(cd "$HERE/.." && pwd)
: "${OUT:=$SCRATCH/netsplit-runs/gdrprobe-${SLURM_JOB_ID:?run inside an allocation}}"
CKPT_TOOL=$REPO/cuda-checkpoint/bin/x86_64_Linux/cuda-checkpoint

module unload darshan vasp >/dev/null 2>&1 || true
module load cudatoolkit/13.2 >/dev/null 2>&1
export DARSHAN_DISABLE=1 FI_LOG_LEVEL=warn FI_MR_CACHE_MAX_COUNT=0
mkdir -p "$OUT"

run() { # name [ENV=VALUE ...]
    local name=$1 d=$OUT/$1
    shift
    rm -rf "$d" && mkdir -p "$d"
    (
        export "$@"
        "$HERE/bin/gdr_probe" -d "$d" &
        pid=$!
        for _ in $(seq 600); do [ -f "$d/ready" ] && break; sleep 0.1; done
        p=$(cat "$d/ready")
        echo "== checkpoint pid $p: $("$CKPT_TOOL" --toggle --pid "$p" 2>&1) state=$("$CKPT_TOOL" --get-state --pid "$p" 2>&1)"
        sleep 2
        echo "== restore pid $p: $("$CKPT_TOOL" --toggle --pid "$p" 2>&1) state=$("$CKPT_TOOL" --get-state --pid "$p" 2>&1)"
        touch "$d/restored"
        wait $pid
        echo "== exit $?"
    ) >"$d/log" 2>&1
    {
        echo "## $name ($*)"
        grep -h -E '^(REG|XFER|FAIL|==|INFO)' "$d/log" | sed 's/^/    /'
    } | tee -a "$OUT/summary.txt"
}

echo "== $(date -Is) job $SLURM_JOB_ID node $(hostname) driver $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)" \
    | tee -a "$OUT/summary.txt"
# ONLY=name runs one case.
case=(
    "fi-hmem-cuda FI_HMEM_CUDA_USE_DMABUF=1"
    "fi-hmem-cuda-nogdrcopy FI_HMEM_CUDA_USE_DMABUF=1 FI_HMEM_CUDA_USE_GDRCOPY=0"
    "fi-hmem-system FI_HMEM=system"
    "fi-hmem-cuda-nodmabuf-nogdrcopy FI_HMEM_CUDA_USE_DMABUF=0 FI_HMEM_CUDA_USE_GDRCOPY=0"
)
for c in "${case[@]}"; do
    # shellcheck disable=SC2086
    [ -z "${ONLY:-}" ] || [ "${c%% *}" = "$ONLY" ] && run $c
done
