#!/bin/bash
# Can GDRCopy pin GPU memory after a CUDA checkpoint and restore, depending on
# what GDRCopy state crossed the checkpoint (tests/gdr_pin.c, -m)?  Runs as a
# one-GPU step inside an allocation:
#
#   salloc -A m4128 -q interactive -t 00:30:00 -N 1 -C gpu --gpus-per-node=4 --no-shell
#   srun --jobid=J -N 1 -n 1 --gpus-per-node=4 --overlap bash netsplit/tests/gdr_pin.sh
#
# cuda-checkpoint checkpoints and restores the probe in place, once per mode
# (MODES, default "keep closed fresh"; "MODE+t" adds gdr_pin -t). Output:
# $OUT/summary.txt, one log per mode.

set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
REPO=$(cd "$HERE/.." && pwd)
: "${OUT:=$SCRATCH/netsplit-runs/gdrpin-${SLURM_JOB_ID:?run inside an allocation}}"
: "${MODES:=keep closed fresh}"
CKPT_TOOL=$REPO/cuda-checkpoint/bin/x86_64_Linux/cuda-checkpoint

module unload darshan vasp >/dev/null 2>&1 || true
module load cudatoolkit/13.2 >/dev/null 2>&1
export DARSHAN_DISABLE=1
mkdir -p "$OUT"
echo "== $(date -Is) job $SLURM_JOB_ID node $(hostname) driver $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1) gdrdrv $(cat /sys/module/gdrdrv/version 2>/dev/null) use_persistent_mapping=$(cat /sys/module/gdrdrv/parameters/use_persistent_mapping 2>/dev/null)" |
    tee -a "$OUT/summary.txt"

for m in $MODES; do
    d=$OUT/$m
    rm -rf "$d" && mkdir -p "$d"
    opt=()
    case "$m" in *+t) opt=(-t) ;; esac
    (
        "$HERE/bin/gdr_pin" -d "$d" -m "${m%+t}" "${opt[@]}" &
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
        echo "## $m"
        grep -h -E '^(PIN|MAP|TOK|FAIL|==|INFO|RESULT)' "$d/log" | sed 's/^/    /'
    } | tee -a "$OUT/summary.txt"
done
