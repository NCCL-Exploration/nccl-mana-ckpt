#!/bin/bash
# Runs tests/ckpt_race.c over the worker modes, with and without the
# PROT_NONE reservation (-r), NPROC processes at once, one per GPU, as the
# ranks of a node are checkpointed together.  Runs as a one-node step inside
# an allocation:
#
#   salloc -A m4128 -q interactive -t 00:30:00 -N 1 -C gpu --gpus-per-node=4 --no-shell
#   srun --jobid=J --overlap -N 1 -n 1 --gpus=4 --cpu-bind=none \
#       bash -l netsplit/tests/race_matrix.sh <out dir> <cycles> <MiB> [modes] [reserve list]
#
# modes defaults to "idle sync touch both" and the reserve list to "0 1".
# Output: one log per mode, reservation and GPU, and <out dir>/summary.txt
# with one line per process (its exit status, cycles and RESULT line).
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
out=$1 cycles=$2 mib=$3
modes=${4:-"idle sync touch both"}
reserves=${5:-"0 1"}
NPROC=${NPROC:-4}
bin=$HERE/bin/ckpt_race
module unload darshan >/dev/null 2>&1
module load cudatoolkit/13.2 >/dev/null 2>&1
mkdir -p "$out"
for r in $reserves; do
    for m in $modes; do
        tag=$m-r$r
        flag=""
        [ "$r" = 1 ] && flag=-r
        pids=()
        for g in $(seq 0 $((NPROC - 1))); do
            CUDA_VISIBLE_DEVICES=$g timeout 900 "$bin" -m "$m" $flag -n "$cycles" -s "$mib" \
                >"$out/$tag.gpu$g.log" 2>&1 &
            pids+=($!)
        done
        for i in "${!pids[@]}"; do
            wait "${pids[$i]}"
            rc=$?
            res=$(grep -h '^RESULT\|^SIGSEGV' "$out/$tag.gpu$i.log" | head -1)
            ncyc=$(grep -c '^CYCLE' "$out/$tag.gpu$i.log")
            echo "$(date +%T) $tag gpu$i rc=$rc cycles=$ncyc $res" | tee -a "$out/summary.txt"
        done
    done
done
