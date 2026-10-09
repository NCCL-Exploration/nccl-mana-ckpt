#!/bin/bash
# netsplit phase 1 tests.  Runs inside an allocation, as a one-node step:
#
#   salloc -A m4128 -q interactive -t 01:00:00 -N 2 -C gpu --gpus-per-node=4 --no-shell
#   srun --jobid=J -N 1 -n 1 --overlap bash netsplit/tests/run_ns.sh
#
# CASES: "mode:bench:NNxGPN[:ENV=V,...]" separated by spaces.
#   mode   native | launch (no checkpoint) | resume | restart
#          | restart@move (launch on the job's first node(s), restart on the others)
#   bench  allreduce | bw | alltoall | latency (nccl_bench) | is (NAS IS class B)
# Each case: $OUT/<case>/{launch,restart}.log; one line in $OUT/summary.txt.

set -uo pipefail
NS=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
REPO=$(cd "$NS/.." && pwd)
DM=$REPO/dmtcp
: "${BENCH:=$SCRATCH/nccl-ckpt-report/bin/nccl_bench}"
: "${IS_BIN:=$SCRATCH/nccl-ckpt-report/bin/is_nccl_B}"
: "${T_RUN:=25}"        # seconds per run (MANA modes); native: T_NATIVE
: "${T_NATIVE:=8}"
: "${SIZES:=4194304:4194304}"  # nccl_bench -m MIN:MAX
: "${CKPT_DELAY:=6}"
: "${CKPT_TO:=150}"
: "${PROGRESS_TO:=90}"
: "${OUT:=$SCRATCH/netsplit-runs/${SLURM_JOB_ID:?run inside an allocation}${TAG:+-$TAG}}"

module unload darshan vasp >/dev/null 2>&1 || true
module load cudatoolkit/13.2 nccl/2.29.2-cu13 >/dev/null 2>&1
export DARSHAN_DISABLE=1 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET NCCL_RAS_ENABLE=0

# The driver is a one-node step: give the inner sruns the job's nodes.
JOBNODES=$(squeue -j "$SLURM_JOB_ID" -h -o %N)
mapfile -t NODES < <(scontrol show hostnames "$JOBNODES")
export SLURM_JOB_NODELIST=$JOBNODES SLURM_NODELIST=$JOBNODES SLURM_NNODES=${#NODES[@]} \
    SLURM_JOB_NUM_NODES=${#NODES[@]}
unset SLURM_NTASKS SLURM_NPROCS SLURM_TASKS_PER_NODE SLURM_STEP_NODELIST SLURM_STEP_NUM_NODES \
    SLURM_STEP_NUM_TASKS SLURM_STEP_TASKS_PER_NODE SLURM_NTASKS_PER_NODE SLURM_CPUS_PER_TASK \
    SLURM_CPU_BIND SLURM_CPU_BIND_TYPE SLURM_CPU_BIND_LIST
export DMTCP_COORD_HOST=$(hostname)
mkdir -p "$OUT"

now() { date +%s.%N; }
dt() { awk -v a="$1" -v b="$2" 'BEGIN { printf "%.1f", b - a }'; }
cnt() { cat "$@" 2>/dev/null | grep -aoE "$PAT" | wc -l; }
transports() { cat "$@" 2>/dev/null | grep -ahoE 'via (P2P|SHM|NET)/[A-Za-z0-9_/ ]*' | sort -u | tr '\n' ';'; }
first_error() {
    cat "$@" 2>/dev/null | grep -aE 'NCCL WARN|\[netsplit\]|netsplit:.*(cannot|no |WARNING)|JASSERT|Segmentation|Aborted|FAIL|error:' \
        | grep -av 'NCCL INFO' | head -n 1 | cut -c1-220
}
wait_until() { # timeout pid condition
    local to=$1 pid=$2 i
    for ((i = 0; i < to * 4; i++)); do
        eval "$3" && return 0
        kill -0 "$pid" 2>/dev/null || { eval "$3"; return $?; }
        sleep 0.25
    done
    return 1
}
start_coord() {
    export DMTCP_COORD_PORT=$((7700 + RANDOM % 2000))
    "$DM/bin/dmtcp_coordinator" --daemon -p "$DMTCP_COORD_PORT" >>coord.log 2>&1
    for _ in $(seq 20); do "$DM/bin/dmtcp_command" -s >/dev/null 2>&1 && return 0; sleep 0.5; done
    return 1
}
stop_coord() { "$DM/bin/dmtcp_command" --quit >/dev/null 2>&1 || true; }
kill_job() { "$DM/bin/dmtcp_command" --kill >/dev/null 2>&1; sleep 2; kill -TERM "$1" 2>/dev/null; sleep 1; kill -KILL "$1" 2>/dev/null; wait "$1" 2>/dev/null; }
srun_args() { # nn gpn [nodelist]
    local nn=$1 gpn=$2 w=${3:-}
    echo "-N $nn -n $((nn * gpn)) --ntasks-per-node=$gpn --gpus-per-node=4 -c $((128 / gpn)) --cpu-bind=cores --kill-on-bad-exit=1${w:+ -w $w}"
}
# How many CUDA driver instances a rank maps: 1 is right (the upper half's).
libcuda_instances() {
    local node=$1
    srun --overlap -N 1 -n 1 -w "$node" --cpu-bind=none bash -c '
        for p in $(pgrep -u '"$USER"' -f "bin/netsplit-lh"); do
            n=$(grep -E "libcuda\.so\.[0-9.]+$" /proc/$p/maps 2>/dev/null | grep r-xp | awk "{print \$1}" | sort -u | wc -l)
            echo "pid $p libcuda_text_segments=$n stub=$(grep -c cudart_stub /proc/$p/maps)"
        done' 2>/dev/null | head -4 | tr '\n' ';'
}

run_case() {
    local spec=$1 mode bench layout envs nn gpn nr name dir status=ok c= rl= extra=
    IFS=: read -r mode bench layout envs <<<"$spec"
    nn=${layout%x*} gpn=${layout#*x} nr=$((nn * gpn))
    name=$(echo "${bench}_${mode}_${layout}${envs:+_$envs}" | tr ',=@/' '-.._')
    dir=$OUT/$name
    rm -rf "$dir" && mkdir -p "$dir" && cd "$dir" || return
    # nccl_launch/nccl_restart quiet DMTCP (-q -q): its errors go here (opened O_APPEND).
    export JALIB_STDERR_PATH=$dir/dmtcp-errors.log
    local -a envv=()
    [ -n "$envs" ] && IFS=, read -ra envv <<<"$envs"
    local secs=$T_RUN
    [ "$mode" = native ] && secs=$T_NATIVE
    local -a prog
    local ok_re
    if [ "$bench" = is ]; then
        export IS_NCCL_ID_FILE=$dir/id.bin IS_ITER_SLEEP=0 IS_TARGET_SECONDS=$secs
        prog=("$IS_BIN"); PAT='^ +[0-9]+$'; ok_re='Verification += +SUCCESSFUL'
    else
        prog=("$BENCH" -b "$bench" -f "$dir/id.bin" -p 200 -m "$SIZES" -T "$secs")
        PAT='^# Size|# progress'; ok_re='validation PASSED'
    fi
    local SR LSR RSR
    SR=$(srun_args "$nn" "$gpn")
    LSR=$SR RSR=$SR
    if [ "$mode" = restart@move ]; then
        LSR=$(srun_args "$nn" "$gpn" "$(IFS=,; echo "${NODES[*]:0:$nn}")")
        RSR=$(srun_args "$nn" "$gpn" "$(IFS=,; echo "${NODES[*]:$nn:$nn}")")
    fi

    if [ "$mode" = native ]; then
        env "${envv[@]}" timeout $((secs + 180)) srun $SR "${prog[@]}" >launch.log 2>&1 || status=fail_rc$?
        grep -aqE "$ok_re" launch.log || [ $status != ok ] || status=fail_verify
        printf '%-46s %-20s %s\n' "$name" "$status" "transports: $(transports launch.log) $( [ $status != ok ] && first_error launch.log)" | tee -a "$OUT/summary.txt"
        return
    fi

    start_coord || { echo "$name: coordinator failed" | tee -a "$OUT/summary.txt"; return; }
    env "${envv[@]}" srun $LSR "$NS/bin/nccl_launch" "${prog[@]}" >launch.log 2>&1 &
    local pid=$! t0 t1
    if ! wait_until 240 $pid '[ $(cnt launch.log) -ge 1 ]'; then
        if kill -0 $pid 2>/dev/null; then status=hang_startup; kill_job $pid; else status=fail_startup; fi
    else
        extra="maps: $(libcuda_instances "${NODES[0]}")"
        if [ "$mode" = launch ]; then
            wait_until $((secs + 240)) $pid false
            kill -0 $pid 2>/dev/null && { status=hang_run; kill_job $pid; }
        else
            sleep "$CKPT_DELAY"
            rm -f dmtcp_restart_script*.sh
            t0=$(now)
            "$DM/bin/dmtcp_command" -c >ckpt_cmd.log 2>&1
            if wait_until "$CKPT_TO" $pid 'compgen -G "dmtcp_restart_script*.sh" >/dev/null'; then
                t1=$(now); c=$(dt "$t0" "$t1")
            elif kill -0 $pid 2>/dev/null; then status=hang_ckpt; kill_job $pid
            else status=fail_ckpt; fi
        fi
    fi
    if [ $status = ok ] && [ "$mode" = resume ]; then
        local p0
        p0=$(cnt launch.log)
        if ! wait_until "$PROGRESS_TO" $pid '[ $(cnt launch.log) -gt '"$p0"' ] || grep -aqE "'"$ok_re"'" launch.log'; then
            kill -0 $pid 2>/dev/null && { status=hang_after_resume; kill_job $pid; }
        fi
        [ $status = ok ] && { wait_until $((secs + 240)) $pid false; kill -0 $pid 2>/dev/null && { status=hang_after_resume; kill_job $pid; }; }
    fi
    if [ $status = ok ] && [[ $mode == restart* ]]; then
        "$DM/bin/dmtcp_command" --kill >/dev/null 2>&1
        wait $pid 2>/dev/null
        stop_coord
        sleep 1
        start_coord || { echo "$name: coordinator failed" | tee -a "$OUT/summary.txt"; return; }
        t0=$(now)
        env "${envv[@]}" srun $RSR "$NS/bin/nccl_restart" --restartdir "$dir" >restart.log 2>&1 &
        local rpid=$!
        if wait_until 300 $rpid '[ $(cnt restart.log) -ge 1 ] || grep -aqE "'"$ok_re"'" restart.log'; then
            rl=$(dt "$t0" "$(now)")
            wait_until $((secs + 300)) $rpid false
            kill -0 $rpid 2>/dev/null && { status=hang_after_restart; kill_job $rpid; }
        elif kill -0 $rpid 2>/dev/null; then status=hang_restart; kill_job $rpid
        else status=fail_restart; fi
        wait $rpid 2>/dev/null
        local rrc=$?
        [ $status = ok ] && [ $rrc -ne 0 ] && status=fail_restart_rc$rrc
        [ $status = ok ] && ! grep -aqE "$ok_re" restart.log && status=fail_verify
    else
        wait $pid 2>/dev/null
        local rc=$?
        [ $status = ok ] && [ $rc -ne 0 ] && status=fail_rc$rc
        [ $status = ok ] && ! grep -aqE "$ok_re" launch.log && status=fail_verify
    fi
    stop_coord
    local lh
    lh=$(cat launch.log restart.log 2>/dev/null | grep -aoE "netsplit: back end '[^']+' initialized in the lower half \(epoch [0-9]+\)" | sort | uniq -c | awk '{printf "%sx epoch %s; ", $1, $NF}')
    printf '%-46s %-20s ckpt=%-5s restart=%-5s %s\n' "$name" "$status" "${c:--}" "${rl:--}" \
        "transports: $(transports launch.log restart.log) lh-init: ${lh:-none} $extra $( [ $status != ok ] && first_error launch.log restart.log dmtcp-errors.log)" | tee -a "$OUT/summary.txt"
}

{
    echo "== $(date -Is) job $SLURM_JOB_ID nodes $JOBNODES netsplit $(git -C "$REPO" rev-parse --short HEAD)+wt"
    echo "== back end $(readlink -f "$NCCL_HOME/plugin/lib/libnccl-net.so")"
} | tee -a "$OUT/summary.txt"
for c in ${CASES:?set CASES}; do run_case "$c"; done
echo "results: $OUT/summary.txt"
