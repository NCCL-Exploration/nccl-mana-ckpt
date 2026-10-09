#!/bin/bash
# netsplit phase 1 tests.  Runs inside an allocation, as a one-node step:
#
#   salloc -A m4128 -q interactive -t 01:00:00 -N 2 -C gpu --gpus-per-node=4 --no-shell
#   srun --jobid=J -N 1 -n 1 --overlap bash netsplit/tests/run_ns.sh
#
# CASES: "mode:bench:NNxGPN[:ENV=V,...]" separated by spaces.
#   mode   native | launch (no checkpoint) | resume | restart
#          | restart@move (launch on the job's first node(s), restart on the others)
#          resumeN: N checkpoints in one run, each followed by progress;
#          restartN: N generations of checkpoint, kill, restart.
#          A run lasts T_RUN + (N - 1) * CYCLE_S seconds.
#   bench  allreduce | allgather | reduce_scatter | bcast | alltoall | bw | latency
#          (nccl_bench) | is (NAS IS class B) | pytorch (DDP, tests/multi/
#          pytorch_minimal_test.py; run with NS_MODULES=pytorch/2.13.0)
# Ranks come from Slurm; nothing uses MPI.
# Each case: $OUT/<case>/{launch,restart*}.log; one line in $OUT/summary.txt.

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
: "${CYCLE_S:=20}"      # extra run time per checkpoint after the first
: "${PT_WIDTH:=4096}"   # pytorch: 2 x 4096^2 fp32 parameters, ~134 MB of gradients per step
: "${PT_BATCH:=64}"
: "${CKPT_TO:=150}"
: "${PROGRESS_TO:=90}"
: "${OUT:=$SCRATCH/netsplit-runs/${SLURM_JOB_ID:?run inside an allocation}${TAG:+-$TAG}}"

module unload darshan vasp >/dev/null 2>&1 || true
# shellcheck disable=SC2086
module load ${NS_MODULES:-cudatoolkit/13.2 nccl/2.29.2-cu13} >/dev/null 2>&1
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
oks() { cat "$@" 2>/dev/null | grep -aoE "$OK_RE" | wc -l; }
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

# Checkpoints the running job (pid $1): CK gets the seconds it took, or the
# case's status says why not.
checkpoint() {
    local t0
    rm -f dmtcp_restart_script*.sh
    t0=$(now)
    "$DM/bin/dmtcp_command" -c >>ckpt_cmd.log 2>&1
    if wait_until "$CKPT_TO" "$1" 'compgen -G "dmtcp_restart_script*.sh" >/dev/null'; then
        CK=$(dt "$t0" "$(now)")
        return 0
    fi
    if kill -0 "$1" 2>/dev/null; then status=hang_ckpt; kill_job "$1"; else status=fail_ckpt; fi
    return 1
}

run_case() {
    local spec=$1 mode bench layout envs nn gpn nr name dir status=ok extra= kind n
    IFS=: read -r mode bench layout envs <<<"$spec"
    kind=${mode%%[0-9]*}
    n=${mode#"$kind"}
    n=${n:-1}
    nn=${layout%x*} gpn=${layout#*x} nr=$((nn * gpn))
    name=$(echo "${bench}_${mode}_${layout}${envs:+_$envs}" | tr ',=@/' '-.._')
    dir=$OUT/$name
    rm -rf "$dir" && mkdir -p "$dir" && cd "$dir" || return
    # nccl_launch/nccl_restart quiet DMTCP (-q -q): its errors go here (opened O_APPEND).
    export JALIB_STDERR_PATH=$dir/dmtcp-errors.log
    local -a envv=()
    [ -n "$envs" ] && IFS=, read -ra envv <<<"$envs"
    local secs=$T_RUN
    [ "$kind" = native ] && secs=$T_NATIVE
    [ "$n" -gt 1 ] && secs=$((T_RUN + (n - 1) * CYCLE_S))
    local -a prog
    OK_N=1
    case "$bench" in
    is)
        export IS_NCCL_ID_FILE=$dir/id.bin IS_ITER_SLEEP=0 IS_TARGET_SECONDS=$secs
        prog=("$IS_BIN"); PAT='^ +[0-9]+$'; OK_RE='Verification += +SUCCESSFUL'
        ;;
    pytorch)
        # A FileStore: a TCPStore's sockets do not survive a restart.  The
        # shim and USE_LIBUV=0 avoid DMTCP's fork and io_uring trouble
        # (tests/multi/pytorch_dmtcp_env.sh).
        envv+=("PYTHONPATH=$REPO/tests/multi/pyshim${PYTHONPATH:+:$PYTHONPATH}" USE_LIBUV=0
               PYTHONUNBUFFERED=1 "PT_STORE_FILE=$dir/pt_store")
        prog=("$(command -v python)" "$REPO/tests/multi/pytorch_minimal_test.py" --duration "$secs"
              --sleep 0 --width "$PT_WIDTH" --batch-size "$PT_BATCH")
        PAT='all_reduce ok'; OK_RE='PASS pytorch_minimal'; OK_N=$nr
        ;;
    *)
        prog=("$BENCH" -b "$bench" -f "$dir/id.bin" -p 200 -m "$SIZES" -T "$secs")
        PAT='^# Size|# progress'; OK_RE='validation PASSED'
        ;;
    esac
    local SR LSR RSR
    SR=$(srun_args "$nn" "$gpn")
    LSR=$SR RSR=$SR
    if [ "$kind" = restart@move ]; then
        LSR=$(srun_args "$nn" "$gpn" "$(IFS=,; echo "${NODES[*]:0:$nn}")")
        RSR=$(srun_args "$nn" "$gpn" "$(IFS=,; echo "${NODES[*]:$nn:$nn}")")
    fi

    if [ "$kind" = native ]; then
        env "${envv[@]}" timeout $((secs + 180)) srun $SR "$NS/tests/rank_env.sh" "${prog[@]}" \
            >launch.log 2>&1 || status=fail_rc$?
        [ $status = ok ] && [ "$(oks launch.log)" -lt $OK_N ] && status=fail_verify
        printf '%-46s %-20s %s\n' "$name" "$status" "transports: $(transports launch.log) $( [ $status != ok ] && first_error launch.log)" | tee -a "$OUT/summary.txt"
        return
    fi

    start_coord || { echo "$name: coordinator failed" | tee -a "$OUT/summary.txt"; return; }
    env "${envv[@]}" srun $LSR "$NS/tests/rank_env.sh" "$NS/bin/nccl_launch" "${prog[@]}" \
        >launch.log 2>&1 &
    local pid=$! log=launch.log cks= rls= i p0 t0
    if ! wait_until 300 $pid '[ $(cnt launch.log) -ge 1 ]'; then
        if kill -0 $pid 2>/dev/null; then status=hang_startup; kill_job $pid; else status=fail_startup; fi
    else
        extra="maps: $(libcuda_instances "${NODES[0]}")"
        for ((i = 1; i <= n; i++)); do
            [ "$kind" = launch ] || [ $status != ok ] && break
            sleep "$CKPT_DELAY"
            checkpoint $pid || break
            cks+="${cks:+/}$CK"
            if [ "$kind" = resume ]; then
                p0=$(cnt $log)
                if ! wait_until "$PROGRESS_TO" $pid '[ $(cnt '"$log"') -gt '"$p0"' ] || [ $(oks '"$log"') -ge '"$OK_N"' ]'; then
                    kill -0 $pid 2>/dev/null && { status=hang_after_resume; kill_job $pid; }
                fi
                continue
            fi
            # restart: kill, then restore every rank in a new lower half.
            "$DM/bin/dmtcp_command" --kill >/dev/null 2>&1
            wait $pid 2>/dev/null
            stop_coord
            sleep 1
            start_coord || { status=fail_coord; break; }
            log=restart.log
            [ $i -gt 1 ] && log=restart$i.log
            t0=$(now)
            env "${envv[@]}" srun $RSR "$NS/bin/nccl_restart" --restartdir "$dir" >$log 2>&1 &
            pid=$!
            if wait_until 300 $pid '[ $(cnt '"$log"') -ge 1 ] || [ $(oks '"$log"') -ge '"$OK_N"' ]'; then
                rls+="${rls:+/}$(dt "$t0" "$(now)")"
            elif kill -0 $pid 2>/dev/null; then status=hang_restart; kill_job $pid
            else status=fail_restart; fi
        done
    fi
    # Run to the end, then check the result.
    if [ $status = ok ]; then
        wait_until $((secs + 300)) $pid false
        if kill -0 $pid 2>/dev/null; then
            status=hang_run
            [ "$kind" != launch ] && status=hang_after_$kind
            kill_job $pid
        fi
    fi
    wait $pid 2>/dev/null
    local rc=$?
    [ $status = ok ] && [ $rc -ne 0 ] && status=fail_rc$rc
    [ $status = ok ] && [ "$(oks $log)" -lt $OK_N ] && status=fail_verify
    stop_coord
    local lh logs
    logs=$(ls launch.log restart*.log 2>/dev/null)
    # shellcheck disable=SC2086
    lh=$(cat $logs 2>/dev/null | grep -aoE "netsplit: back end '[^']+' initialized in the lower half \(epoch [0-9]+\)" | sort | uniq -c | awk '{printf "%sx epoch %s; ", $1, $NF}')
    # shellcheck disable=SC2086
    printf '%-46s %-20s ckpt=%-5s restart=%-5s %s\n' "$name" "$status" "${cks:--}" "${rls:--}" \
        "transports: $(transports $logs) lh-init: ${lh:-none} $extra $( [ $status != ok ] && first_error $logs dmtcp-errors.log)" | tee -a "$OUT/summary.txt"
}

{
    echo "== $(date -Is) job $SLURM_JOB_ID nodes $JOBNODES netsplit $(git -C "$REPO" rev-parse --short HEAD)+wt"
    echo "== back end $(readlink -f "$NCCL_HOME/plugin/lib/libnccl-net.so")"
} | tee -a "$OUT/summary.txt"
for c in ${CASES:?set CASES}; do run_case "$c"; done
echo "results: $OUT/summary.txt"
