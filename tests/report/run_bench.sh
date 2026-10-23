#!/bin/bash
# One measurement: one benchmark, one mode, one scale, one repetition.
# Must run inside a Slurm allocation (campaign.sbatch calls it).
#
#   run_bench.sh <bench> <mode> <nodes> <gpus_per_node> <rep>
#
# bench: is | pytorch | nccl_latency | nccl_bw | nccl_bibw | nccl_allreduce |
#        nccl_allgather | nccl_bcast | nccl_reduce | nccl_reduce_scatter | nccl_alltoall |
#        p2p_recv_first | p2p_recv_first_cold | p2p_send_first | p2p_chain | p2p_relay | p2p_multi |
#        p2p_create | p2p_split | p2p_destroy
#        (tests/multi/p2p_pending.cu: a checkpoint while a send or receive lacks its
#        counterpart; run them with CKPT_DELAY=1, as tests/multi/run_p2p_pending.sh does)
# Every rank is a plain srun task; no MPI is involved anywhere.
# mode:  native       no DMTCP, NCCL_PROFILE/NET_PROFILE  (baseline)
#        native_ref   no DMTCP, NCCL/module defaults      (production reference)
#        dmtcp        under dmtcp_launch + plugins, no checkpoint (runtime overhead)
#        ckpt_resume  one checkpoint mid-run, then continue to completion
#        ckpt_restart one checkpoint, kill, dmtcp_restart, run to completion
#        ns           under netsplit (netsplit/bin/nccl_launch), no checkpoint
#        ns_resume    ns, one checkpoint mid-run, then continue to completion
#        ns_restart   ns, one checkpoint, kill, nccl_restart, run to completion
# Ranks run on the job's first <nodes> nodes (srun -w), at launch and restart.
#
# Every phase has a time limit (common.sh, netsplit/TEST_PLAN.md §7). A job
# still running when one passes is killed, and the run's status names the
# phase: hang_startup, hang_ckpt, hang_after_resume, hang_restart, hang_run
# (completion), or timeout (RUN_TIMEOUT). A failed run leaves its first error
# line in first_error.txt.
# Writes $RUN/result.env and appends one row to $CAMPAIGN_DIR/results.csv.

set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

BENCH=$1 MODE=$2 NN=$3 GPN=$4 REP=$5
NR=$((NN * GPN))
info=$(mode_info "$MODE") || { echo "unknown mode $MODE" >&2; exit 2; }
read -r LAUNCHER KIND <<<"$info"
if [ "$KIND" = perf ]; then PHASE=perf; else PHASE=ckpt; fi
use_job_nodes
if [ "$NN" -gt "${#JOB_NODES[@]}" ]; then
    echo "$NN nodes asked, the job has ${#JOB_NODES[@]}" >&2
    exit 2
fi
RUN_NODES=$(IFS=,; echo "${JOB_NODES[*]:0:$NN}")
RUN=$CAMPAIGN_DIR/runs/$BENCH/n${NN}x${GPN}/$MODE/rep$REP
CK=$RUN/ckpt
rm -rf "$RUN"
mkdir -p "$CK"
cd "$RUN" || exit 1
# CORES=1: crashes leave core.* files in the run directory (GBs per GPU rank).
if [ "${CORES:-0}" = 1 ]; then ulimit -c unlimited; fi
if [ -n "$CKPT_STRIPE_COUNT" ] && command -v lfs >/dev/null; then
    lfs setstripe -c "$CKPT_STRIPE_COUNT" -S "$CKPT_STRIPE_SIZE" "$CK"
fi

prof=$NCCL_PROFILE net=$NET_PROFILE
if [ "$MODE" = native_ref ]; then prof=default net=ofi; fi

# --- Per-benchmark command, progress marker and success marker -------------
READY_RE= READY_N=1 OK_RE= OK_N=1
case "$BENCH" in
is)
    load_modules is
    export IS_NCCL_ID_FILE=$RUN/is_nccl_id.bin # shared FS: multi-node safe
    rm -f "$IS_NCCL_ID_FILE"
    export IS_ITER_SLEEP=0
    if [ $PHASE = perf ]; then export IS_TARGET_SECONDS=$IS_TARGET_S; else export IS_TARGET_SECONDS=$IS_CKPT_TARGET_S; fi
    CMD=("$BIN_DIR/is_nccl_$IS_CLASS")
    READY_RE='^ +[0-9]+$' # rank 0 prints each iteration number
    OK_RE='Verification += +SUCCESSFUL'
    ;;
cg)
    load_modules cg
    export CG_NCCL_ID_FILE=$RUN/cg_nccl_id.bin # shared FS: multi-node safe
    rm -f "$CG_NCCL_ID_FILE"
    export CG_ITER_SLEEP=0
    if [ $PHASE = perf ]; then export CG_TARGET_SECONDS=$CG_TARGET_S; else export CG_TARGET_SECONDS=$CG_CKPT_TARGET_S; fi
    CMD=("$BIN_DIR/cg_nccl_$CG_CLASS")
    READY_RE='^ +[0-9]+ +[0-9.eE+-]+' # rank 0 prints each iteration line
    OK_RE='Verification += +SUCCESSFUL'
    ;;
mg)
    load_modules mg
    export MG_NCCL_ID_FILE=$RUN/mg_nccl_id.bin # shared FS: multi-node safe
    rm -f "$MG_NCCL_ID_FILE"
    export MG_ITER_SLEEP=0
    if [ $PHASE = perf ]; then export MG_TARGET_SECONDS=$MG_TARGET_S; else export MG_TARGET_SECONDS=$MG_CKPT_TARGET_S; fi
    CMD=("$BIN_DIR/mg_nccl_$MG_CLASS")
    READY_RE='^  iter +[0-9]+$' # rank 0 prints each iteration line
    OK_RE='Verification += +SUCCESSFUL'
    ;;
pytorch)
    load_modules pytorch
    PT_ENV_DIR=$REPO_ROOT/tests/multi . "$REPO_ROOT/tests/multi/pytorch_dmtcp_env.sh"
    export MASTER_ADDR=${JOB_NODES[0]} MASTER_PORT=$((29500 + RANDOM % 1000))
    if [ "$LAUNCHER" = netsplit ]; then
        # A FileStore: the TCPStore's sockets do not survive a netsplit restart.
        export PT_STORE_FILE=$RUN/pt_store
    fi
    if [ $PHASE = perf ]; then dur=$PT_TARGET_S; else dur=$PT_CKPT_TARGET_S; fi
    CMD=("$(command -v python)" "$REPO_ROOT/tests/multi/pytorch_minimal_test.py"
        --duration "$dur" --sleep 0 --width "$PT_WIDTH" --batch-size "$PT_BATCH")
    READY_RE='all_reduce ok' READY_N=$NR
    OK_RE='PASS pytorch_minimal' OK_N=$NR
    ;;
nccl_*)
    load_modules nccl
    name=${BENCH#nccl_}
    case "$name" in
    latency | bw | bibw)
        if [ "$NR" -ne 2 ]; then echo "$BENCH needs exactly 2 ranks" >&2; exit 2; fi ;;
    esac
    CMD=("$NCCL_BENCH" -b "$name" -f "$RUN/nccl_bench_id.bin")
    if [ $PHASE = perf ]; then
        CMD+=(-m "$NB_MIN_SIZE:$NB_MAX_SIZE")
        if [ -n "$NB_ITERS" ]; then CMD+=(-i "$NB_ITERS"); else CMD+=(-T "$NB_SIZE_BUDGET_S"); fi
    else
        # bw/bibw iterations move a 64-message window, so report progress
        # proportionally less often.
        pr=$NB_PROGRESS
        case "$name" in bw | bibw) pr=$((pr / 64 + 1)) ;; esac
        CMD+=(-m "$NB_CKPT_SIZE:$NB_CKPT_SIZE" -T "$NB_CKPT_TARGET_S" -p "$pr")
    fi
    # The header starts the run; "# progress" lines (checkpoint runs) show
    # that it continues after a checkpoint.
    READY_RE='^# Size|^# progress'
    OK_RE='validation PASSED'
    ;;
p2p_*)
    load_modules nccl
    scenario=${BENCH#p2p_}
    CMD=("$P2P_PENDING" "${scenario//_/-}" "$RUN/p2p_id.bin" "$P2P_WINDOW_S")
    READY_RE='^\[p2p\] rank [0-9]+ in window' READY_N=$NR
    OK_RE='PASS p2p_pending' OK_N=$NR
    ;;
*) echo "unknown bench $BENCH" >&2; exit 2 ;;
esac
apply_nccl_profile "$prof" "$net" || exit 2
RUN_TARGET_S=$(bench_target_s "$BENCH" "$PHASE")

export PLUGIN DMTCP_LAUNCH DMTCP_RESTART NCCL_LAUNCH NCCL_RESTART
export COORD_HOST=$(hostname)
unset COORD_PORT
# nccl_launch and nccl_restart quiet DMTCP (-q -q); its errors go here.
if [ "$LAUNCHER" = netsplit ]; then export JALIB_STDERR_PATH=$RUN/dmtcp-errors.log; fi
# -n is explicit: inside an "srun --overlap" step (interactive QOS) the
# inherited SLURM_NTASKS would otherwise cap the job step at that step's size.
# -w: a restart must land on the nodes of the launch (TEST_PLAN.md §6).
# Block distribution: ranks 0..GPN-1 on the first node, and so on.
SRUN=(srun -N "$NN" -n "$NR" -w "$RUN_NODES" --ntasks-per-node="$GPN" --gpus-per-node="$GPN"
    -c $((128 / GPN)) --cpu-bind=cores --distribution=block --kill-on-bad-exit=1)
LAUNCH=$REPORT_DIR/launch_rank.sh
RLAUNCHER=restart
[ "$LAUNCHER" = netsplit ] && RLAUNCHER=netsplit-restart

{
    echo "BENCH=$BENCH MODE=$MODE NODES=$NN GPUS_PER_NODE=$GPN NRANKS=$NR REP=$REP"
    echo "PLAN=$PLAN LAUNCHER=$LAUNCHER KIND=$KIND"
    echo "NCCL_PROFILE=$prof NET_PROFILE=$net CKPT_STRIPE_COUNT=$CKPT_STRIPE_COUNT"
    echo "PLUGIN=$PLUGIN"
    echo "CMD=${CMD[*]}"
    echo "NODELIST=$SLURM_JOB_NODELIST RUN_NODES=$RUN_NODES JOBID=$SLURM_JOB_ID"
    echo "RUN_TARGET_S=$RUN_TARGET_S STARTUP_TIMEOUT=$STARTUP_TIMEOUT CKPT_TIMEOUT=$CKPT_TIMEOUT" \
        "PROGRESS_TIMEOUT=$PROGRESS_TIMEOUT RESTART_TIMEOUT=$RESTART_TIMEOUT" \
        "HANG_GRACE_S=$HANG_GRACE_S RUN_TIMEOUT=$RUN_TIMEOUT"
    if [ "$LAUNCHER" = netsplit ]; then
        echo "NETSPLIT=$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null)"
    fi
    env | grep -E '^(NCCL_|IS_|CG_|MG_|PT_|NB_|P2P_|USE_LIBUV|CKPT_|NETSPLIT_|FI_|DMTCP_PLUGINS)' | sort
} >meta.env

count() { grep -aEc "$1" "${@:2}" 2>/dev/null | awk -F: '{s += $NF} END {print s + 0}'; }
maxval() { grep -ahoE "$1 [0-9.]+" "${@:2}" 2>/dev/null | awk '{v = $NF; if (v > m) m = v} END {print m + 0}'; }
# have <regex> <n> <files...>: at least n matching lines.
have() { [ "$(count "$1" "${@:3}")" -ge "$2" ]; }

start_coord() {
    COORD_PORT=$((7700 + RANDOM % 2000))
    export COORD_PORT
    "$DMTCP_COORDINATOR" --daemon -p "$COORD_PORT" >>coord.log 2>&1
    for _ in $(seq 1 20); do
        "$DMTCP_COMMAND" -h "$COORD_HOST" -p "$COORD_PORT" -s >/dev/null 2>&1 && return 0
        sleep 0.5
    done
    echo "coordinator failed to start" >&2
    return 1
}
stop_coord() {
    [ -n "${COORD_PORT:-}" ] || return 0
    "$DMTCP_COMMAND" -h "$COORD_HOST" -p "$COORD_PORT" --quit >/dev/null 2>&1 || true
}

# wait_until <timeout> <pid> <command...>: 0 once the command succeeds, 1 if
# the timeout passes or the pid exits while the command still fails. POLL is
# the interval (s).
wait_until() {
    local pid=$2 end=$((SECONDS + $1))
    shift 2
    while [ "$SECONDS" -lt "$end" ]; do
        "$@" && return 0
        kill -0 "$pid" 2>/dev/null || { "$@"; return $?; }
        sleep "${POLL:-0.5}"
    done
    "$@"
}
# wait_for <regex> <n> <pid> <timeout> <files...>: n matching lines.
wait_for() {
    local re=$1 n=$2 pid=$3 to=$4
    shift 4
    wait_until "$to" "$pid" have "$re" "$n" "$@"
}
# wait_exit <pid> <timeout>: 0 if the pid has exited by then.
wait_exit() {
    POLL=0.05 wait_until "$2" "$1" false
    ! kill -0 "$1" 2>/dev/null
}
ckpt_written() { compgen -G 'dmtcp_restart_script*.sh' >/dev/null; }
# New progress lines since p0, or the job's success markers.
progressed() { [ "$(count "$READY_RE" "$@")" -gt "$p0" ] || have "$OK_RE" "$OK_N" "$@"; }
image_files() { find "$CK" -name 'ckpt_*.dmtcp' "$@"; }

# kill_job <pid>: ends a job step, pid being its "timeout srun ...": the
# coordinator's kill, then SIGTERM (timeout passes it to srun), then SIGKILL
# after 10 s.
kill_job() {
    local pid=$1
    if [ -n "${COORD_PORT:-}" ]; then
        "$DMTCP_COMMAND" -h "$COORD_HOST" -p "$COORD_PORT" --kill >/dev/null 2>&1
        wait_exit "$pid" 10 && { wait "$pid" 2>/dev/null; return 0; }
    fi
    kill -TERM "$pid" 2>/dev/null
    if ! wait_exit "$pid" 10; then
        pkill -KILL -P "$pid" 2>/dev/null
        kill -KILL "$pid" 2>/dev/null
    fi
    wait "$pid" 2>/dev/null
    return 0
}
# phase_failed <status if still running> <status if exited> <pid>: a phase
# did not complete in time; a job still running is killed.
phase_failed() {
    if kill -0 "$3" 2>/dev/null; then
        STATUS=$1
        kill_job "$3"
    else
        STATUS=$2
    fi
}
# rc_status <rc> <prefix>: the status for a step that exited with rc.
rc_status() { if [ "$1" -eq 124 ]; then echo timeout; else echo "$2$1"; fi; }
# hostmap <launch|restart>: where each rank ran, from launch_rank.sh.
hostmap() {
    local r
    for ((r = 0; r < NR; r++)); do
        echo "$r $(cat "$CK/hosts/$1.$r" 2>/dev/null || echo missing)"
    done
}

STATUS=ok WALL= CKPT_S= DRAIN_S= IMG_BYTES= IMG_FILES= WRITE_MBPS= RESUME_REBUILD_S=
RESTART_LAT_S= RESTART_TOTAL_S= RESTART_REBUILD_S= NOTE=

if [ "$LAUNCHER" != native ] && ! start_coord; then STATUS=fail_coord; fi
T0=$(now)
if [ "$STATUS" = ok ]; then
    timeout "$RUN_TIMEOUT" "${SRUN[@]}" "$LAUNCH" "$LAUNCHER" "$CK" -- "${CMD[@]}" >out.log 2>&1 &
    PID=$!
    if ! wait_for "$READY_RE" "$READY_N" $PID "$STARTUP_TIMEOUT" out.log; then
        phase_failed hang_startup fail_startup $PID
    elif [ "$KIND" = perf ]; then
        wait_exit $PID $((RUN_TARGET_S + HANG_GRACE_S)) || phase_failed hang_run hang_run $PID
    else
        sleep "$CKPT_DELAY"
        if ! kill -0 $PID 2>/dev/null; then
            STATUS=ckpt_missed # finished before the checkpoint; lengthen the run
        else
            # DMTCP 4.2's "-bc" does not block: the coordinator compares
            # "bc" against the one-character command "b", so blockUntilDone
            # is never set. Checkpoint with -c and wait for the restart
            # script instead -- the coordinator writes it once every image
            # is on disk ("Checkpoint complete").
            rm -f dmtcp_restart_script*.sh
            TC0=$(now)
            "$DMTCP_COMMAND" -h "$COORD_HOST" -p "$COORD_PORT" -c >ckpt_cmd.log 2>&1
            crc=$?
            if [ $crc -ne 0 ]; then
                STATUS=fail_ckpt_rc$crc
                kill_job $PID
            elif POLL=0.05 wait_until "$CKPT_TIMEOUT" $PID ckpt_written; then
                CKPT_S=$(elapsed "$TC0" "$(now)")
                IMG_FILES=$(image_files | wc -l)
                IMG_BYTES=$(image_files -printf '%s\n' | awk '{s += $1} END {print s + 0}')
                WRITE_MBPS=$(awk -v b="$IMG_BYTES" -v t="$CKPT_S" 'BEGIN { if (t > 0) printf "%.1f", b / 1048576 / t }')
                DRAIN_S=$(maxval 'Checkpoint overhead:' out.log)
                if [ "$IMG_FILES" -lt "$NR" ]; then
                    STATUS=fail_images_${IMG_FILES}_of_$NR
                    kill_job $PID
                fi
            else
                phase_failed hang_ckpt fail_ckpt $PID
            fi
        fi
    fi

    if [ "$STATUS" = ok ] && [ "$KIND" = resume ]; then
        # The job must make progress again, then finish.
        p0=$(count "$READY_RE" out.log)
        if wait_until "$PROGRESS_TIMEOUT" $PID progressed out.log; then
            wait_for 'Rebuild overhead:' "$NR" $PID 30 out.log
            RESUME_REBUILD_S=$(maxval 'Rebuild overhead:' out.log)
            wait_exit $PID $((RUN_TARGET_S + HANG_GRACE_S)) || phase_failed hang_run hang_run $PID
        else
            phase_failed hang_after_resume "$STATUS" $PID
        fi
    fi

    if [ "$STATUS" = ok ] && [ "$KIND" = restart ]; then
        # Discard the running job, then restore from the images.
        "$DMTCP_COMMAND" -h "$COORD_HOST" -p "$COORD_PORT" --kill >/dev/null 2>&1
        wait_exit $PID 60 || kill_job $PID
        wait $PID 2>/dev/null
        stop_coord
        before=$(count 'Rebuild overhead:' out.log)
        if ! start_coord; then
            STATUS=fail_coord
        else
            TR0=$(now)
            timeout "$RUN_TIMEOUT" "${SRUN[@]}" "$LAUNCH" "$RLAUNCHER" "$CK" >restart.log 2>&1 &
            RPID=$!
            # Restored ranks may write to restart.log (new srun's stdout) or reopen out.log.
            if wait_for 'Rebuild overhead:' $((NR + before)) $RPID "$RESTART_TIMEOUT" restart.log out.log; then
                RESTART_LAT_S=$(elapsed "$TR0" "$(now)")
                wait_exit $RPID $((RUN_TARGET_S + HANG_GRACE_S)) || phase_failed hang_run hang_run $RPID
            else
                phase_failed hang_restart fail_restart $RPID
            fi
            wait $RPID 2>/dev/null
            rc=$?
            TR2=$(now)
            RESTART_TOTAL_S=$(elapsed "$TR0" "$TR2")
            WALL=$(elapsed "$T0" "$TR2")
            RESTART_REBUILD_S=$(maxval 'Rebuild overhead:' restart.log)
            [ "$STATUS" = ok ] && [ $rc -ne 0 ] && STATUS=$(rc_status $rc fail_restart_rc)
            # Every rank must come back on the node and GPU it left (TEST_PLAN.md §6).
            hostmap launch >hosts_launch.txt
            hostmap restart >hosts_restart.txt
            if ! cmp -s hosts_launch.txt hosts_restart.txt; then
                NOTE="rank placement changed at restart: $(diff hosts_launch.txt hosts_restart.txt |
                    grep -m 2 '^[<>]' | tr '\n' ' ')"
                [ "$STATUS" = ok ] && STATUS=fail_hostmap
            fi
        fi
    else
        wait $PID 2>/dev/null
        rc=$?
        WALL=$(elapsed "$T0" "$(now)")
        [ "$STATUS" = ok ] && [ $rc -ne 0 ] && STATUS=$(rc_status $rc fail_rc)
    fi
fi
stop_coord

# --- Correctness and application metric -------------------------------------
LOGS=(out.log)
[ -f restart.log ] && LOGS+=(restart.log)
if [ "$STATUS" = ok ] && [ "$(count "$OK_RE" "${LOGS[@]}")" -lt "$OK_N" ]; then
    STATUS=fail_verify
fi
# Guard against a launch with the wrong number of ranks, which would otherwise
# "pass" (e.g. a 1-rank collective). Each program reports its world size.
case "$BENCH" in
is | mg) WS_RE="Total number of processes: +$NR\$" ;;
cg) WS_RE="Number of active processes: +$NR\$" ;;
pytorch) WS_RE="^\[pt\] rank 0/$NR process group ready" ;;
nccl_*) WS_RE="^# nccl_bench [a-z_]+: $NR ranks" ;;
p2p_*) WS_RE="^# p2p_pending [a-z-]+: $NR ranks" ;;
esac
if [ "$STATUS" = ok ] && [ "$(count "$WS_RE" "${LOGS[@]}")" -lt 1 ]; then
    STATUS=fail_worldsize
fi
METRIC_NAME= METRIC=
case "$BENCH" in
is)
    # Mop/s, not "Time in seconds": with IS_TARGET_SECONDS the time is the
    # target by construction, while Mop/s is normalized by the work done.
    METRIC_NAME=is_mops
    METRIC=$(grep -ahE 'Mop/s total' "${LOGS[@]}" | tail -n 1 | awk '{print $NF}')
    {
        echo "IS_TIME_S=$(grep -ahE 'Time in seconds' "${LOGS[@]}" | tail -n 1 | awk '{print $NF}')"
        echo "IS_PASSES=$(grep -ahoE -- '-> [0-9]+ passes' "${LOGS[@]}" | tail -n 1 | awk '{print $2}')"
    } >>meta.env
    ;;
cg)
    METRIC_NAME=cg_mops
    METRIC=$(grep -ahE 'Mop/s total' "${LOGS[@]}" | tail -n 1 | awk '{print $NF}')
    {
        echo "CG_TIME_S=$(grep -ahE 'Time in seconds' "${LOGS[@]}" | tail -n 1 | awk '{print $NF}')"
        echo "CG_PASSES=$(grep -ahoE -- '-> [0-9]+ passes' "${LOGS[@]}" | tail -n 1 | awk '{print $2}')"
    } >>meta.env
    ;;
mg)
    METRIC_NAME=mg_mops
    METRIC=$(grep -ahE 'Mop/s total' "${LOGS[@]}" | tail -n 1 | awk '{print $NF}')
    {
        echo "MG_TIME_S=$(grep -ahE 'Time in seconds' "${LOGS[@]}" | tail -n 1 | awk '{print $NF}')"
        echo "MG_PASSES=$(grep -ahoE -- '-> [0-9]+ passes' "${LOGS[@]}" | tail -n 1 | awk '{print $2}')"
    } >>meta.env
    ;;
pytorch)
    # Slowest rank's throughput (the job runs at that rate).
    METRIC_NAME=pt_steps_per_s
    METRIC=$(grep -ahoE '\(([0-9.]+) steps/s\)' "${LOGS[@]}" | tr -d '()' |
        awk 'NR == 1 || $1 < m {m = $1} END {if (NR) print m}')
    ;;
nccl_*)
    # Full per-size table for the report; headline metric = largest size.
    grep -ahE '^[0-9]+ +[0-9.]+' "${LOGS[@]}" | awk -v OFS=, '{print $1, $2}' >sizes.csv
    case "$BENCH" in
    nccl_bw | nccl_bibw) METRIC_NAME=MBps_at_max ;;
    *) METRIC_NAME=us_at_max ;;
    esac
    METRIC=$(tail -n 1 sizes.csv | cut -d, -f2)
    ;;
esac

# The first error line of a failed run, for the report's list of failures.
if [ "$STATUS" != ok ]; then
    {
        [ -n "$NOTE" ] && echo "$NOTE"
        cat "${LOGS[@]}" dmtcp-errors.log 2>/dev/null | grep -av 'NCCL INFO' |
            grep -aE 'NCCL WARN|JASSERT|Segmentation|Aborted|FAIL|[Ee]rror|netsplit:.*(cannot|WARNING)|launch_rank:|nccl_(launch|restart):' |
            head -n 1
    } | head -n 1 | cut -c1-240 >first_error.txt
fi

# Images are large (GBs per rank for PyTorch); sizes are already recorded.
if [ "${KEEP_IMAGES:-0}" != 1 ]; then image_files -delete; fi

FIELDS=(campaign bench mode nodes gpus_per_node nranks rep status wall_s metric_name metric
    ckpt_s nccl_drain_s image_files image_bytes write_MBps resume_rebuild_s
    restart_latency_s restart_total_s restart_rebuild_s nccl_profile net_profile run_dir)
VALUES=("$CAMPAIGN" "$BENCH" "$MODE" "$NN" "$GPN" "$NR" "$REP" "$STATUS" "$WALL" "$METRIC_NAME"
    "$METRIC" "$CKPT_S" "$DRAIN_S" "$IMG_FILES" "$IMG_BYTES" "$WRITE_MBPS" "$RESUME_REBUILD_S"
    "$RESTART_LAT_S" "$RESTART_TOTAL_S" "$RESTART_REBUILD_S" "$prof" "$net" "$RUN")
for i in "${!FIELDS[@]}"; do echo "${FIELDS[$i]}=${VALUES[$i]}"; done >result.env

CSV=$CAMPAIGN_DIR/results.csv
(
    flock 9
    [ -s "$CSV" ] || (IFS=,; echo "${FIELDS[*]}") >"$CSV"
    (IFS=,; echo "${VALUES[*]}") >>"$CSV"
) 9>"$CSV.lock"

echo "[$BENCH $MODE ${NN}x$GPN rep$REP] status=$STATUS wall=${WALL}s $METRIC_NAME=$METRIC" \
    "${CKPT_S:+ckpt=${CKPT_S}s image=${IMG_BYTES}B}" \
    "${RESTART_TOTAL_S:+restart_latency=${RESTART_LAT_S}s restart_total=${RESTART_TOTAL_S}s}"
[ "$STATUS" = ok ]
