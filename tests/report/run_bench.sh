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
#        ns_resumeN   ns, N checkpoints, each followed by progress (also ckpt_resumeN)
#        ns_restartN  ns, N generations of checkpoint, kill, restart (also ckpt_restartN)
# Ranks run on the job's first <nodes> nodes (srun -w), at launch and restart.
# Each checkpoint is taken CKPT_DELAY s after the program reports progress
# (first at launch, then again after each resume or restart). With
# DELAY_IN_RUN_DIR=1 (campaign.sbatch) a checkpoint run's directory names its
# delay: .../<mode>/d<CKPT_DELAY>/rep<rep>.
#
# Every phase has a time limit (common.sh, netsplit/TEST_PLAN.md §7). A job
# still running when one passes is killed, and the run's status names the
# phase: hang_startup, hang_ckpt, hang_after_resume, hang_restart,
# hang_after_restart (no progress before the next checkpoint), hang_run
# (completion), or timeout (RUN_TIMEOUT). A failed run leaves its first error
# line in first_error.txt.
# Writes $RUN/result.env and appends one row to $CAMPAIGN_DIR/results.csv;
# a checkpoint run also writes one row per checkpoint to $RUN/ckpts.csv and
# $CAMPAIGN_DIR/ckpts.csv. In results.csv, a run with several checkpoints
# reports their means.

set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

BENCH=$1 MODE=$2 NN=$3 GPN=$4 REP=$5
NR=$((NN * GPN))
info=$(mode_info "$MODE") || { echo "unknown mode $MODE" >&2; exit 2; }
read -r LAUNCHER KIND NCKPT <<<"$info"
if [ "$KIND" = perf ]; then PHASE=perf; else PHASE=ckpt; fi
# How long the program is asked to run; longer for every extra checkpoint.
if [ $PHASE = perf ]; then
    RUN_TARGET_S=$(bench_target_s "$BENCH" perf)
else
    RUN_TARGET_S=$(run_target_s "$BENCH" "$NCKPT")
fi
use_job_nodes
if [ "$NN" -gt "${#JOB_NODES[@]}" ]; then
    echo "$NN nodes asked, the job has ${#JOB_NODES[@]}" >&2
    exit 2
fi
RUN_NODES=$(IFS=,; echo "${JOB_NODES[*]:0:$NN}")
RUN=$CAMPAIGN_DIR/runs/$BENCH/n${NN}x${GPN}/$MODE
if [ $PHASE = ckpt ] && [ "${DELAY_IN_RUN_DIR:-0}" = 1 ]; then RUN+=/d$CKPT_DELAY; fi
RUN+=/rep$REP
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
    export IS_TARGET_SECONDS=$RUN_TARGET_S
    CMD=("$BIN_DIR/is_nccl_$IS_CLASS")
    READY_RE='^ +[0-9]+$' # rank 0 prints each iteration number
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
    CMD=("$(command -v python)" "$REPO_ROOT/tests/multi/pytorch_minimal_test.py"
        --duration "$RUN_TARGET_S" --sleep 0 --width "$PT_WIDTH" --batch-size "$PT_BATCH")
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
        CMD+=(-m "$NB_CKPT_SIZE:$NB_CKPT_SIZE" -T "$RUN_TARGET_S" -p "$pr")
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
    echo "PLAN=$PLAN LAUNCHER=$LAUNCHER KIND=$KIND NCKPT=$NCKPT CKPT_DELAY=$CKPT_DELAY"
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
    env | grep -E '^(NCCL_|IS_|PT_|NB_|P2P_|USE_LIBUV|CKPT_|NETSPLIT_|FI_|DMTCP_PLUGINS)' | sort
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

# --- Per-checkpoint measurements --------------------------------------------
# Logs grow during a run, and several checkpoints write to the same files, so
# each measurement reads only what a log gained since a mark.
LOGS_NOW=(out.log)
declare -A OFF
mark() {
    local f
    OFF=()
    for f in "${LOGS_NOW[@]}"; do OFF[$f]=$(stat -c %s "$f" 2>/dev/null || echo 0); done
}
since() {
    local f
    for f in "${LOGS_NOW[@]}"; do tail -c +$((${OFF[$f]:-0} + 1)) "$f" 2>/dev/null; done
}
# have_since <regex> <n>: at least n matching lines since the mark.
have_since() { since >.since; have "$1" "$2" .since; }
# progressed_now: new progress lines since p0 (set by the caller), in any log.
progressed_now() { progressed "${LOGS_NOW[@]}"; }
# Per checkpoint (in order): CK_S, CK_NCCL, CK_FILES, CK_BYTES, CK_ROUNDS,
# CK_DRAINED; RB_S (rebuild after a resume); RL_S, RR_S (restart latency and
# rebuild).
CK_S=() CK_NCCL=() CK_FILES=() CK_BYTES=() CK_ROUNDS=() CK_DRAINED=() RB_S=() RL_S=() RR_S=()
RESTARTS=0 ERR_OFF=0

# take_checkpoint <pid>: waits CKPT_DELAY, checkpoints and records it. On a
# failure it sets STATUS (killing the job if it still runs) and returns 1.
take_checkpoint() {
    local pid=$1 crc t0 cs files bytes
    sleep "$CKPT_DELAY"
    if ! kill -0 "$pid" 2>/dev/null; then
        STATUS=ckpt_missed # finished before the checkpoint; lengthen the run
        return 1
    fi
    mark
    # DMTCP 4.2's "-bc" does not block: the coordinator compares "bc"
    # against the one-character command "b", so blockUntilDone is never set.
    # Checkpoint with -c and wait for the restart script instead -- the
    # coordinator writes it once every image is on disk ("Checkpoint
    # complete").
    rm -f dmtcp_restart_script*.sh
    # Lustre keeps mtimes to the second: date the mark 2 s back, so that an
    # image finished within a second of it still counts as newer. The
    # previous checkpoint ended at least CKPT_DELAY earlier.
    touch -d "@$(($(date +%s) - 2))" .ckpt_mark
    t0=$(now)
    "$DMTCP_COMMAND" -h "$COORD_HOST" -p "$COORD_PORT" -c >>ckpt_cmd.log 2>&1
    crc=$?
    if [ $crc -ne 0 ]; then
        STATUS=fail_ckpt_rc$crc
        kill_job "$pid"
        return 1
    fi
    if ! POLL=0.05 wait_until "$CKPT_TIMEOUT" "$pid" ckpt_written; then
        phase_failed hang_ckpt fail_ckpt "$pid"
        return 1
    fi
    cs=$(elapsed "$t0" "$(now)")
    # Only this checkpoint's images count; older ones are removed, so that a
    # restart finds exactly one image per rank.
    files=$(image_files -newer .ckpt_mark | wc -l)
    bytes=$(image_files -newer .ckpt_mark -printf '%s\n' | awk '{s += $1} END {print s + 0}')
    image_files ! -newer .ckpt_mark -delete
    since >.since
    CK_S+=("$cs") CK_FILES+=("$files") CK_BYTES+=("$bytes")
    CK_NCCL+=("$(maxval 'Checkpoint overhead:' .since)")
    CK_ROUNDS+=("$(grep -aoE 'converged in [0-9]+ round' .since | awk '$3 > m {m = $3} END {print m + 0}')")
    CK_DRAINED+=("$(count '\[NCCL Drain\]' .since)")
    if [ "$files" -lt "$NR" ]; then
        STATUS=fail_images_${files}_of_$NR
        kill_job "$pid"
        return 1
    fi
}

# restart_job <pid> <generation>: kills the checkpointed job, restores it
# from the images on the same nodes, and waits until every rank has rebuilt
# its communicators. Sets PID to the restored job.
restart_job() {
    local pid=$1 g=$2 log=restart.log t0
    [ "$g" -gt 1 ] && log=restart$g.log
    "$DMTCP_COMMAND" -h "$COORD_HOST" -p "$COORD_PORT" --kill >/dev/null 2>&1
    wait_exit "$pid" 60 || kill_job "$pid"
    wait "$pid" 2>/dev/null
    stop_coord
    start_coord || { STATUS=fail_coord; return 1; }
    # The killed job's errors (its kvdbRequest JASSERT, exit code 99) are
    # expected; a failed restart's first error comes from what follows.
    [ $RESTARTS -eq 0 ] && ERR_OFF=$(stat -c %s dmtcp-errors.log 2>/dev/null || echo 0)
    RESTARTS=$((RESTARTS + 1))
    : >"$log"
    LOGS_NOW+=("$log")
    mark
    t0=$(now)
    TR0=$t0
    timeout "$RUN_TIMEOUT" "${SRUN[@]}" "$LAUNCH" "$RLAUNCHER" "$CK" >"$log" 2>&1 &
    PID=$!
    # Restored ranks may write to the new srun's stdout or reopen out.log.
    if ! wait_until "$RESTART_TIMEOUT" $PID have_since 'Rebuild overhead:' "$NR"; then
        phase_failed hang_restart fail_restart $PID
        return 1
    fi
    RL_S+=("$(elapsed "$t0" "$(now)")")
    since >.since
    RR_S+=("$(maxval 'Rebuild overhead:' .since)")
    # Every rank must come back on the node and GPU it left (TEST_PLAN.md §6).
    [ -s hosts_launch.txt ] || hostmap launch >hosts_launch.txt
    hostmap restart >hosts_restart.txt
    if ! cmp -s hosts_launch.txt hosts_restart.txt; then
        NOTE="rank placement changed at restart $g: $(diff hosts_launch.txt hosts_restart.txt |
            grep -m 2 '^[<>]' | tr '\n' ' ')"
        STATUS=fail_hostmap
        kill_job $PID
        return 1
    fi
}

STATUS=ok WALL= RESTART_TOTAL_S= NOTE= TR0=

if [ "$LAUNCHER" != native ] && ! start_coord; then STATUS=fail_coord; fi
T0=$(now)
if [ "$STATUS" = ok ]; then
    timeout "$RUN_TIMEOUT" "${SRUN[@]}" "$LAUNCH" "$LAUNCHER" "$CK" -- "${CMD[@]}" >out.log 2>&1 &
    PID=$!
    if ! wait_for "$READY_RE" "$READY_N" $PID "$STARTUP_TIMEOUT" out.log; then
        phase_failed hang_startup fail_startup $PID
    elif [ "$KIND" != perf ]; then
        for ((i = 1; i <= NCKPT; i++)); do
            take_checkpoint $PID || break
            if [ "$KIND" = resume ]; then
                # The job must make progress again; by then its ranks have rebuilt.
                p0=$(count "$READY_RE" "${LOGS_NOW[@]}")
                if ! wait_until "$PROGRESS_TIMEOUT" $PID progressed_now; then
                    phase_failed hang_after_resume "$STATUS" $PID
                    break
                fi
                wait_until 30 $PID have_since 'Rebuild overhead:' "$NR"
                since >.since
                RB_S+=("$(maxval 'Rebuild overhead:' .since)")
            else
                restart_job $PID "$i" || break
                if [ "$i" -lt "$NCKPT" ]; then
                    # The next checkpoint must find the restored job running.
                    p0=$(count "$READY_RE" "${LOGS_NOW[@]}")
                    if ! wait_until "$PROGRESS_TIMEOUT" $PID progressed_now; then
                        phase_failed hang_after_restart "$STATUS" $PID
                        break
                    fi
                fi
            fi
        done
        # A run that ended before its last checkpoint did not test it.
        if [ "$STATUS" = ok ] && [ ${#CK_S[@]} -lt "$NCKPT" ]; then STATUS=ckpt_missed; fi
    fi
    # Run to completion.
    if [ "$STATUS" = ok ]; then
        wait_exit $PID $((RUN_TARGET_S + HANG_GRACE_S)) || phase_failed hang_run hang_run $PID
    fi
    wait $PID 2>/dev/null
    rc=$?
    T2=$(now)
    WALL=$(elapsed "$T0" "$T2")
    if [ $RESTARTS -gt 0 ]; then
        RESTART_TOTAL_S=$(elapsed "$TR0" "$T2")
        [ "$STATUS" = ok ] && [ $rc -ne 0 ] && STATUS=$(rc_status $rc fail_restart_rc)
    else
        [ "$STATUS" = ok ] && [ $rc -ne 0 ] && STATUS=$(rc_status $rc fail_rc)
    fi
fi
stop_coord

# Means over the run's checkpoints, for results.csv (one checkpoint: its values).
mean() { printf '%s\n' "$@" | awk 'NF {s += $1; n++} END {if (n) printf "%.6g", s / n}'; }
CKPT_S=$(mean "${CK_S[@]}") DRAIN_S=$(mean "${CK_NCCL[@]}")
IMG_FILES=$(mean "${CK_FILES[@]}") IMG_BYTES=$(mean "${CK_BYTES[@]}")
WRITE_MBPS=$(awk -v b="$IMG_BYTES" -v t="$CKPT_S" 'BEGIN { if (t > 0) printf "%.1f", b / 1048576 / t }')
RESUME_REBUILD_S=$(mean "${RB_S[@]}") RESTART_LAT_S=$(mean "${RL_S[@]}")
RESTART_REBUILD_S=$(mean "${RR_S[@]}")
ROUNDS_MAX=$(printf '%s\n' "${CK_ROUNDS[@]}" | awk 'NF && $1 > m {m = $1} END {if (m != "") print m + 0}')
DRAINED=$(printf '%s\n' "${CK_DRAINED[@]}" | awk 'NF {s += $1; n++} END {if (n) print s + 0}')
DELAY=
[ $PHASE = ckpt ] && DELAY=$CKPT_DELAY

# --- Correctness and application metric -------------------------------------
LOGS=("${LOGS_NOW[@]}")
if [ "$STATUS" = ok ] && [ "$(count "$OK_RE" "${LOGS[@]}")" -lt "$OK_N" ]; then
    STATUS=fail_verify
fi
# Guard against a launch with the wrong number of ranks, which would otherwise
# "pass" (e.g. a 1-rank collective). Each program reports its world size.
case "$BENCH" in
is) WS_RE="Total number of processes: +$NR\$" ;;
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
first_error_line() {
    grep -av 'NCCL INFO' | grep -av 'lib wrapper not initialized' |
        grep -aE 'NCCL WARN|JASSERT|Segmentation|Aborted|FAIL|[Ee]rror|netsplit:.*(cannot|WARNING)|launch_rank:|nccl_(launch|restart):' |
        head -n 1
}
if [ "$STATUS" != ok ]; then
    {
        [ -n "$NOTE" ] && echo "$NOTE"
        if [ $RESTARTS -gt 0 ]; then
            # A failure after a restart: DMTCP's errors since the first
            # restart (a JASSERT names the cause; srun's "exit code 99" line in
            # the restart log does not), then the restarts' own logs, newest
            # first. The killed job's kvdbRequest JASSERT (exit code 99) is
            # expected (TEST_PLAN.md §8).
            tail -c +$((ERR_OFF + 1)) dmtcp-errors.log 2>/dev/null | first_error_line
            for ((g = RESTARTS; g >= 1; g--)); do
                f=restart.log
                [ $g -gt 1 ] && f=restart$g.log
                first_error_line <"$f"
            done
        fi
        cat "${LOGS[@]}" dmtcp-errors.log 2>/dev/null | grep -av 'kvdbRequest' | first_error_line
    } 2>/dev/null | head -n 1 | cut -c1-240 >first_error.txt
fi

# One row per checkpoint, in the run directory and campaign-wide.
if [ ${#CK_S[@]} -gt 0 ]; then
    CKF=(campaign bench mode nodes gpus_per_node nranks rep ckpt_delay_s index ckpt_s nccl_s
        image_files image_bytes rounds drained resume_rebuild_s restart_latency_s
        restart_rebuild_s run_status run_dir)
    for ((i = 0; i < ${#CK_S[@]}; i++)); do
        echo "$CAMPAIGN,$BENCH,$MODE,$NN,$GPN,$NR,$REP,$DELAY,$((i + 1)),${CK_S[$i]},${CK_NCCL[$i]},${CK_FILES[$i]},${CK_BYTES[$i]},${CK_ROUNDS[$i]},${CK_DRAINED[$i]},${RB_S[$i]:-},${RL_S[$i]:-},${RR_S[$i]:-},$STATUS,$RUN"
    done >ckpts.rows
    { (IFS=,; echo "${CKF[*]}"); cat ckpts.rows; } >ckpts.csv
    (
        flock 9
        [ -s "$CAMPAIGN_DIR/ckpts.csv" ] || (IFS=,; echo "${CKF[*]}") >"$CAMPAIGN_DIR/ckpts.csv"
        cat ckpts.rows >>"$CAMPAIGN_DIR/ckpts.csv"
    ) 9>"$CAMPAIGN_DIR/ckpts.csv.lock"
    rm -f ckpts.rows
fi

# Images are large (GBs per rank for PyTorch); sizes are already recorded.
if [ "${KEEP_IMAGES:-0}" != 1 ]; then image_files -delete; fi

# The last six columns are new in round 2 (netsplit/TEST_PLAN.md); readers
# go by column name.
FIELDS=(campaign bench mode nodes gpus_per_node nranks rep status wall_s metric_name metric
    ckpt_s nccl_drain_s image_files image_bytes write_MBps resume_rebuild_s
    restart_latency_s restart_total_s restart_rebuild_s nccl_profile net_profile
    ckpt_delay_s n_ckpt n_restart rounds_max drained run_dir)
VALUES=("$CAMPAIGN" "$BENCH" "$MODE" "$NN" "$GPN" "$NR" "$REP" "$STATUS" "$WALL" "$METRIC_NAME"
    "$METRIC" "$CKPT_S" "$DRAIN_S" "$IMG_FILES" "$IMG_BYTES" "$WRITE_MBPS" "$RESUME_REBUILD_S"
    "$RESTART_LAT_S" "$RESTART_TOTAL_S" "$RESTART_REBUILD_S" "$prof" "$net"
    "$DELAY" "${#CK_S[@]}" "${#RL_S[@]}" "$ROUNDS_MAX" "$DRAINED" "$RUN")
for i in "${!FIELDS[@]}"; do echo "${FIELDS[$i]}=${VALUES[$i]}"; done >result.env

CSV=$CAMPAIGN_DIR/results.csv
(
    flock 9
    [ -s "$CSV" ] || (IFS=,; echo "${FIELDS[*]}") >"$CSV"
    (IFS=,; echo "${VALUES[*]}") >>"$CSV"
) 9>"$CSV.lock"

echo "[$BENCH $MODE${DELAY:+ d$DELAY} ${NN}x$GPN rep$REP] status=$STATUS wall=${WALL}s $METRIC_NAME=$METRIC" \
    "${CKPT_S:+ckpt=${CKPT_S}s (${#CK_S[@]}) image=${IMG_BYTES}B}" \
    "${RESTART_TOTAL_S:+restart_latency=${RESTART_LAT_S}s (${#RL_S[@]}) restart_total=${RESTART_TOTAL_S}s}"
[ "$STATUS" = ok ]
