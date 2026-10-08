#!/bin/bash
# One measurement: one benchmark, one mode, one scale, one repetition.
# Must run inside a Slurm allocation (campaign.sbatch calls it).
#
#   run_bench.sh <bench> <mode> <nodes> <gpus_per_node> <rep>
#
# bench: is | pytorch | nccl_latency | nccl_bw | nccl_bibw | nccl_allreduce |
#        nccl_allgather | nccl_bcast | nccl_reduce | nccl_reduce_scatter | nccl_alltoall
# Every rank is a plain srun task; no MPI is involved anywhere.
# mode:  native      no DMTCP, NCCL_PROFILE/NET_PROFILE  (baseline)
#        native_ref  no DMTCP, NCCL/module defaults      (production reference)
#        dmtcp       under dmtcp_launch + plugins, no checkpoint (runtime overhead)
#        ckpt_resume one blocking checkpoint mid-run, then continue to completion
#        ckpt_restart one blocking checkpoint, kill, dmtcp_restart, run to completion
#
# Writes $RUN/result.env and appends one row to $CAMPAIGN_DIR/results.csv.

set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

BENCH=$1 MODE=$2 NN=$3 GPN=$4 REP=$5
NR=$((NN * GPN))
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

case "$MODE" in
native | native_ref | dmtcp) PHASE=perf ;;
ckpt_resume | ckpt_restart) PHASE=ckpt ;;
*) echo "unknown mode $MODE" >&2; exit 2 ;;
esac

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
pytorch)
    load_modules pytorch
    PT_ENV_DIR=$REPO_ROOT/tests/multi . "$REPO_ROOT/tests/multi/pytorch_dmtcp_env.sh"
    MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
    export MASTER_ADDR MASTER_PORT=$((29500 + RANDOM % 1000))
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
    READY_RE='^# Size'
    OK_RE='validation PASSED'
    ;;
*) echo "unknown bench $BENCH" >&2; exit 2 ;;
esac
apply_nccl_profile "$prof" "$net" || exit 2

export PLUGIN DMTCP_LAUNCH DMTCP_RESTART
export COORD_HOST=$(hostname)
# -n is explicit: inside an "srun --overlap" step (interactive QOS) the
# inherited SLURM_NTASKS would otherwise cap the job step at that step's size.
SRUN=(srun -N "$NN" -n "$NR" --ntasks-per-node="$GPN" --gpus-per-node="$GPN" -c $((128 / GPN))
    --cpu-bind=cores)
LAUNCH=$REPORT_DIR/launch_rank.sh

{
    echo "BENCH=$BENCH MODE=$MODE NODES=$NN GPUS_PER_NODE=$GPN NRANKS=$NR REP=$REP"
    echo "NCCL_PROFILE=$prof NET_PROFILE=$net CKPT_STRIPE_COUNT=$CKPT_STRIPE_COUNT"
    echo "PLUGIN=$PLUGIN"
    echo "CMD=${CMD[*]}"
    echo "NODELIST=$SLURM_JOB_NODELIST JOBID=$SLURM_JOB_ID"
    env | grep -E '^(NCCL_|IS_|PT_|NB_|USE_LIBUV|CKPT_)' | sort
} >meta.env

count() { grep -aEc "$1" "${@:2}" 2>/dev/null | awk -F: '{s += $NF} END {print s + 0}'; }
maxval() { grep -ahoE "$1 [0-9.]+" "${@:2}" 2>/dev/null | awk '{v = $NF; if (v > m) m = v} END {print m + 0}'; }

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
stop_coord() { "$DMTCP_COMMAND" -h "$COORD_HOST" -p "$COORD_PORT" --quit >/dev/null 2>&1 || true; }

# wait_for <regex> <n> <pid> <timeout> <files...>: 0 when found, 1 if pid died/timeout
wait_for() {
    local re=$1 n=$2 pid=$3 to=$4 t=0
    shift 4
    while [ "$t" -lt $((to * 2)) ]; do
        [ "$(count "$re" "$@")" -ge "$n" ] && return 0
        kill -0 "$pid" 2>/dev/null || { [ "$(count "$re" "$@")" -ge "$n" ]; return $?; }
        sleep 0.5
        t=$((t + 1))
    done
    return 1
}

STATUS=ok WALL= CKPT_S= DRAIN_S= IMG_BYTES= IMG_FILES= WRITE_MBPS= RESUME_REBUILD_S=
RESTART_LAT_S= RESTART_TOTAL_S= RESTART_REBUILD_S=

T0=$(now)
case "$MODE" in
native | native_ref)
    timeout "$RUN_TIMEOUT" "${SRUN[@]}" "$LAUNCH" native "$CK" -- "${CMD[@]}" >out.log 2>&1
    rc=$?
    WALL=$(elapsed "$T0" "$(now)")
    [ $rc -ne 0 ] && STATUS=fail_rc$rc
    ;;
dmtcp)
    start_coord || exit 1
    T0=$(now)
    timeout "$RUN_TIMEOUT" "${SRUN[@]}" "$LAUNCH" dmtcp "$CK" -- "${CMD[@]}" >out.log 2>&1
    rc=$?
    WALL=$(elapsed "$T0" "$(now)")
    [ $rc -ne 0 ] && STATUS=fail_rc$rc
    stop_coord
    ;;
ckpt_resume | ckpt_restart)
    start_coord || exit 1
    T0=$(now)
    timeout "$RUN_TIMEOUT" "${SRUN[@]}" "$LAUNCH" dmtcp "$CK" -- "${CMD[@]}" >out.log 2>&1 &
    PID=$!
    if ! wait_for "$READY_RE" "$READY_N" $PID "$STARTUP_TIMEOUT" out.log; then
        STATUS=fail_startup
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
            for _ in $(seq 1 18000); do # 900 s at 50 ms
                compgen -G 'dmtcp_restart_script*.sh' >/dev/null && break
                kill -0 $PID 2>/dev/null || break
                sleep 0.05
            done
            TC1=$(now)
            compgen -G 'dmtcp_restart_script*.sh' >/dev/null || { [ $crc -eq 0 ] && crc=124; }
            CKPT_S=$(elapsed "$TC0" "$TC1")
            [ $crc -ne 0 ] && STATUS=fail_ckpt_rc$crc
            IMG_FILES=$(find "$CK" -name 'ckpt_*.dmtcp' | wc -l)
            IMG_BYTES=$(find "$CK" -name 'ckpt_*.dmtcp' -printf '%s\n' | awk '{s += $1} END {print s + 0}')
            WRITE_MBPS=$(awk -v b="$IMG_BYTES" -v t="$CKPT_S" 'BEGIN { if (t > 0) printf "%.1f", b / 1048576 / t }')
            DRAIN_S=$(maxval 'Checkpoint overhead:' out.log)
            [ "$IMG_FILES" -lt "$NR" ] && STATUS=fail_images_${IMG_FILES}_of_$NR
        fi
    fi

    if [ "$MODE" = ckpt_resume ] || [ "$STATUS" != ok ]; then
        if [ "$STATUS" = ok ]; then
            wait_for 'Rebuild overhead:' "$NR" $PID 300 out.log
            RESUME_REBUILD_S=$(maxval 'Rebuild overhead:' out.log)
        fi
        wait $PID
        rc=$?
        WALL=$(elapsed "$T0" "$(now)")
        [ "$STATUS" = ok ] && [ $rc -ne 0 ] && STATUS=fail_rc$rc
        stop_coord
    else
        # ckpt_restart: discard the running job, then restore from the images.
        "$DMTCP_COMMAND" -h "$COORD_HOST" -p "$COORD_PORT" --kill >/dev/null 2>&1
        wait $PID
        stop_coord
        before=$(count 'Rebuild overhead:' out.log)
        start_coord || exit 1
        TR0=$(now)
        timeout "$RUN_TIMEOUT" "${SRUN[@]}" "$LAUNCH" restart "$CK" >restart.log 2>&1 &
        RPID=$!
        # Restored ranks may write to restart.log (new srun's stdout) or reopen out.log.
        if wait_for 'Rebuild overhead:' $((NR + before)) $RPID 600 restart.log out.log; then
            RESTART_LAT_S=$(elapsed "$TR0" "$(now)")
        fi
        wait $RPID
        rc=$?
        TR2=$(now)
        RESTART_TOTAL_S=$(elapsed "$TR0" "$TR2")
        WALL=$(elapsed "$T0" "$TR2")
        RESTART_REBUILD_S=$(maxval 'Rebuild overhead:' restart.log)
        [ $rc -ne 0 ] && STATUS=fail_restart_rc$rc
        stop_coord
    fi
    ;;
esac

# --- Correctness and application metric -------------------------------------
LOGS=(out.log)
[ -f restart.log ] && LOGS+=(restart.log)
if [ "$STATUS" = ok ] && [ "$(count "$OK_RE" "${LOGS[@]}")" -lt "$OK_N" ]; then
    STATUS=fail_verify
fi
# Guard against a launch with the wrong number of ranks, which would otherwise
# "pass" (e.g. a 1-rank collective). Each program reports its world size.
case "$BENCH" in
is) WS_RE="Total number of processes: +$NR\$" ;;
pytorch) WS_RE="^\[pt\] rank 0/$NR process group ready" ;;
nccl_*) WS_RE="^# nccl_bench [a-z_]+: $NR ranks" ;;
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

# Images are large (GBs per rank for PyTorch); sizes are already recorded.
if [ "${KEEP_IMAGES:-0}" != 1 ]; then find "$CK" -name 'ckpt_*.dmtcp' -delete; fi

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
