#!/bin/bash
# Checkpoint/restart of NCCL programs with NCCL in MANA's lower half
# (~/mana-nccl-sp, branch nccl-lower-half), using the *native* NCCL setup:
# NCCL's defaults, the module's binary libnccl (static CUDA runtime, redirected
# by the lower half's audit module), the AWS-OFI plugin.  NCCL RAS is off
# (RAS=1 keeps it); NET=socket and GDR=0 change the network setup.
#
# Runs inside an allocation, e.g.
#   salloc -A m4128 -q interactive -t 00:45:00 -N 1 -C gpu --gpus-per-node=4 \
#       srun -N 1 -n 1 --overlap bash tests/mana_lh/run_lh.sh
# Knobs: CASES ("mode:bench ...", mode native|mana_resume|mana_restart, bench
# allreduce|bw|is), NN and GPN (bw needs exactly 2 ranks), TAG, MANA, BENCH.
#
# The programs call MPI_Init/MPI_Finalize only (MANA's checkpoint waits for an
# initialized MPI); all their communication is NCCL.
# Each MANA case: mana_launch --cuda -> checkpoint -> resume to completion, or
# -> kill -> mana_restart --cuda -> run to completion; nccl_bench validates its
# data at the end. NCCL_DEBUG=INFO records the transport NCCL used.
# Results: $OUT/summary.txt and $OUT/<case>/*.log.

set -uo pipefail
: "${MANA:=$SCRATCH/mana-nccl-sp}"
: "${BENCH:=$SCRATCH/nccl-ckpt-report/bin/nccl_bench_mpi}"   # MPI_Init only: MANA needs it
: "${IS_BIN:=$SCRATCH/nccl-ckpt-report/bin/is_nccl_mpi_B}"
: "${NN:=${SLURM_JOB_NUM_NODES:-1}}"
: "${GPN:=2}"
: "${CASES:=native:allreduce native:bw native:is mana_resume:allreduce mana_restart:allreduce mana_resume:bw mana_restart:bw mana_resume:is mana_restart:is}"
: "${TARGET_S:=25}"     # checkpoint runs: one message size for this long
: "${CKPT_DELAY:=5}"
NR=$((NN * GPN))
OUT=$MANA-runs/${SLURM_JOB_ID:?run inside a Slurm allocation}-n${NN}x${GPN}${TAG:+-$TAG}
RC=$HOME/.mana-slurm-$SLURM_JOB_ID.rc

module unload darshan vasp >/dev/null 2>&1 || true
module load cudatoolkit/13.2 nccl/2.29.2-cu13 >/dev/null 2>&1
export DARSHAN_DISABLE=1 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET
# NET=socket: NCCL's own TCP transport instead of the AWS-OFI plugin.  Set here,
# after the module, which exports NCCL_NET="AWS Libfabric" over any caller's.
[ "${NET:-}" = socket ] && export NCCL_NET=Socket NCCL_NET_PLUGIN=none
# GDR=0: no GPUDirect RDMA (the module sets NCCL_NET_GDR_LEVEL=PHB).
[ "${GDR:-1}" = 0 ] && export NCCL_NET_GDR_LEVEL=LOC
# NCCL RAS keeps TCP links between ranks that DMTCP's socket plugin still sees
# even when NCCL is in the lower half, and draining them deadlocked the first
# checkpoint (2026-10-07, job 59528884). Off unless RAS=1.
[ "${RAS:-0}" = 1 ] || export NCCL_RAS_ENABLE=0
ulimit -c unlimited
mkdir -p "$OUT"
SRUN=(srun -N "$NN" -n "$NR" --ntasks-per-node="$GPN" --gpus-per-node="$GPN" -c $((128 / GPN))
    --cpu-bind=cores)

now() { date +%s.%N; }
dt() { awk -v a="$1" -v b="$2" 'BEGIN { printf "%.2f", b - a }'; }
coord() { awk '$1 == "Host:" {h = $2} $1 == "Port:" {p = $2} END {print "-h " h " -p " p}' "$RC"; }
cnt() { grep -aEc "$1" "${@:2}" 2>/dev/null | awk -F: '{s += $NF} END {print s + 0}'; }
transports() { grep -ahoE 'via (P2P|SHM|NET)[^ ]*' "$@" 2>/dev/null | sort | uniq -c | awk '{printf "%s(%s) ", $3, $1}'; }

wait_until() { # <timeout_s> <pid> <condition>
    local to=$1 pid=$2 i
    shift 2
    for ((i = 0; i < to * 4; i++)); do
        eval "$@" && return 0
        kill -0 "$pid" 2>/dev/null || { eval "$@"; return $?; }
        sleep 0.25
    done
    return 1
}

run_case() {
    local mode=${1%%:*} b=${1##*:} dir=$OUT/${1/:/_} status=ok c= rl= rt= pid rpid t0 t1
    rm -rf "$dir" && mkdir -p "$dir" && cd "$dir" || return
    local args ready_re ok_re
    if [ "$b" = is ]; then
        # IS: rank from SLURM_PROCID; id file on the shared FS; duration target.
        export IS_NCCL_ID_FILE=$dir/id.bin IS_ITER_SLEEP=0
        if [ "$mode" = native ]; then export IS_TARGET_SECONDS=10; else export IS_TARGET_SECONDS=$TARGET_S; fi
        args=("$IS_BIN") ready_re='^ +[0-9]+$' ok_re='Verification += +SUCCESSFUL'
    else
        args=("$BENCH" -b "$b" -f "$dir/id.bin" -p 200) ready_re='^# Size' ok_re='validation PASSED'
        if [ "$mode" = native ]; then
            args+=(-m 4:67108864 -T 0.2)
        else
            args+=(-m 4194304:4194304 -T "$TARGET_S")
        fi
    fi

    if [ "$mode" = native ]; then
        timeout 900 "${SRUN[@]}" "${args[@]}" >launch.log 2>&1 || status=fail_rc$?
        [ "$(cnt "$ok_re" launch.log)" -ge 1 ] || [ $status != ok ] || status=fail_verify
        echo "$1 status=$status $(grep -aE '^[0-9]+ |Mop/s total' launch.log | tail -n 1 | tr -s ' ')" \
            "transports: $(transports launch.log)" | tee -a "$OUT/summary.txt"
        return
    fi

    "$MANA/bin/mana_coordinator" -p 0 >coord.log 2>&1 || { echo "$1 coordinator failed" | tee -a "$OUT/summary.txt"; return; }
    "${SRUN[@]}" "$MANA/bin/mana_launch" --cuda "${args[@]}" >launch.log 2>&1 &
    pid=$!
    if ! wait_until 300 $pid '[ $(cnt "$ready_re" launch.log) -ge 1 ]'; then
        status=fail_startup
    else
        sleep "$CKPT_DELAY"
        rm -f dmtcp_restart_script*.sh
        t0=$(now)
        "$MANA/bin/dmtcp_command" $(coord) -c >ckpt_cmd.log 2>&1
        # dmtcp_command -bc does not block in this DMTCP: wait for the restart
        # script, which the coordinator writes once every image is on disk.
        wait_until 600 $pid 'compgen -G "dmtcp_restart_script*.sh" >/dev/null' || status=fail_ckpt
        t1=$(now)
        c=$(dt "$t0" "$t1")
    fi

    if [ "$mode" = mana_resume ] || [ $status != ok ]; then
        wait $pid
        local rc=$?
        [ $rc -ne 0 ] && [ $status = ok ] && status=fail_rc$rc
        "$MANA/bin/dmtcp_command" $(coord) --quit >/dev/null 2>&1
    else
        "$MANA/bin/dmtcp_command" $(coord) --kill >/dev/null 2>&1
        wait $pid
        "$MANA/bin/dmtcp_command" $(coord) --quit >/dev/null 2>&1
        "$MANA/bin/mana_coordinator" -p 0 >>coord.log 2>&1
        t0=$(now)
        timeout 900 "${SRUN[@]}" "$MANA/bin/mana_restart" --cuda --restartdir "$dir" >restart.log 2>&1 &
        rpid=$!
        wait_until 600 $rpid '[ $(cnt "rebuild: comm" restart.log) -ge '"$NR"' ]' && rl=$(dt "$t0" "$(now)")
        wait $rpid
        local rrc=$?
        [ $rrc -ne 0 ] && status=fail_restart_rc$rrc
        rt=$(dt "$t0" "$(now)")
        "$MANA/bin/dmtcp_command" $(coord) --quit >/dev/null 2>&1
    fi
    [ $status = ok ] && [ "$(cnt "$ok_re" launch.log restart.log)" -lt 1 ] && status=fail_verify
    grep -aE 'WARN|JASSERT|Segmentation|FAIL|error' launch.log restart.log 2>/dev/null | grep -v 'NCCL INFO' | head -n 8 >errors.txt
    echo "$1 status=$status ckpt=${c}s restart_latency=${rl}s restart_total=${rt}s" \
        "images=$(du -shc ckpt_rank_* 2>/dev/null | tail -n 1 | cut -f1)" \
        "transports: $(transports launch.log restart.log)" | tee -a "$OUT/summary.txt"
}

{
    echo "== $(date -Is) job $SLURM_JOB_ID ${NN}x$GPN MANA $(git -C "$MANA" rev-parse --short HEAD)"
    echo "== libnccl: $(readlink -f "$NCCL_HOME/lib/libnccl.so.2")"
} | tee "$OUT/summary.txt"
for c in $CASES; do run_case "$c"; done
echo "results: $OUT/summary.txt"
