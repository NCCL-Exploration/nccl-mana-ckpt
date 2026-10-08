#!/bin/bash
#SBATCH -C gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -J mana-nccl
#SBATCH -o mana_nccl_%j.out
#
# Does MANA + the NCCL plugin checkpoint/restart an MPI+NCCL program, and with
# which NCCL intra-node transports (CUDA IPC in particular)?
#
#   sbatch -A m4128 tests/mana/run.sh            # or, inside salloc: tests/mana/run.sh
#   VARIANTS="p2p_cumem p2p_legacy" NR=4 sbatch -A m4128 tests/mana/run.sh
#
# Each MANA variant: launch -> checkpoint (resume) -> progress check ->
# checkpoint -> kill -> mana_restart -> run to completion, with every iteration
# of the program verified. NCCL's own log records the transport actually used.
# cuda_only is the control: same program without NCCL, MANA + CUDA plugin only.
# Results: $OUT/summary.txt (one line per variant) and $OUT/<variant>/*.log.

set -uo pipefail
: "${WORK:=$SCRATCH/mana-nccl}"
: "${NR:=2}"           # ranks = GPUs on the node
: "${ITERS:=600}"
: "${SLEEP_MS:=100}"
: "${VARIANTS:=native cuda_only no_ipc shm p2p_cumem p2p_legacy p2p_legacy_novmm}"
OUT=$WORK/runs/${SLURM_JOB_ID:?run inside a Slurm allocation}
MANA=$WORK/mana
PLUGIN=$WORK/nccl-plugin/bin/libncclplugin.so
TEST=$WORK/mana_nccl_test
RC=$HOME/.mana-slurm-$SLURM_JOB_ID.rc

module unload darshan >/dev/null 2>&1 || true
module load cudatoolkit/13.2 nccl/2.29.2-cu13 >/dev/null 2>&1
export DARSHAN_DISABLE=1 MPICH_GPU_SUPPORT_ENABLED=0
ulimit -c unlimited # a crash leaves core.* in the variant's directory
mkdir -p "$OUT"
# --cuda (MANA's CUDA forwarding + VMM allocator) when its shims are built;
# otherwise plain MANA with the DMTCP CUDA plugin loaded explicitly, which is
# enough here: MPI only touches host buffers. NCCL plugin before CUDA plugin.
if [ -e "$MANA/lib/dmtcp/cuda-forward/libcuda.so.1" ]; then
    MANA_MODE=cuda LAUNCH_FLAGS=(--cuda --with-plugin "$PLUGIN") RESTART_FLAGS=(--cuda)
else
    MANA_MODE=plain LAUNCH_FLAGS=(--with-plugin "$PLUGIN:$MANA/lib/dmtcp/libdmtcp_cuda.so") RESTART_FLAGS=()
    export DMTCP_DISABLE_PRGNAME_PREFIX=1 # the CUDA checkpoint API finds processes by name
    VARIANTS=${VARIANTS//p2p_legacy_novmm/} # the VMM allocator exists only with --cuda
fi
SRUN=(srun -N 1 -n "$NR" --gpus-per-node="$NR" -c $((128 / NR)) --cpu-bind=cores)

now() { date +%s.%N; }
dt() { awk -v a="$1" -v b="$2" 'BEGIN { printf "%.2f", b - a }'; }
coord() { awk '$1 == "Host:" {h = $2} $1 == "Port:" {p = $2} END {print "-h " h " -p " p}' "$RC"; }
cmd() { "$MANA/bin/dmtcp_command" $(coord) "$@"; }
iters_done() { grep -ac '\[mana-nccl\] iter' "$@" 2>/dev/null | awk -F: '{s += $NF} END {print s + 0}'; }

# wait_until <timeout_s> <pid> <shell condition>: 0 when true, 1 on timeout or exit
wait_until() {
    local to=$1 pid=$2 i
    shift 2
    for ((i = 0; i < to * 2; i++)); do
        eval "$@" && return 0
        kill -0 "$pid" 2>/dev/null || { eval "$@"; return $?; }
        sleep 0.5
    done
    return 1
}

set_nccl_env() {
    # Common: never load the libfabric/CXI net plugin (its device fds are not
    # checkpointable); between GPUs of one node NCCL uses P2P, SHM or this.
    export NCCL_NET=Socket NCCL_NET_PLUGIN=none NCCL_SOCKET_IFNAME=hsn
    export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,P2P,SHM,NET
    unset NCCL_P2P_DISABLE NCCL_SHM_DISABLE NCCL_CUMEM_ENABLE MANA_CUDA_VMM_ALLOC
    case "$1" in
    native | p2p_cumem) export NCCL_SHM_DISABLE=1 NCCL_CUMEM_ENABLE=1 ;;
    cuda_only) ;; # no NCCL at all; see run_variant
    no_ipc) export NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_CUMEM_ENABLE=0 ;;
    shm) export NCCL_P2P_DISABLE=1 NCCL_CUMEM_ENABLE=0 ;;
    p2p_legacy) export NCCL_SHM_DISABLE=1 NCCL_CUMEM_ENABLE=0 ;;
    p2p_legacy_novmm) export NCCL_SHM_DISABLE=1 NCCL_CUMEM_ENABLE=0 MANA_CUDA_VMM_ALLOC=0 ;;
    *) echo "unknown variant $1" >&2; return 1 ;;
    esac
}

transports() { grep -ahoE 'via (P2P|SHM|NET)[^ ]*' "$@" 2>/dev/null | sort | uniq -c | awk '{printf "%s(%s) ", $3, $1}'; }

run_variant() {
    local v=$1 dir=$OUT/$1 status=ok c1= c2= rs= t0 t1 pid rpid
    rm -rf "$dir" && mkdir -p "$dir" && cd "$dir" || return
    set_nccl_env "$v" || return
    env | grep -E '^(NCCL_|MANA_)' | sort >env.txt

    if [ "$v" = native ]; then
        timeout 600 "${SRUN[@]}" "$TEST" -n 100 -s 0 >launch.log 2>&1 || status=fail_native
        [ "$(grep -ac 'PASS rank' launch.log)" -eq "$NR" ] || status=fail_native
        echo "$v status=$status transports: $(transports launch.log)" | tee -a "$OUT/summary.txt"
        return
    fi

    # cuda_only: the control -- MANA + CUDA plugin, no NCCL plugin, no NCCL.
    local lflags=("${LAUNCH_FLAGS[@]}") targs=()
    if [ "$v" = cuda_only ]; then
        targs=(-c)
        if [ "$MANA_MODE" = plain ]; then lflags=(--with-plugin "$MANA/lib/dmtcp/libdmtcp_cuda.so"); else lflags=(--cuda); fi
    fi
    "$MANA/bin/mana_coordinator" -p 0 >coord.log 2>&1 || { echo "$v: coordinator failed" | tee -a "$OUT/summary.txt"; return; }
    "${SRUN[@]}" "$MANA/bin/mana_launch" "${lflags[@]}" \
        "$TEST" -n "$ITERS" -s "$SLEEP_MS" "${targs[@]}" >launch.log 2>&1 &
    pid=$!

    if ! wait_until 300 $pid '[ $(iters_done launch.log) -ge 3 ]'; then
        status=fail_startup
    else
        t0=$(now); timeout 600 "$MANA/bin/dmtcp_command" $(coord) -bc >ckpt1.log 2>&1 || status=fail_ckpt1; t1=$(now); c1=$(dt "$t0" "$t1")
        n=$(iters_done launch.log)
        if [ $status = ok ] && ! wait_until 300 $pid "[ \$(iters_done launch.log) -ge $((n + 3)) ]"; then
            status=fail_resume
        fi
    fi
    if [ $status = ok ]; then
        t0=$(now); timeout 600 "$MANA/bin/dmtcp_command" $(coord) -bc >ckpt2.log 2>&1 || status=fail_ckpt2; t1=$(now); c2=$(dt "$t0" "$t1")
    fi
    cmd --kill >/dev/null 2>&1
    wait $pid
    cmd --quit >/dev/null 2>&1

    if [ $status = ok ]; then
        ls -d ckpt_rank_* >/dev/null 2>&1 || status=fail_no_images
    fi
    if [ $status = ok ]; then
        "$MANA/bin/mana_coordinator" -p 0 >>coord.log 2>&1
        t0=$(now)
        timeout 900 "${SRUN[@]}" "$MANA/bin/mana_restart" "${RESTART_FLAGS[@]}" --restartdir "$dir" >restart.log 2>&1 &
        rpid=$!
        wait $rpid || status=fail_restart_rc$?
        rs=$(dt "$t0" "$(now)")
        cmd --quit >/dev/null 2>&1
        [ "$(grep -ac 'PASS rank' restart.log launch.log | awk -F: '{s += $NF} END {print s}')" -ge "$NR" ] ||
            { [ $status = ok ] && status=fail_no_pass; }
    fi
    [ "$v" = cuda_only ] || grep -aq 'NCCL PLUGIN\] Checkpoint overhead' launch.log || status="$status(nccl_plugin_silent)"
    grep -aE 'FAIL|JASSERT|Segmentation|cuCheckpoint|error' launch.log restart.log 2>/dev/null | head -n 5 >errors.txt
    echo "$v status=$status ckpt1=${c1}s ckpt2=${c2}s restart_to_end=${rs}s" \
        "rebuilds=$(grep -ac 'Rebuild overhead' launch.log restart.log 2>/dev/null | awk -F: '{s += $NF} END {print s}')" \
        "images=$(du -sh ckpt_rank_* 2>/dev/null | awk '{printf "%s ", $1}')" \
        "transports: $(transports launch.log)" | tee -a "$OUT/summary.txt"
}

{
    echo "== $(date -Is) job $SLURM_JOB_ID node $(hostname) NR=$NR"
    echo "== MANA $(git -C "$MANA" rev-parse --short HEAD) DMTCP $(git -C "$MANA/dmtcp" rev-parse --short HEAD) mode=$MANA_MODE"
    nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader
} | tee "$OUT/summary.txt"
for v in $VARIANTS; do run_variant "$v"; done
echo "results: $OUT/summary.txt"
