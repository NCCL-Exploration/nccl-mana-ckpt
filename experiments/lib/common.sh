# Shared configuration for the paper experiments. Sourced, not executed.
#
# Every knob below can be overridden from the environment, e.g.
#   REPS=5 CG_CLASS=B experiments/submit.sh scaling
# and every value actually used is recorded in each run's meta.env.
#
# Derived from the tests/report harness of the netsplit branch, without
# netsplit: every checkpoint here is the single-process plugin
# (libncclplugin.so + DMTCP's CUDA plugin) under plain dmtcp_launch.

LIB_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$LIB_DIR/../.." && pwd)
# /global/u1 and /global/homes name the same directory; DMTCP records absolute
# paths, so use one spelling everywhere (matches run_perlmutter_ckpt.sh).
REPO_ROOT=${REPO_ROOT//\/global\/u1//global/homes}
EXP_DIR=$REPO_ROOT/experiments
LIB_DIR=$EXP_DIR/lib

# --- Site / allocation ---------------------------------------------------
: "${ACCOUNT:=m4128}"
: "${QOS:=regular}"
: "${CUDA_MODULE:=cudatoolkit/13.2}"
: "${NCCL_MODULE:=nccl/2.29.2-cu13}"
: "${PYTORCH_MODULE:=pytorch/2.13.0}" # pulls in cudatoolkit/13.0 + the same NCCL

# --- Where everything lives (all on $SCRATCH: Lustre, best I/O) ----------
: "${RESULTS_ROOT:=$SCRATCH/nccl-ckpt-paper}"
: "${CAMPAIGN:=$(date +%Y%m%d)}"
CAMPAIGN_DIR=$RESULTS_ROOT/$CAMPAIGN
BIN_DIR=$RESULTS_ROOT/bin
NCCL_BENCH=$BIN_DIR/nccl_bench
P2P_PENDING=$BIN_DIR/p2p_pending # tests/multi/p2p_pending.cu
TORCH_SAVE_BENCH=$EXP_DIR/torchsave/torch_save_bench.py
# Lustre striping for checkpoint directories. Empty = filesystem default
# (Perlmutter scratch: 1 stripe, 1 MiB). Each rank writes its own image file,
# so a stripe count > 1 only helps when single images are large (PyTorch).
# The torch.save runs use the same striping, so the two are comparable.
: "${CKPT_STRIPE_COUNT:=}"
: "${CKPT_STRIPE_SIZE:=16M}"

# --- DMTCP / plugin --------------------------------------------------------
DMTCP_ROOT=$REPO_ROOT/dmtcp
DMTCP_LAUNCH=$DMTCP_ROOT/bin/dmtcp_launch
DMTCP_RESTART=$DMTCP_ROOT/bin/dmtcp_restart
DMTCP_COMMAND=$DMTCP_ROOT/bin/dmtcp_command
DMTCP_COORDINATOR=$DMTCP_ROOT/bin/dmtcp_coordinator
# Absolute paths and NCCL-before-CUDA order are both required.
PLUGIN=$REPO_ROOT/bin/libncclplugin.so:$DMTCP_ROOT/plugin/cuda/libdmtcp_cuda.so
# Diagnostics only: DMTCP_PLUGINS replaces the plugin list; set but empty
# means dmtcp_launch with no plugins at all (runs, but cannot checkpoint GPUs).
if [ "${DMTCP_PLUGINS+set}" = set ]; then PLUGIN=$DMTCP_PLUGINS; fi

# --- Experiment parameters -------------------------------------------------
: "${REPS:=3}"
: "${CKPT_DELAY:=10}"        # s after the app reports progress before checkpointing
: "${RUN_TIMEOUT:=1800}"     # s, hard cap on any single srun step
# Per-phase limits. A job still running when one passes is killed, and the
# run is recorded as hang_<phase>.
: "${STARTUP_TIMEOUT:=600}"  # s to wait for the app's progress marker
: "${CKPT_TIMEOUT:=900}"     # s from dmtcp_command -c until the restart script exists
: "${PROGRESS_TIMEOUT:=120}" # s for new progress after a checkpoint the job survives
: "${RESTART_TIMEOUT:=600}"  # s until every restored rank has rebuilt its communicators
: "${HANG_GRACE_S:=300}"     # s past the run's target before a running job counts as hung
: "${TIME_GUARD:=1}"         # 1: campaign.sbatch skips runs that could outlast the job
# NCCL settings applied identically to native, dmtcp and checkpoint runs:
#   default       -- NCCL defaults within a node (P2P/NVLink, SHM, cuMem), as
#                    the paper's methodology states (Section "Methodology")
#   nogdr         -- NCCL defaults minus GPUDirect RDMA
#   conservative  -- P2P/SHM/cuMem/NVLS/DMABUF off: the set
#                    run_perlmutter_ckpt.sh checkpoints with, for diagnosis
: "${NCCL_PROFILE:=default}"
# Inter-node transport: socket (NCCL built-in TCP over hsn*; no /dev/cxi fds
# for DMTCP to trip over; a restarted process cannot reopen OFI endpoints) or
# ofi (the module default, AWS Libfabric on CXI). native_ref always uses ofi
# with NCCL's defaults: the paper's "Cost of the Checkpoint-Safe Network".
: "${NET_PROFILE:=socket}"

# Run lengths are targets in seconds, not iteration counts: each benchmark
# times its first iterations, agrees on the slowest rank, and sizes the rest
# of the run to fit -- so a run lasts about as long at 2 GPUs as at 16.
#   *_TARGET_S       timing runs (native, native_ref, dmtcp)
#   *_CKPT_TARGET_S  checkpoint runs; the checkpoint lands CKPT_DELAY s in,
#                    leaving the rest for the resume / restart to finish
: "${IS_CLASS:=C}"
: "${IS_TARGET_S:=20}"
: "${IS_CKPT_TARGET_S:=45}"
: "${CG_CLASS:=C}"
: "${CG_TARGET_S:=20}"
: "${CG_CKPT_TARGET_S:=45}"
: "${MG_CLASS:=C}"
: "${MG_TARGET_S:=20}"
: "${MG_CKPT_TARGET_S:=45}"
: "${NAS_CLASSES:=B C}"      # build.sh: classes built for each NAS benchmark
: "${PT_WIDTH:=4096}"        # 2 x 4096^2 fp32 params -> ~134 MB of grads/step
: "${PT_BATCH:=64}"
: "${PT_TARGET_S:=20}"
: "${PT_CKPT_TARGET_S:=45}"
: "${NB_MIN_SIZE:=4}"           # nccl_bench message sizes (bytes)
: "${NB_MAX_SIZE:=67108864}"    # 64 MiB
: "${NB_SIZE_BUDGET_S:=0.5}"    # timing runs: seconds per message size
: "${NB_ITERS:=}"               # set to use a fixed count instead of the budget
: "${NB_CKPT_SIZE:=4194304}"    # checkpoint runs: one size ...
: "${NB_CKPT_TARGET_S:=45}"     # ... for this long
: "${NB_PROGRESS:=200}"         # '# progress' line every N iterations
: "${P2P_WINDOW_S:=20}"         # p2p_*: how long one side waits before its operation
# torch.save baseline (torchsave/torch_save_bench.py). TS_WIDTHS are the
# pytorch_minimal_test.py model widths to save; each is also run as a
# checkpointed PyTorch job (bench pytorch_w<width>) so the two compare
# model for model. The model has 2 x width^2 fp32 parameters: 4096 (PT_WIDTH)
# is 134 MB, 8192 is 537 MB, 16384 is 2.1 GB; SGD momentum doubles that.
: "${TS_WIDTHS:=4096 8192 16384}"
: "${TS_REPS:=5}"               # saves per width and policy within one run
: "${TS_POLICIES:=all rank0}"   # all: every rank saves its replica; rank0: rank 0 only
: "${TS_OPTIM:=1}"              # 1: also save the optimizer state, as a training checkpoint does

now() { date +%s.%N; }
elapsed() { awk -v a="$1" -v b="$2" 'BEGIN { printf "%.3f", b - a }'; }

load_modules() {
    # $1: nas | nccl | pytorch
    if ! type module >/dev/null 2>&1; then
        # shellcheck disable=SC1091
        source "$LMOD_PKG/init/bash"
    fi
    # Darshan's LD_PRELOAD wrappers recurse with DMTCP's (dup2 -> stack
    # overflow), see run_perlmutter_ckpt.sh. Off for every mode, native too.
    export DARSHAN_DISABLE=1
    module unload darshan vasp >/dev/null 2>&1 || true
    case "$1" in
    pytorch)
        module load "$PYTORCH_MODULE"
        ;;
    *)
        module load "$CUDA_MODULE" "$NCCL_MODULE"
        ;;
    esac
}

apply_nccl_profile() {
    # $1: profile, $2: net profile. Exports, so srun propagates them.
    export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
    # NCCL RAS (2.24+) keeps TCP links to peer processes that outlive every
    # communicator, so the plugin's teardown cannot close them, and DMTCP's
    # socket drain on those links deadlocked a 2-node checkpoint. Off in every
    # mode, native_ref included, so all modes stay comparable.
    export NCCL_RAS_ENABLE=0
    local cons="NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_CUMEM_ENABLE=0
        NCCL_DMABUF_ENABLE=0 NCCL_NVLS_ENABLE=0 NCCL_NET_GDR_LEVEL=0"
    case "$1" in
    conservative) export $cons ;;
    nogdr) export NCCL_NET_GDR_LEVEL=0 NCCL_DMABUF_ENABLE=0 ;;
    default) ;;
    *) echo "unknown NCCL_PROFILE=$1" >&2; return 1 ;;
    esac
    case "$2" in
    socket)
        export NCCL_NET=Socket NCCL_NET_PLUGIN=none NCCL_SOCKET_IFNAME=hsn
        ;;
    ofi) ;; # keep the module's NCCL_NET="AWS Libfabric"
    *) echo "unknown NET_PROFILE=$2" >&2; return 1 ;;
    esac
}

# mode_info <mode>: prints "<launcher> <kind> <n>". launcher: native | dmtcp
# (launch_rank.sh's modes); kind: perf (no checkpoint) | resume | restart;
# n: checkpoints in the run (resume) or generations of checkpoint, kill and
# restart (restart), 0 for perf. A trailing number sets n: ckpt_resume5
# takes 5 checkpoints, ckpt_restart3 restarts 3 times.
mode_info() {
    local base=$1 n=1
    if [[ $1 =~ ^ckpt_(resume|restart)([0-9]+)$ ]]; then
        base=${1%"${BASH_REMATCH[2]}"} n=${BASH_REMATCH[2]}
        [ "$n" -ge 1 ] || return 1
    fi
    case "$base" in
    native | native_ref) echo "native perf 0" ;;
    dmtcp) echo "dmtcp perf 0" ;;
    ckpt_resume) echo "dmtcp resume $n" ;;
    ckpt_restart) echo "dmtcp restart $n" ;;
    *) return 1 ;;
    esac
}

# layout_ok <bench> <nodes> <gpus_per_node>: can the benchmark run there?
layout_ok() {
    local nr=$(($2 * $3))
    case "$1" in
    nccl_latency | nccl_bw | nccl_bibw | p2p_recv_first | p2p_recv_first_cold | p2p_send_first)
        [ "$nr" -eq 2 ] ;;
    p2p_relay) [ "$nr" -eq 3 ] ;; # a 4th rank's allreduce would let rank 1 through anyway
    p2p_chain) [ "$nr" -ge 3 ] ;;
    p2p_multi) [ "$nr" -ge 4 ] ;;
    p2p_create | p2p_split | p2p_destroy) [ "$nr" -ge 2 ] ;;
    is | cg | mg) [ $((nr & (nr - 1))) -eq 0 ] ;; # NPB needs a power-of-two rank count
    *) return 0 ;;
    esac
}

# pt_width <bench>: the model width of a PyTorch benchmark (pytorch = PT_WIDTH,
# pytorch_w<N> = N).
pt_width() {
    case "$1" in
    pytorch_w*) echo "${1#pytorch_w}" ;;
    *) echo "$PT_WIDTH" ;;
    esac
}

# bench_target_s <bench> <perf|ckpt>: about how long the program is asked to
# run, in seconds; the completion limit is this plus HANG_GRACE_S.
bench_target_s() {
    case "$1:$2" in
    is:perf) echo "$IS_TARGET_S" ;;
    is:ckpt) echo "$IS_CKPT_TARGET_S" ;;
    cg:perf) echo "$CG_TARGET_S" ;;
    cg:ckpt) echo "$CG_CKPT_TARGET_S" ;;
    mg:perf) echo "$MG_TARGET_S" ;;
    mg:ckpt) echo "$MG_CKPT_TARGET_S" ;;
    pytorch*:perf) echo "$PT_TARGET_S" ;;
    pytorch*:ckpt) echo "$PT_CKPT_TARGET_S" ;;
    nccl_*:ckpt) echo "$NB_CKPT_TARGET_S" ;;
    p2p_*) echo $((P2P_WINDOW_S + 10)) ;;
    torchsave:*)
        # Rough: TS_REPS saves and loads per width and policy at >= 0.5 GB/s.
        awk -v ws="$TS_WIDTHS" -v r="$TS_REPS" -v p="$TS_POLICIES" 'BEGIN {
            n = split(ws, w, " "); np = split(p, q, " "); t = 60
            for (i = 1; i <= n; i++) t += np * r * 2 * (3 * 2 * w[i] * w[i] * 4 / 5e8 + 2)
            printf "%d\n", t }'
        ;;
    nccl_*:perf)
        # A fixed iteration count has no target: only RUN_TIMEOUT applies.
        if [ -n "$NB_ITERS" ]; then echo "$RUN_TIMEOUT"; return; fi
        # Each size runs its budget plus an untimed warm-up of similar length.
        awk -v a="$NB_MIN_SIZE" -v b="$NB_MAX_SIZE" -v t="$NB_SIZE_BUDGET_S" \
            'BEGIN { for (s = a; s <= b; s *= 2) n++; printf "%d\n", 2 * n * t + 1 }'
        ;;
    *) echo "$RUN_TIMEOUT" ;;
    esac
}

# run_target_s <bench> <n>: a checkpoint run's target, 20 s longer for each
# of its n checkpoints after the first.
run_target_s() {
    local t
    t=$(bench_target_s "$1" ckpt)
    if [ "$2" -gt 1 ]; then t=$((t + ($2 - 1) * 20)); fi
    echo "$t"
}

imin() { if [ "$1" -lt "$2" ]; then echo "$1"; else echo "$2"; fi; }

# run_worst_case_s <bench> <mode>: the longest a run can take with every
# phase limit reached, plus a minute for kills and cleanup.
run_worst_case_s() {
    local launcher kind n t w i cap=$RUN_TIMEOUT
    read -r launcher kind n <<<"$(mode_info "$2")"
    if [ "$kind" = perf ]; then t=$(bench_target_s "$1" perf); else t=$(run_target_s "$1" "$n"); fi
    case "$kind" in
    perf) w=$(imin $((STARTUP_TIMEOUT + t + HANG_GRACE_S)) "$cap") ;;
    resume)
        w=$(imin $((STARTUP_TIMEOUT + n * (CKPT_DELAY + CKPT_TIMEOUT + PROGRESS_TIMEOUT) + t +
            HANG_GRACE_S)) "$cap")
        ;;
    restart)
        # n + 1 job steps, each under the RUN_TIMEOUT cap.
        w=$(imin $((STARTUP_TIMEOUT + CKPT_DELAY + CKPT_TIMEOUT)) "$cap")
        for ((i = 1; i < n; i++)); do
            w=$((w + $(imin $((RESTART_TIMEOUT + PROGRESS_TIMEOUT + CKPT_DELAY + CKPT_TIMEOUT)) "$cap")))
        done
        w=$((w + $(imin $((RESTART_TIMEOUT + t + HANG_GRACE_S)) "$cap")))
        ;;
    esac
    echo $((w + 60))
}

# --- Allocation -------------------------------------------------------------
# job_time_left_s: seconds before Slurm ends this job (large if unknown).
job_time_left_s() {
    local l d=0 h=0 m=0 s=0 p
    l=$(squeue -h -j "$SLURM_JOB_ID" -o %L 2>/dev/null)
    case "$l" in "" | *[!0-9:-]*) echo 999999; return ;; esac
    if [[ $l == *-* ]]; then d=${l%%-*} l=${l#*-}; fi
    IFS=: read -ra p <<<"$l"
    case ${#p[@]} in
    3) h=${p[0]} m=${p[1]} s=${p[2]} ;;
    2) m=${p[0]} s=${p[1]} ;;
    *) s=${p[0]} ;;
    esac
    echo $((((10#$d * 24 + 10#$h) * 60 + 10#$m) * 60 + 10#$s))
}

# use_job_nodes: sets JOB_NODES to the job's nodes and points the Slurm
# variables at the whole job. A driver run as a one-node step (the interactive
# QOS: salloc, then "srun --overlap") otherwise sees only its step, and its
# sruns fail with "Only allocated 1 nodes" or inherit its CPU binding and its
# cyclic task distribution.
use_job_nodes() {
    local nl
    nl=$(squeue -h -j "$SLURM_JOB_ID" -o %N 2>/dev/null)
    [ -n "$nl" ] || nl=$SLURM_JOB_NODELIST
    mapfile -t JOB_NODES < <(scontrol show hostnames "$nl")
    export SLURM_JOB_NODELIST=$nl SLURM_NODELIST=$nl SLURM_NNODES=${#JOB_NODES[@]} \
        SLURM_JOB_NUM_NODES=${#JOB_NODES[@]}
    unset SLURM_NTASKS SLURM_NPROCS SLURM_TASKS_PER_NODE SLURM_STEP_NODELIST \
        SLURM_STEP_NUM_NODES SLURM_STEP_NUM_TASKS SLURM_STEP_TASKS_PER_NODE \
        SLURM_NTASKS_PER_NODE SLURM_CPUS_PER_TASK SLURM_CPU_BIND SLURM_CPU_BIND_TYPE \
        SLURM_CPU_BIND_LIST SLURM_DISTRIBUTION
}

# cleanup_leftovers: kills what a failed run left on the job's nodes, so it
# cannot disturb the next run: job steps of launch_rank.sh, ranks (the
# benchmarks, PyTorch, dmtcp_restart) and coordinators. Run one campaign per
# allocation: this would kill another's too.
cleanup_leftovers() {
    squeue -h -s -j "$SLURM_JOB_ID" -o '%i %j' 2>/dev/null |
        awk '$2 == "launch_rank.sh" {print $1}' | xargs -r scancel 2>/dev/null
    # Each alternative has a bracket expression, so that it does not match the
    # command lines that carry the pattern (this srun's and its tasks').
    local re='/[n]ccl_bench |/[p]2p_pending |/[i]s_nccl_[A-Z]|/[c]g_nccl_[A-Z]|/[m]g_nccl_[A-Z]'
    re+='|[p]ytorch_minimal_test|[t]orch_save_bench|[d]mtcp_restart|(^|/)[m]tcp_restart'
    # </dev/null: srun forwards its stdin to the tasks, and the caller's stdin
    # may be the plan campaign.sbatch is reading.
    srun --overlap -N "${#JOB_NODES[@]}" --ntasks-per-node=1 --cpu-bind=none \
        pkill -9 -u "$USER" -f "$re" </dev/null >/dev/null 2>&1
    pkill -9 -u "$USER" -f '[d]mtcp_coordinator' 2>/dev/null
    return 0
}
