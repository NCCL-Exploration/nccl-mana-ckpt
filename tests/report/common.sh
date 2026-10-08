# Shared configuration for the tech-report experiments. Sourced, not executed.
#
# Every knob below can be overridden from the environment, e.g.
#   IS_CLASS=B REPS=5 sbatch ...
# and every value actually used is recorded in each run's meta.env.

REPORT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$REPORT_DIR/../.." && pwd)
# /global/u1 and /global/homes name the same directory; DMTCP records absolute
# paths, so use one spelling everywhere (matches run_perlmutter_ckpt.sh).
REPO_ROOT=${REPO_ROOT//\/global\/u1//global/homes}
REPORT_DIR=$REPO_ROOT/tests/report

# --- Site / allocation ---------------------------------------------------
: "${ACCOUNT:=m4128}"
: "${QOS:=regular}"
: "${CUDA_MODULE:=cudatoolkit/13.2}"
: "${NCCL_MODULE:=nccl/2.29.2-cu13}"
: "${PYTORCH_MODULE:=pytorch/2.13.0}" # pulls in cudatoolkit/13.0 + the same NCCL

# --- Where everything lives (all on $SCRATCH: Lustre, best I/O) ----------
: "${RESULTS_ROOT:=$SCRATCH/nccl-ckpt-report}"
: "${CAMPAIGN:=$(date +%Y%m%d)}"
CAMPAIGN_DIR=$RESULTS_ROOT/$CAMPAIGN
BIN_DIR=$RESULTS_ROOT/bin
NCCL_BENCH=$BIN_DIR/nccl_bench
# Lustre striping for checkpoint directories. Empty = filesystem default
# (Perlmutter scratch: 1 stripe, 1 MiB). Each rank writes its own image file,
# so a stripe count > 1 only helps when single images are large (PyTorch).
# Nothing here uses MPI: every job is launched by srun, one task per GPU.
: "${CKPT_STRIPE_COUNT:=}"
: "${CKPT_STRIPE_SIZE:=16M}"

# --- DMTCP / plugin --------------------------------------------------------
DMTCP_ROOT=$REPO_ROOT/dmtcp
DMTCP_LAUNCH=$DMTCP_ROOT/bin/dmtcp_launch
DMTCP_RESTART=$DMTCP_ROOT/bin/dmtcp_restart
DMTCP_COMMAND=$DMTCP_ROOT/bin/dmtcp_command
DMTCP_COORDINATOR=$DMTCP_ROOT/bin/dmtcp_coordinator
# Absolute paths and NCCL-before-CUDA order are both required; see CLAUDE.md.
PLUGIN=$REPO_ROOT/bin/libncclplugin.so:$DMTCP_ROOT/plugin/cuda/libdmtcp_cuda.so
# Diagnostics only: DMTCP_PLUGINS replaces the plugin list; set but empty
# means dmtcp_launch with no plugins at all (runs, but cannot checkpoint GPUs).
if [ "${DMTCP_PLUGINS+set}" = set ]; then PLUGIN=$DMTCP_PLUGINS; fi

# --- Experiment parameters -------------------------------------------------
: "${REPS:=3}"
: "${CKPT_DELAY:=10}"        # s after the app reports progress before checkpointing
: "${RUN_TIMEOUT:=1800}"     # s, hard cap on any single srun step
: "${STARTUP_TIMEOUT:=600}"  # s to wait for the app's progress marker
# NCCL settings applied identically to native, dmtcp and checkpoint runs:
#   conservative  -- the set run_perlmutter_ckpt.sh is known to checkpoint with
#   default       -- NCCL defaults (P2P/SHM/NVLS on); try in the smoke test
: "${NCCL_PROFILE:=conservative}"
# Inter-node transport: socket (NCCL built-in TCP over hsn*; no /dev/cxi fds
# for DMTCP to trip over) or ofi (the module default, AWS Libfabric on CXI).
: "${NET_PROFILE:=socket}"

# Run lengths are targets in seconds, not iteration counts: each benchmark
# times its first iterations, agrees on the slowest rank, and sizes the rest
# of the run to fit -- so a run lasts about as long at 2 GPUs as at 32.
#   *_TARGET_S       timing runs (native, native_ref, dmtcp)
#   *_CKPT_TARGET_S  checkpoint runs; the checkpoint lands CKPT_DELAY s in,
#                    leaving the rest for the resume / restart to finish
: "${IS_CLASS:=C}"
: "${IS_TARGET_S:=20}"
: "${IS_CKPT_TARGET_S:=45}"
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

now() { date +%s.%N; }
elapsed() { awk -v a="$1" -v b="$2" 'BEGIN { printf "%.3f", b - a }'; }

load_modules() {
    # $1: is | nccl | pytorch
    if ! type module >/dev/null 2>&1; then
        # shellcheck disable=SC1091
        source "$LMOD_PKG/init/bash"
    fi
    # Darshan's LD_PRELOAD wrappers recurse with DMTCP's (see run_perlmutter_ckpt.sh).
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
    # NCCL RAS (2.24+) runs a per-process service with persistent TCP links to
    # peer processes that outlive every communicator, so the plugin's teardown
    # cannot close them, and DMTCP's socket drain/handshake on those links
    # deadlocked a 2-node checkpoint (2026-10-07). RAS is diagnostics only; off
    # in every mode, native_ref included, so all modes stay comparable.
    export NCCL_RAS_ENABLE=0
    # The ablation profiles start from conservative and re-enable one feature.
    local cons="NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_CUMEM_ENABLE=0
        NCCL_DMABUF_ENABLE=0 NCCL_NVLS_ENABLE=0 NCCL_NET_GDR_LEVEL=0"
    case "$1" in
    conservative) export $cons ;;
    p2p) export $cons NCCL_P2P_DISABLE=0 ;;                        # legacy cudaIpc
    p2p_cumem) export $cons NCCL_P2P_DISABLE=0 NCCL_CUMEM_ENABLE=1 ;; # VMM fd IPC
    shm) export $cons NCCL_SHM_DISABLE=0 ;;
    cumem) export $cons NCCL_CUMEM_ENABLE=1 ;;
    gdr) export $cons NCCL_NET_GDR_LEVEL=PHB NCCL_DMABUF_ENABLE=1 ;; # with NET_PROFILE=ofi
    # NCCL defaults minus GPUDirect RDMA, the one feature the 2026-10-07
    # ablation found not to survive a checkpoint (NET/OFI cannot re-register
    # GPU memory afterwards: register_mr_buffers RC 22). Candidate profile
    # for the campaign, with NET_PROFILE=ofi.
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

