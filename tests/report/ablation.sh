#!/bin/bash
# NCCL-feature ablation + latency-overhead probe, on one node (2 GPUs).
# Runs inside an allocation, e.g. on the interactive QOS:
#
#   salloc -A m4128 -q interactive -t 01:00:00 -N 1 -C gpu --gpus-per-node=4 \
#       srun -N 1 -n 1 --overlap bash tests/report/ablation.sh
#
# Part 0  IS timing modes (native / native_ref / dmtcp).
# Part 1  Ablation: starting from the checkpoint-safe "conservative" NCCL
#         profile, re-enable one feature at a time and check whether
#         checkpoint/resume and checkpoint/restart still work, and which
#         transport NCCL actually picked (NCCL_DEBUG=INFO).
# Part 2  Latency probe: where does nccl_latency's large-message overhead
#         under DMTCP come from? 3 repetitions per variant, interleaved.
#
# Everything goes to $RESULTS_ROOT/$ABL/<part>/...; the summary is
# $RESULTS_ROOT/$ABL/ablation_summary.md (ablation_summary.py).

set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
: "${ABL:=ablation-$(date +%Y%m%d-%H%M)}"
: "${PARTS:=0 1 2}"
: "${ABL_CONFIGS:=conservative:socket default:socket p2p:socket p2p_cumem:socket shm:socket cumem:socket conservative:ofi gdr:ofi}"
: "${PROBE_REPS:=3}"
OUT=$RESULTS_ROOT/$ABL
mkdir -p "$OUT"

# Short, smoke-sized runs.
export REPS=1 IS_CLASS=B CKPT_DELAY=5 IS_TARGET_S=10 PT_TARGET_S=10
export NB_SIZE_BUDGET_S=0.2 NB_MAX_SIZE=1048576
export IS_CKPT_TARGET_S=25 PT_CKPT_TARGET_S=25 NB_CKPT_TARGET_S=25

# run <sub-campaign> <bench> <mode> <rep>  (extra env comes from the caller)
run() {
    CAMPAIGN=$ABL/$1 "$REPORT_DIR/run_bench.sh" "$2" "$3" 1 2 "$4"
    pkill -9 -u "$USER" -f dmtcp_coordinator 2>/dev/null || true
}

"$REPORT_DIR/sysinfo.sh" "$OUT/sysinfo" >/dev/null

if [[ " $PARTS " == *" 0 "* ]]; then
    for m in native native_ref dmtcp; do run is_timing is "$m" 1; done
fi

if [[ " $PARTS " == *" 1 "* ]]; then
    for c in $ABL_CONFIGS; do
        prof=${c%%:*} net=${c##*:}
        for b in nccl_bw nccl_allreduce; do
            for m in native ckpt_resume ckpt_restart; do
                NCCL_PROFILE=$prof NET_PROFILE=$net NCCL_DEBUG=INFO \
                    NCCL_DEBUG_SUBSYS=INIT,P2P,SHM,NET run "abl_${prof}_${net}" "$b" "$m" 1
            done
        done
    done
fi

if [[ " $PARTS " == *" 2 "* ]]; then
    # Sizes around the onset seen in the smoke tests (256 KiB) and beyond.
    export NB_MIN_SIZE=65536 NB_MAX_SIZE=4194304 NB_SIZE_BUDGET_S=1
    CUDA_ONLY=$DMTCP_ROOT/plugin/cuda/libdmtcp_cuda.so
    for rep in $(seq 1 "$PROBE_REPS"); do
        run probe_native nccl_latency native "$rep"
        run probe_dmtcp_full nccl_latency dmtcp "$rep"
        DMTCP_PLUGINS="" run probe_dmtcp_noplugins nccl_latency dmtcp "$rep"
        DMTCP_PLUGINS=$CUDA_ONLY run probe_dmtcp_cudaonly nccl_latency dmtcp "$rep"
        NCCL_PROFILE=p2p run probe_p2p_native nccl_latency native "$rep"
        NCCL_PROFILE=p2p run probe_p2p_dmtcp nccl_latency dmtcp "$rep"
        NCCL_PROTO=LL128 run probe_ll128_native nccl_latency native "$rep"
        NCCL_PROTO=LL128 run probe_ll128_dmtcp nccl_latency dmtcp "$rep"
    done
fi

python3 "$REPORT_DIR/ablation_summary.py" "$OUT" >/dev/null
echo "ablation $ABL done: $OUT/ablation_summary.md"
