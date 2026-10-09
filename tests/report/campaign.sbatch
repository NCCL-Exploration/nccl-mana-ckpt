#!/bin/bash
#SBATCH -C gpu
#SBATCH --gpus-per-node=4
#SBATCH -J nccl-ckpt-report
#SBATCH -o %x_%j.out
#
# One allocation = one node count. Submit through submit_all.sh, which sets
# -N / -t / -A / -q and exports:
#   REPORT_DIR  tests/report in the repo (sbatch copies this file to a spool dir)
#   PLAN        plugin (default) or netsplit (common.sh)
#   GPN_LIST    GPUs-per-node values to run at this node count, e.g. "2 4"
#   LAYOUTS     instead of GPN_LIST: NODESxGPUS layouts to run in this
#               allocation, e.g. "1x4 2x4" (a layout's ranks use the job's
#               first NODES nodes)
#   BENCHES     benchmarks for the runtime-overhead study (no checkpoint)
#   CKPT_BENCHES benchmarks for the checkpoint/restart study
#   MODES       plugin:   subset of native native_ref dmtcp ckpt_resume ckpt_restart
#               netsplit: subset of native ns ns_resume ns_restart

set -uo pipefail
source "$REPORT_DIR/common.sh"

if [ "$PLAN" = netsplit ]; then
    # netsplit/TEST_PLAN.md §3 and §4: `native` is production NCCL
    # (common.sh sets the module defaults for this plan).
    : "${BENCHES:=is pytorch nccl_latency nccl_bw nccl_allreduce nccl_allgather nccl_reduce_scatter nccl_bcast nccl_reduce nccl_alltoall}"
    : "${CKPT_BENCHES:=is pytorch}"
    : "${MODES:=native ns ns_resume ns_restart}"
fi
: "${GPN_LIST:=4}"
: "${BENCHES:=is pytorch nccl_latency nccl_bw nccl_bibw nccl_allreduce nccl_allgather nccl_bcast nccl_reduce nccl_reduce_scatter nccl_alltoall}"
: "${CKPT_BENCHES:=is pytorch nccl_bw nccl_allreduce nccl_alltoall}"
: "${MODES:=native native_ref dmtcp ckpt_resume ckpt_restart}"
use_job_nodes
NN=${#JOB_NODES[@]}
if [ -z "${LAYOUTS:-}" ]; then
    LAYOUTS=
    for GPN in $GPN_LIST; do LAYOUTS+=" ${NN}x$GPN"; done
fi

mkdir -p "$CAMPAIGN_DIR"
"$REPORT_DIR/sysinfo.sh" "$CAMPAIGN_DIR/sysinfo/job$SLURM_JOB_ID"
echo "plan $PLAN campaign $CAMPAIGN job $SLURM_JOB_ID nodes $SLURM_JOB_NODELIST layouts $LAYOUTS"

in_list() { case " $2 " in *" $1 "*) return 0 ;; esac; return 1; }

# fits <bench> <mode> <layout>: does the job have time left for the run's
# worst case (every phase limit reached)? Otherwise the run is skipped and
# listed, rather than cut off by the end of the job without a result.
fits() {
    [ "$TIME_GUARD" = 1 ] || return 0
    local need left
    need=$(run_worst_case_s "$1" "$2")
    left=$(job_time_left_s)
    [ "$left" -ge "$need" ] && return 0
    echo "skipped $1 $2 $3 rep$REP: ${left}s left in job $SLURM_JOB_ID, the run may take ${need}s" |
        tee -a "$CAMPAIGN_DIR/skipped.txt"
    return 1
}

for L in $LAYOUTS; do
    LNN=${L%x*} GPN=${L#*x}
    if [ "$LNN" -gt "$NN" ]; then
        echo "skipped layout $L: the job has $NN nodes" | tee -a "$CAMPAIGN_DIR/skipped.txt"
        continue
    fi
    SEEN=
    for BENCH in $BENCHES $CKPT_BENCHES; do
        # BENCHES and CKPT_BENCHES overlap; visit each benchmark once.
        in_list "$BENCH" "$SEEN" && continue
        SEEN="$SEEN $BENCH"
        layout_ok "$BENCH" "$LNN" "$GPN" || continue
        for REP in $(seq 1 "$REPS"); do
            # Modes interleaved within a repetition, so drift hits all equally.
            for MODE in $MODES; do
                read -r _ kind <<<"$(mode_info "$MODE")"
                case "$kind" in
                perf) in_list "$BENCH" "$BENCHES" || continue ;;
                *) in_list "$BENCH" "$CKPT_BENCHES" || continue ;;
                esac
                fits "$BENCH" "$MODE" "$L" || continue
                "$REPORT_DIR/run_bench.sh" "$BENCH" "$MODE" "$LNN" "$GPN" "$REP"
                # Nothing from a failed run may leak into the next one.
                cleanup_leftovers
            done
        done
    done
done

echo "campaign $CAMPAIGN job $SLURM_JOB_ID done; summarize with:"
echo "  python3 $REPORT_DIR/summarize.py $CAMPAIGN_DIR"
