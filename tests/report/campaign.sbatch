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
#               netsplit: subset of native native_ref ns ns_resume ns_restart
#   REF_BENCHES, REF_LAYOUTS  native_ref runs only these benchmarks and
#               layouts (empty: all)
#   CKPT_DELAYS checkpoint modes run once per delay (s), interleaved
#   STRESS_CASES, STRESS_MODES  bench:NNxGPN cases run once each in every
#               STRESS_MODES mode (e.g. ns_resume5 ns_restart3), before the
#               rest of their layout, with CKPT_DELAY=STRESS_CKPT_DELAY

set -uo pipefail
source "$REPORT_DIR/common.sh"

if [ "$PLAN" = netsplit ]; then
    # netsplit/TEST_PLAN.md §3-§5: `native` is NCCL with GDR off, netsplit's
    # data path (common.sh); `native_ref`, production NCCL with GDR on, runs
    # only the point-to-point microbenchmarks across nodes.
    : "${BENCHES:=is pytorch nccl_latency nccl_bw nccl_allreduce nccl_allgather nccl_reduce_scatter nccl_bcast nccl_reduce nccl_alltoall}"
    : "${CKPT_BENCHES:=is pytorch}"
    : "${MODES:=native native_ref ns ns_resume ns_restart}"
    : "${REF_BENCHES=nccl_latency nccl_bw}"
    : "${REF_LAYOUTS=2x1}"
    : "${STRESS_CASES=nccl_allreduce:2x4 nccl_alltoall:2x4 nccl_allgather:2x4 nccl_reduce_scatter:2x4 nccl_bcast:2x4 nccl_bw:2x1 is:2x4 is:4x4 pytorch:2x4 pytorch:4x4}"
    : "${STRESS_MODES=ns_resume5 ns_restart3}"
fi
: "${REF_BENCHES=}"
: "${REF_LAYOUTS=}"
: "${STRESS_CASES=}"
: "${STRESS_MODES=}"
# Checkpoint runs name their delay in the run directory (run_bench.sh).
export DELAY_IN_RUN_DIR=1
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
echo "modes $MODES; checkpoint delays $CKPT_DELAYS; stress $STRESS_MODES on $STRESS_CASES"

in_list() { case " $2 " in *" $1 "*) return 0 ;; esac; return 1; }

# fits <bench> <mode> <layout>: does the job have time left for the run's
# worst case (every phase limit reached, at the CKPT_DELAY in effect)?
# Otherwise the run is skipped and listed, rather than cut off by the end of
# the job without a result.
fits() {
    [ "$TIME_GUARD" = 1 ] || return 0
    local need left
    need=$(run_worst_case_s "$1" "$2")
    left=$(job_time_left_s)
    [ "$left" -ge "$need" ] && return 0
    echo "skipped $1 $2${DELAY_TAG:+ $DELAY_TAG} $3 rep$REP: ${left}s left in job $SLURM_JOB_ID, the run may take ${need}s" |
        tee -a "$CAMPAIGN_DIR/skipped.txt"
    return 1
}

# run_one <bench> <mode> <nodes> <gpus_per_node> <rep> [<ckpt delay>]: one
# run if the job has time for it, then the cleanup.
run_one() {
    local DELAY_TAG=${6:+d$6}
    if [ -n "${6:-}" ]; then
        CKPT_DELAY=$6 fits "$1" "$2" "$3x$4" || return 0
        CKPT_DELAY=$6 "$REPORT_DIR/run_bench.sh" "$1" "$2" "$3" "$4" "$5"
    else
        fits "$1" "$2" "$3x$4" || return 0
        "$REPORT_DIR/run_bench.sh" "$1" "$2" "$3" "$4" "$5"
    fi
    # Nothing from a failed run may leak into the next one.
    cleanup_leftovers
}

for L in $LAYOUTS; do
    LNN=${L%x*} GPN=${L#*x}
    if [ "$LNN" -gt "$NN" ]; then
        echo "skipped layout $L: the job has $NN nodes" | tee -a "$CAMPAIGN_DIR/skipped.txt"
        continue
    fi
    # Repeated checkpoints first: their worst case is the longest, and the
    # time guard would skip them at the end of the job.
    REP=1
    for c in $STRESS_CASES; do
        [ "${c#*:}" = "$L" ] || continue
        for MODE in $STRESS_MODES; do
            run_one "${c%%:*}" "$MODE" "$LNN" "$GPN" 1 "$STRESS_CKPT_DELAY"
        done
    done
    SEEN=
    for BENCH in $BENCHES $CKPT_BENCHES; do
        # BENCHES and CKPT_BENCHES overlap; visit each benchmark once.
        in_list "$BENCH" "$SEEN" && continue
        SEEN="$SEEN $BENCH"
        layout_ok "$BENCH" "$LNN" "$GPN" || continue
        for REP in $(seq 1 "$REPS"); do
            # Modes interleaved within a repetition, so drift hits all equally.
            for MODE in $MODES; do
                read -r _ kind _ <<<"$(mode_info "$MODE")"
                case "$kind" in
                perf) in_list "$BENCH" "$BENCHES" || continue ;;
                *) in_list "$BENCH" "$CKPT_BENCHES" || continue ;;
                esac
                if [ "$MODE" = native_ref ]; then
                    [ -z "$REF_BENCHES" ] || in_list "$BENCH" "$REF_BENCHES" || continue
                    [ -z "$REF_LAYOUTS" ] || in_list "$L" "$REF_LAYOUTS" || continue
                fi
                if [ "$kind" = perf ]; then
                    run_one "$BENCH" "$MODE" "$LNN" "$GPN" "$REP"
                else
                    for D in $CKPT_DELAYS; do
                        run_one "$BENCH" "$MODE" "$LNN" "$GPN" "$REP" "$D"
                    done
                fi
            done
        done
    done
done

echo "campaign $CAMPAIGN job $SLURM_JOB_ID done; summarize with:"
echo "  python3 $REPORT_DIR/summarize.py $CAMPAIGN_DIR"
