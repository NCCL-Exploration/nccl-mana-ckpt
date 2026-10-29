#!/bin/bash
#SBATCH -C gpu
#SBATCH --gpus-per-node=4
#SBATCH -J nccl-ckpt-paper
#SBATCH -o %x_%j.out
#
# One allocation = one node count: runs every run of plan.sh's EXPERIMENTS
# that needs exactly this many nodes. Submit through submit.sh, which sets
# -N / -t / -A / -q and exports:
#   EXP_DIR      experiments/ in the repo (sbatch copies this file to a spool dir)
#   EXPERIMENTS  subset of: micro scaling torchsave p2p
#   CAMPAIGN     results go to $RESULTS_ROOT/$CAMPAIGN
# Also runs as an srun step inside an interactive allocation (submit.sh --here).

set -uo pipefail
source "$EXP_DIR/lib/common.sh"
source "$EXP_DIR/plan.sh"
: "${EXPERIMENTS:=$EXPERIMENTS_ALL}"
use_job_nodes
NN=${#JOB_NODES[@]}

mkdir -p "$CAMPAIGN_DIR"
"$LIB_DIR/sysinfo.sh" "$CAMPAIGN_DIR/sysinfo/job$SLURM_JOB_ID"
SKIPPED=$CAMPAIGN_DIR/skipped.txt

for exp in $EXPERIMENTS; do
    plan_runs "$exp" "$NN" >"$CAMPAIGN_DIR/plan_${exp}_n$NN.txt" || continue
    echo "== $exp at $NN node(s): $(wc -l <"$CAMPAIGN_DIR/plan_${exp}_n$NN.txt") runs"
    while read -r bench mode nn gpn rep envs; do
        [ -n "$bench" ] || continue
        # Skip a run that could outlast the job, rather than lose it half-done.
        if [ "$TIME_GUARD" = 1 ]; then
            need=$(env $envs bash -c "source '$LIB_DIR/common.sh'; run_worst_case_s $bench $mode")
            left=$(job_time_left_s)
            if [ "$need" -gt "$left" ]; then
                echo "$(date -Is) job $SLURM_JOB_ID skipped $exp $bench $mode ${nn}x$gpn rep$rep:" \
                    "needs up to ${need}s, ${left}s left" | tee -a "$SKIPPED"
                continue
            fi
        fi
        env $envs "$LIB_DIR/run_bench.sh" "$bench" "$mode" "$nn" "$gpn" "$rep" </dev/null
        # Nothing from a failed run may leak into the next one.
        cleanup_leftovers
    done <"$CAMPAIGN_DIR/plan_${exp}_n$NN.txt"
done

echo "campaign $CAMPAIGN job $SLURM_JOB_ID done; collect with:"
echo "  $EXP_DIR/collect.sh $CAMPAIGN"
