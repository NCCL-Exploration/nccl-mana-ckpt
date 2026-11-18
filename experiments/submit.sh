#!/bin/bash
# Submit the paper experiments: one Slurm job per node count they need.
#
#   experiments/submit.sh [experiment...]        # default: micro scaling torchsave p2p
#   experiments/submit.sh micro                  # E1 only (2 jobs: 1 node, 2 nodes)
#   SMOKE=1 experiments/submit.sh                # short runs, 1 rep, debug QOS
#   DRY_RUN=1 experiments/submit.sh              # print the plan and the sbatch lines
#   experiments/submit.sh --here [experiment...] # inside an allocation you already
#                                                # hold (salloc), in the foreground
#
# All jobs of one invocation share CAMPAIGN (default: today's date, or
# smoke-<date-time> with SMOKE=1), so their rows land in the same
# $RESULTS_ROOT/$CAMPAIGN/results.csv. Then: experiments/collect.sh $CAMPAIGN
#
# Walltime per job: WALLTIME_<nodes> (e.g. WALLTIME_4=02:00:00), else below.
# campaign.sbatch skips (and lists in skipped.txt) runs that would not fit.
# Build first: experiments/build.sh

set -euo pipefail
HERE=0
if [ "${1:-}" = --here ]; then HERE=1; shift; fi
if [ "${SMOKE:-0}" = 1 ]; then
    : "${CAMPAIGN:=smoke-$(date +%Y%m%d-%H%M)}"
    export REPS=${REPS:-1} IS_CLASS=${IS_CLASS:-B} CG_CLASS=${CG_CLASS:-B} MG_CLASS=${MG_CLASS:-B}
    export NB_MAX_SIZE=${NB_MAX_SIZE:-1048576} CKPT_DELAY=${CKPT_DELAY:-5}
    export IS_TARGET_S=10 CG_TARGET_S=10 MG_TARGET_S=10 PT_TARGET_S=10 NB_SIZE_BUDGET_S=0.2
    export IS_CKPT_TARGET_S=25 CG_CKPT_TARGET_S=25 MG_CKPT_TARGET_S=25 PT_CKPT_TARGET_S=25
    export NB_CKPT_TARGET_S=25 TS_WIDTHS=${TS_WIDTHS:-"4096 8192"} TS_REPS=${TS_REPS:-2}
    export MICRO_BENCHES=${MICRO_BENCHES:-"nccl_latency nccl_bw nccl_allreduce nccl_alltoall"}
    export MICRO_CKPT_BENCHES=${MICRO_CKPT_BENCHES:-"nccl_allreduce"}
    : "${QOS:=debug}"
fi
source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"
source "$EXP_DIR/plan.sh"
EXPERIMENTS=${*:-$EXPERIMENTS_ALL}
for e in $EXPERIMENTS; do _in "$e" "$EXPERIMENTS_ALL" || { echo "unknown experiment $e" >&2; exit 2; }; done
export CAMPAIGN EXP_DIR EXPERIMENTS
[ "${DRY_RUN:-0}" = 1 ] || mkdir -p "$CAMPAIGN_DIR/slurm"
# The exact code and settings of the campaign, for the paper's artifact notes.
[ "${DRY_RUN:-0}" = 1 ] || {
    echo "campaign=$CAMPAIGN experiments=$EXPERIMENTS submitted=$(date -Is) smoke=${SMOKE:-0}"
    echo "repo=$(git -C "$REPO_ROOT" rev-parse HEAD) $(git -C "$REPO_ROOT" status --porcelain -- src include experiments tests | wc -l) modified files"
    echo "dmtcp=$(git -C "$DMTCP_ROOT" rev-parse HEAD)"
    (set -o posix; set) | grep -E '^(REPS|CKPT_|NCCL_PROFILE|NET_PROFILE|IS_|CG_|MG_|PT_|NB_|TS_|P2P_|MICRO_|SCALE_)='
} >>"$CAMPAIGN_DIR/campaign.txt"

if [ $HERE = 1 ]; then
    [ -n "${SLURM_JOB_ID:-}" ] || { echo "--here needs an allocation (salloc first)" >&2; exit 2; }
    srun -N 1 -n 1 --overlap bash "$EXP_DIR/campaign.sbatch" 2>&1 |
        tee "$CAMPAIGN_DIR/slurm/here_$SLURM_JOB_ID.out"
    exit "${PIPESTATUS[0]}"
fi

for NN in $(plan_nodes $EXPERIMENTS); do
    v=WALLTIME_$NN
    if [ -n "${!v:-}" ]; then
        T=${!v}
    elif [ "${SMOKE:-0}" = 1 ]; then
        T=00:30:00
    else
        case "$NN" in
        1) T=06:00:00 ;; # micro 1x2, scaling 1x4, torchsave 1x4, p2p
        2) T=06:00:00 ;; # micro 2x1, scaling 2x4, torchsave 2x4, p2p
        *) T=04:00:00 ;;
        esac
    fi
    n=0
    for e in $EXPERIMENTS; do n=$((n + $(plan_runs "$e" "$NN" | wc -l))); done
    cmd=(sbatch -A "$ACCOUNT" -q "$QOS" -t "$T" -N "$NN" --export=ALL
        -o "$CAMPAIGN_DIR/slurm/%x_n${NN}_%j.out" "$EXP_DIR/campaign.sbatch")
    echo "$NN node(s): $n runs, walltime $T"
    if [ "${DRY_RUN:-0}" = 1 ]; then
        for e in $EXPERIMENTS; do plan_runs "$e" "$NN" | sed "s/^/    $e: /"; done
        echo "    ${cmd[*]}"
    else
        "${cmd[@]}"
    fi
done
echo "campaign $CAMPAIGN -> $CAMPAIGN_DIR"
