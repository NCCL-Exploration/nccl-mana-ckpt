#!/bin/bash
# Submit the scaling campaign: one job per node count.
#
#   tests/report/submit_all.sh            # full campaign
#   SMOKE=1 tests/report/submit_all.sh    # one 1-node debug job, 1 rep, short runs
#   SMOKE=1 INTERACTIVE=1 tests/report/submit_all.sh
#       same, on the interactive QOS (starts sooner; runs in the foreground --
#       that QOS takes only salloc, so the batch script runs as an srun step)
#   NODES_LIST="1 2" tests/report/submit_all.sh
#
# PLAN=netsplit runs netsplit/TEST_PLAN.md instead: layouts 1x2 and 1x4 (1
# node), 2x1 and 2x4 (2 nodes), 4x4 (4 nodes); modes native (GDR off),
# native_ref (latency and bw at 2x1), ns, ns_resume and ns_restart at two
# checkpoint delays, and the stress modes ns_resume5 and ns_restart3. Its
# smoke test runs 2x1, 1x4 and 2x4 in one 2-node job, every mode once, with
# IS class D and one stress case of each kind:
#   PLAN=netsplit SMOKE=1 INTERACTIVE=1 tests/report/submit_all.sh
# INTERACTIVE=1 without SMOKE runs the campaign's jobs one after another on
# the interactive QOS (at most 4 nodes and 4 hours, 2 jobs per user), in the
# foreground; NODES_LIST picks the node counts.
#
# All jobs of one invocation share CAMPAIGN (default: today's date), so their
# rows land in the same $RESULTS_ROOT/$CAMPAIGN/results.csv.

set -euo pipefail
if [ "${SMOKE:-0}" = 1 ]; then
    prefix=
    [ "${PLAN:-}" = netsplit ] && prefix=ns-
    : "${CAMPAIGN:=${prefix}smoke-$(date +%Y%m%d-%H%M)}"
fi
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
export CAMPAIGN REPORT_DIR
mkdir -p "$CAMPAIGN_DIR/slurm"

if [ "${SMOKE:-0}" = 1 ] && [ "$PLAN" = netsplit ]; then
    # Every benchmark and mode at 2x1, 1x4 and 2x4, both checkpoint delays,
    # IS class D (common.sh), and one stress case of each kind.
    export PLAN REPS=1 NB_MAX_SIZE=1048576 LAYOUTS="2x1 1x4 2x4"
    export PT_TARGET_S=10 NB_SIZE_BUDGET_S=0.2 PT_CKPT_TARGET_S=25 NB_CKPT_TARGET_S=25
    export STRESS_CASES=${STRESS_CASES-"nccl_bw:2x1 pytorch:2x4"}
    if [ "${INTERACTIVE:-0}" = 1 ]; then
        salloc -A "$ACCOUNT" -q interactive -t 01:30:00 -N 2 -C gpu --gpus-per-node=4 \
            -J ns-report-smoke srun -N 1 -n 1 --overlap bash "$REPORT_DIR/campaign.sbatch" \
            2>&1 | tee "$CAMPAIGN_DIR/slurm/smoke_interactive.out"
    else
        sbatch -A "$ACCOUNT" -q debug -t 00:30:00 -N 2 -J ns-report-smoke --export=ALL \
            -o "$CAMPAIGN_DIR/slurm/%x_%j.out" "$REPORT_DIR/campaign.sbatch"
    fi
    exit 0
fi
if [ "${SMOKE:-0}" = 1 ]; then
    export REPS=1 IS_CLASS=B NB_MAX_SIZE=1048576 CKPT_DELAY=5
    export IS_TARGET_S=10 PT_TARGET_S=10 NB_SIZE_BUDGET_S=0.2
    export IS_CKPT_TARGET_S=25 PT_CKPT_TARGET_S=25 NB_CKPT_TARGET_S=25
    export BENCHES=${BENCHES:-"is pytorch nccl_latency nccl_bw nccl_allreduce nccl_alltoall"}
    export CKPT_BENCHES=${CKPT_BENCHES:-"is pytorch nccl_bw nccl_allreduce"}
    if [ "${INTERACTIVE:-0}" = 1 ]; then
        export GPN_LIST=2
        salloc -A "$ACCOUNT" -q interactive -t 00:30:00 -N 1 -C gpu --gpus-per-node=4 \
            -J nccl-ckpt-smoke srun -N 1 -n 1 --overlap bash "$REPORT_DIR/campaign.sbatch" \
            2>&1 | tee "$CAMPAIGN_DIR/slurm/smoke_interactive.out"
    else
        sbatch -A "$ACCOUNT" -q debug -t 00:30:00 -N 1 --export=ALL,GPN_LIST=2 \
            -o "$CAMPAIGN_DIR/slurm/%x_%j.out" "$REPORT_DIR/campaign.sbatch"
    fi
    exit 0
fi

# nodes:GPUs-per-node list:walltime. 2x1 is the inter-node point-to-point case;
# 1x2 / 1x4 are intra-node; 2x4 .. 8x4 are the multi-node scaling points.
# The netsplit plan stops at 4x4 (netsplit/TEST_PLAN.md §2); campaign.sbatch
# picks each layout's benchmarks.
if [ "$PLAN" = netsplit ]; then : "${NODES_LIST:=1 2 4}"; else : "${NODES_LIST:=1 2 4 8}"; fi
export PLAN
for NN in $NODES_LIST; do
    case "$PLAN:$NN" in
    netsplit:1) GPN_LIST="2 4" T=02:30:00 ;;
    *:1) GPN_LIST="2 4" T=04:00:00 ;;
    netsplit:2) GPN_LIST="1 4" T=03:30:00 ;;
    *:2) GPN_LIST="1 4" T=04:00:00 ;;
    netsplit:*) GPN_LIST="4" T=02:30:00 ;;
    *) GPN_LIST="4" T=02:00:00 ;;
    esac
    if [ "${INTERACTIVE:-0}" = 1 ]; then
        GPN_LIST=$GPN_LIST salloc -A "$ACCOUNT" -q interactive -t "$T" -N "$NN" -C gpu \
            --gpus-per-node=4 -J nccl-ckpt-report srun -N 1 -n 1 --overlap \
            bash "$REPORT_DIR/campaign.sbatch" 2>&1 | tee "$CAMPAIGN_DIR/slurm/interactive_n$NN.out"
        continue
    fi
    sbatch -A "$ACCOUNT" -q "$QOS" -t "$T" -N "$NN" --export=ALL,GPN_LIST="$GPN_LIST" \
        -o "$CAMPAIGN_DIR/slurm/%x_%j.out" "$REPORT_DIR/campaign.sbatch"
done
