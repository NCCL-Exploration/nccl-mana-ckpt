#!/bin/bash
# What a checkpoint does when a point-to-point operation's counterpart does not
# come (tests/multi/p2p_pending.cu, src/libncclplugin.c sequence_consensus and
# refuse_zero_fill). Runs inside an allocation, as a one-node step, like
# run_p2p_pending.sh:
#
#   salloc -A m4128 -q interactive -t 00:30:00 -N 2 -C gpu --gpus-per-node=4 --no-shell
#   srun --jobid=J -N 1 -n 1 --overlap bash tests/multi/run_p2p_policy.sh
#
# Each case states what must happen:
#   wait          no limit (the default): the checkpoint waits for the 45 s
#                 window and reports what is owed after 30 s, then passes;
#   limit-send    NCCL_PLUGIN_P2P_WAIT=5, pending send: drained into host memory,
#                 replayed, passes;
#   limit-recv    NCCL_PLUGIN_P2P_WAIT=5, pending receive: every rank aborts with
#                 an ERROR naming it;
#   zero-fill     the same with NCCL_PLUGIN_P2P_ZERO_FILL=1: the checkpoint
#                 completes, the receiver gets zeros, with a WARNING;
#   off-send      NCCL_PLUGIN_P2P_WAIT=0 (no catch-up), pending send: replayed;
#   off-recv      NCCL_PLUGIN_P2P_WAIT=0, pending receive: aborts.
# Results go to $SCRATCH/nccl-ckpt-report/p2p-policy-<job>/summary.txt.

set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPORT=$(cd "$HERE/../report" && pwd)
: "${CAMPAIGN:=p2p-policy-${SLURM_JOB_ID:?run inside an allocation}${TAG:+-$TAG}}"
export CAMPAIGN CKPT_DELAY=1 REPS=1 HANG_GRACE_S=60
source "$REPORT/common.sh"
use_job_nodes
mkdir -p "$CAMPAIGN_DIR"
SUMMARY=$CAMPAIGN_DIR/summary.txt
echo "== $(date -Is) job $SLURM_JOB_ID nodes $SLURM_JOB_NODELIST plugin $(git -C "$REPO_ROOT" rev-parse --short HEAD)+wt" |
    tee -a "$SUMMARY"

fail=0
# case <name> <bench> <mode> <layout> <expect> [ENV=VALUE ...]
#   expect: pass-waited | pass-drained | abort | zero-fill
case_() {
    local name=$1 bench=$2 mode=$3 layout=$4 expect=$5 plan=plugin run logs status verdict=PASS
    shift 5
    case "$mode" in ns*) plan=netsplit ;; esac
    run=$CAMPAIGN_DIR/runs/$name
    (
        export PLAN=$plan P2P_WINDOW_S=20 "$@"
        "$REPORT/run_bench.sh" "$bench" "$mode" "${layout%x*}" "${layout#*x}" 1 >/dev/null 2>&1
    )
    cleanup_leftovers
    rm -rf "$run" && mkdir -p "$(dirname "$run")" &&
        mv "$CAMPAIGN_DIR/runs/$bench/n$layout/$mode/rep1" "$run"
    status=$(sed -n 's/^status=//p' "$run/result.env" 2>/dev/null)
    logs=$(cat "$run"/out.log "$run"/restart.log 2>/dev/null)
    has() { grep -aqE "$1" <<<"$logs"; }
    case "$expect" in
    pass-waited) [ "$status" = ok ] && has 'checkpoint still waiting' && ! has '^\[NCCL Drain\]' ;;
    pass-drained) [ "$status" = ok ] && has '^\[NCCL Drain\].*draining recv' ;;
    abort) [ "$status" != ok ] && has 'never matched by a send' && has 'posted a receive of [0-9]+ B' &&
        ! has '^\[NCCL Drain\]' ;;
    zero-fill) has 'gets zeros' && has 'FAIL recv from rank 0' && ! has 'never matched by a send' ;;
    esac || { verdict=FAIL; fail=1; }
    printf '%-11s %-15s %-12s %-5s %-4s expect=%-13s status=%s %s\n' "$name" "$bench" "$mode" "$layout" \
        "$verdict" "$expect" "${status:-none}" "$*" | tee -a "$SUMMARY"
}

case_ wait p2p_recv_first ckpt_resume 1x2 pass-waited P2P_WINDOW_S=45
case_ limit-send p2p_send_first ns_resume 2x1 pass-drained NCCL_PLUGIN_P2P_WAIT=5
case_ limit-recv p2p_recv_first ckpt_resume 1x2 abort NCCL_PLUGIN_P2P_WAIT=5
case_ zero-fill p2p_recv_first ckpt_resume 1x2 zero-fill NCCL_PLUGIN_P2P_WAIT=5 NCCL_PLUGIN_P2P_ZERO_FILL=1
case_ off-send p2p_send_first ns_resume 2x1 pass-drained NCCL_PLUGIN_P2P_WAIT=0
case_ off-recv p2p_recv_first ns_resume 2x1 abort NCCL_PLUGIN_P2P_WAIT=0
echo "results: $SUMMARY"
exit $fail
