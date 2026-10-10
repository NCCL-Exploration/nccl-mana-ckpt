#!/bin/bash
# Checkpoints taken while a point-to-point operation lacks its counterpart
# (tests/multi/p2p_pending.cu), under the single-process plugin and under
# netsplit, with resume and restart. Runs inside an allocation, as a one-node
# step; no MPI:
#
#   salloc -A m4128 -q interactive -t 01:00:00 -N 3 -C gpu --gpus-per-node=4 --no-shell
#   srun --jobid=J -N 1 -n 1 --overlap bash tests/multi/run_p2p_pending.sh
#
# Build first: tests/report/build.sh plugin p2ppending.
#
# CASES: "bench:mode:NNxGPN" separated by spaces (default: every scenario at
# one node with the plugin, and recv-first, chain and relay across nodes with
# netsplit, each with resume and restart). bench is p2p_recv_first,
# p2p_recv_first_cold, p2p_send_first, p2p_chain, p2p_relay, p2p_multi,
# p2p_create, p2p_split or p2p_destroy (the scenarios
# of tests/multi/p2p_pending.cu); mode is ckpt_resume, ckpt_restart
# (single-process plugin) or ns_resume, ns_restart (netsplit).
#
# A case passes when tests/report/run_bench.sh reports ok (every rank received
# its sender's data, before and after the checkpoint) and no "[NCCL Drain]"
# line appears: the checkpoint waited for the missing operation instead of
# completing the other side with a dummy message. Results go to
# $SCRATCH/nccl-ckpt-report/p2p-pending-<job>/ (summary.txt, runs/).
#
# NCCL_PLUGIN_P2P_WAIT=5 (shorter than the 20 s window) tests the fallback
# instead; expect drained > 0 there, so read the logs rather than the verdict:
# send-first passes (the drained message is replayed), recv-first completes
# the checkpoint and its receiver gets zeros, with a WARNING.
#
# DMTCP_PLUGINS=<libncclplugin.so>:<libdmtcp_cuda.so> runs the plugin modes
# with another build of the plugin (absolute paths), e.g. the previous one, to
# see what this test catches.

set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPORT=$(cd "$HERE/../report" && pwd)
: "${CAMPAIGN:=p2p-pending-${SLURM_JOB_ID:?run inside an allocation}${TAG:+-$TAG}}"
export CAMPAIGN
# The checkpoint must land inside the window, while one side still waits.
export CKPT_DELAY=${CKPT_DELAY:-1} P2P_WINDOW_S=${P2P_WINDOW_S:-20} REPS=1
# A failure here is usually a lost message, which leaves one rank waiting
# forever; do not wait the report's 300 s for it.
export HANG_GRACE_S=${HANG_GRACE_S:-60}
: "${CASES:=p2p_recv_first:ckpt_resume:1x2 p2p_recv_first:ckpt_restart:1x2
p2p_recv_first_cold:ckpt_resume:1x2 p2p_recv_first_cold:ckpt_restart:1x2
p2p_send_first:ckpt_resume:1x2 p2p_send_first:ckpt_restart:1x2
p2p_chain:ckpt_resume:1x3 p2p_chain:ckpt_restart:1x3
p2p_relay:ckpt_resume:1x3 p2p_relay:ckpt_restart:1x3
p2p_recv_first:ns_resume:2x1 p2p_recv_first:ns_restart:2x1
p2p_recv_first_cold:ns_resume:2x1
p2p_chain:ns_resume:3x1 p2p_chain:ns_restart:3x1
p2p_relay:ns_resume:3x1 p2p_relay:ns_restart:3x1
p2p_multi:ckpt_resume:1x4 p2p_multi:ns_restart:2x4
p2p_create:ckpt_resume:1x2 p2p_split:ckpt_restart:1x2 p2p_destroy:ckpt_resume:1x2
p2p_create:ns_restart:2x4 p2p_split:ns_resume:2x4 p2p_destroy:ns_restart:2x4}"

source "$REPORT/common.sh"
use_job_nodes
mkdir -p "$CAMPAIGN_DIR"
SUMMARY=$CAMPAIGN_DIR/summary.txt
echo "== $(date -Is) job $SLURM_JOB_ID nodes $SLURM_JOB_NODELIST plugin $(git -C "$REPO_ROOT" rev-parse --short HEAD)+wt${DMTCP_PLUGINS:+ DMTCP_PLUGINS=$DMTCP_PLUGINS}" |
    tee -a "$SUMMARY"

fail=0
for c in $CASES; do
    IFS=: read -r bench mode layout <<<"$c"
    nn=${layout%x*} gpn=${layout#*x}
    if [ "$nn" -gt "${#JOB_NODES[@]}" ]; then
        printf '%-16s %-13s %-5s skipped: needs %s nodes\n' "$bench" "$mode" "$layout" "$nn" |
            tee -a "$SUMMARY"
        continue
    fi
    # The netsplit modes measure netsplit with production NCCL settings.
    case "$mode" in ns*) plan=netsplit ;; *) plan=plugin ;; esac
    PLAN=$plan "$REPORT/run_bench.sh" "$bench" "$mode" "$nn" "$gpn" 1 >/dev/null 2>&1
    cleanup_leftovers
    run=$CAMPAIGN_DIR/runs/$bench/n${nn}x$gpn/$mode/rep1
    status=$(sed -n 's/^status=//p' "$run/result.env" 2>/dev/null)
    logs=("$run"/out.log)
    [ -f "$run/restart.log" ] && logs+=("$run/restart.log")
    drained=$(cat "${logs[@]}" 2>/dev/null | grep -ac '^\[NCCL Drain\]')
    caught=$(cat "${logs[@]}" 2>/dev/null | grep -ac 'while catching up')
    ckpt=$(sed -n 's/^ckpt_s=//p' "$run/result.env" 2>/dev/null)
    verdict=PASS
    if [ "${status:-none}" != ok ] || [ "$drained" -ne 0 ]; then
        verdict=FAIL
        fail=1
    fi
    printf '%-16s %-13s %-5s %-4s status=%-18s drained=%s caught-up-ranks=%s ckpt=%ss %s\n' \
        "$bench" "$mode" "$layout" "$verdict" "${status:-none}" "$drained" "$caught" "${ckpt:--}" \
        "$( [ $verdict = FAIL ] && head -c 160 "$run/first_error.txt" 2>/dev/null)" | tee -a "$SUMMARY"
done
echo "results: $SUMMARY"
exit $fail
