#!/bin/bash
# Reproduce the intermittent teardown crash seen in the 2026-10-07 ablation:
# P2P/CUMEM nccl_allreduce ckpt_resume, rank 0 segfaulting in ncclCommDestroy
# while the peer's service thread was converting cuMem handles over UDS.
#
#   salloc -A m4128 -q interactive -t 00:30:00 -N 1 -C gpu --gpus-per-node=4 \
#       srun -N 1 -n 1 --overlap bash tests/report/repro_cumem.sh
#
# Results: $RESULTS_ROOT/$REPRO/<profile>/runs/... (with core.* on a crash),
# summary in $RESULTS_ROOT/$REPRO/repro_summary.txt.

set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
: "${REPRO:=repro-cumem-$(date +%Y%m%d-%H%M)}"
: "${REPRO_N:=10}"
: "${REPRO_PROFILES:=p2p_cumem default}"
: "${REPRO_BENCH:=nccl_allreduce}"
OUT=$RESULTS_ROOT/$REPRO
mkdir -p "$OUT"

export CORES=1 CKPT_DELAY=5 NB_CKPT_TARGET_S=25 NET_PROFILE=socket
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,P2P,PROXY,ALLOC NCCL_DEBUG_TIMESTAMP_LEVELS=ALL

for i in $(seq 1 "$REPRO_N"); do
    for prof in $REPRO_PROFILES; do
        NCCL_PROFILE=$prof CAMPAIGN=$REPRO/$prof \
            "$REPORT_DIR/run_bench.sh" "$REPRO_BENCH" ckpt_resume 1 2 "$i"
        pkill -9 -u "$USER" -f dmtcp_coordinator 2>/dev/null || true
    done
done

{
    echo "== $REPRO: $REPRO_BENCH ckpt_resume x $REPRO_N per profile, 1x2"
    for prof in $REPRO_PROFILES; do
        csv=$OUT/$prof/results.csv
        ok=$(awk -F, 'NR > 1 && $8 == "ok"' "$csv" | wc -l)
        all=$(awk -F, 'NR > 1' "$csv" | wc -l)
        echo "$prof: $ok/$all ok"
        awk -F, 'NR > 1 && $8 != "ok" {print "  rep " $7 ": " $8 "  " $NF}' "$csv"
    done
    echo "== core files"
    find "$OUT" -name 'core.*' -printf '%s %p\n'
} | tee "$OUT/repro_summary.txt"
