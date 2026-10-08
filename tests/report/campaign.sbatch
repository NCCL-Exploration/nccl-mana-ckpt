#!/bin/bash
#SBATCH -C gpu
#SBATCH --gpus-per-node=4
#SBATCH -J nccl-ckpt-report
#SBATCH -o %x_%j.out
#
# One allocation = one node count. Submit through submit_all.sh, which sets
# -N / -t / -A / -q and exports:
#   REPORT_DIR  tests/report in the repo (sbatch copies this file to a spool dir)
#   GPN_LIST    GPUs-per-node values to run at this node count, e.g. "2 4"
#   BENCHES     benchmarks for the runtime-overhead study (native/dmtcp)
#   CKPT_BENCHES benchmarks for the checkpoint/restart study
#   MODES       subset of: native native_ref dmtcp ckpt_resume ckpt_restart

set -uo pipefail
source "$REPORT_DIR/common.sh"

: "${GPN_LIST:=4}"
: "${BENCHES:=is pytorch nccl_latency nccl_bw nccl_bibw nccl_allreduce nccl_allgather nccl_bcast nccl_reduce nccl_reduce_scatter nccl_alltoall}"
: "${CKPT_BENCHES:=is pytorch nccl_bw nccl_allreduce nccl_alltoall}"
: "${MODES:=native native_ref dmtcp ckpt_resume ckpt_restart}"
NN=$SLURM_JOB_NUM_NODES

mkdir -p "$CAMPAIGN_DIR"
"$REPORT_DIR/sysinfo.sh" "$CAMPAIGN_DIR/sysinfo/job$SLURM_JOB_ID"

is_pt2pt() { case "$1" in nccl_latency | nccl_bw | nccl_bibw) return 0 ;; esac; return 1; }
in_list() { case " $2 " in *" $1 "*) return 0 ;; esac; return 1; }

for GPN in $GPN_LIST; do
    NR=$((NN * GPN))
    for BENCH in $BENCHES $CKPT_BENCHES; do
        # BENCHES and CKPT_BENCHES overlap; visit each benchmark once.
        in_list "$BENCH" "${SEEN:-}" && continue
        SEEN="${SEEN:-} $BENCH"
        if is_pt2pt "$BENCH" && [ "$NR" -ne 2 ]; then continue; fi
        # IS needs a power-of-two rank count.
        if [ "$BENCH" = is ] && [ $((NR & (NR - 1))) -ne 0 ]; then continue; fi
        for REP in $(seq 1 "$REPS"); do
            # Modes interleaved within a repetition, so drift hits all equally.
            for MODE in $MODES; do
                case "$MODE" in
                native | native_ref | dmtcp) in_list "$BENCH" "$BENCHES" || continue ;;
                *) in_list "$BENCH" "$CKPT_BENCHES" || continue ;;
                esac
                "$REPORT_DIR/run_bench.sh" "$BENCH" "$MODE" "$NN" "$GPN" "$REP"
                # Nothing from a failed run may leak into the next one.
                pkill -9 -u "$USER" -f dmtcp_coordinator 2>/dev/null || true
            done
        done
    done
    SEEN=
done

echo "campaign $CAMPAIGN job $SLURM_JOB_ID done; summarize with:"
echo "  python3 $REPORT_DIR/summarize.py $CAMPAIGN_DIR"
