# The experiments of the paper, as lists of runs. Sourced, not executed.
#
#   plan_runs <experiment> <nodes>
#
# prints one line per run that a job with <nodes> nodes does for <experiment>:
#   <bench> <mode> <nodes> <gpus_per_node> <rep> [VAR=value ...]
# The VAR=value words are environment overrides for that run only. Within a
# repetition the modes are interleaved, so drift hits all of them equally.
# README.md describes each experiment; change the lists here, not there.

# E1 micro: NCCL microbenchmarks on 2 GPUs, intra-node (1x2) and inter-node
# (2x1), native vs. under DMTCP + plugin, and through a checkpoint.
MICRO_LAYOUTS=${MICRO_LAYOUTS:-"1x2 2x1"}
MICRO_BENCHES=${MICRO_BENCHES:-"nccl_latency nccl_bw nccl_bibw nccl_allreduce nccl_allgather nccl_reduce_scatter nccl_bcast nccl_reduce nccl_alltoall"}
MICRO_MODES=${MICRO_MODES:-"native native_ref dmtcp"}
MICRO_CKPT_BENCHES=${MICRO_CKPT_BENCHES:-"nccl_bw nccl_allreduce nccl_alltoall"}
MICRO_CKPT_MODES=${MICRO_CKPT_MODES:-"ckpt_resume ckpt_restart"}

# E2 scaling: full transparent checkpoint, resume and restart of applications
# on every GPU of 1, 2 and 4 nodes (4 GPUs each).
SCALE_LAYOUTS=${SCALE_LAYOUTS:-"1x4 2x4 4x4"}
SCALE_BENCHES=${SCALE_BENCHES:-"is cg mg pytorch"}
# native_ref (OFI, GDR) is the reference for the cost of the socket transport.
SCALE_MODES=${SCALE_MODES:-"native native_ref dmtcp ckpt_resume ckpt_restart"}

# E3 torchsave: torch.save / torch.load of the PyTorch model at TS_WIDTHS, on
# the layouts of E2, plus a checkpointed PyTorch job at each width other than
# PT_WIDTH (E2 has that one), so every torch.save point has its DMTCP twin.
TS_LAYOUTS=${TS_LAYOUTS:-"1x4 2x4 4x4"}
TS_CKPT_MODES=${TS_CKPT_MODES:-"ckpt_resume ckpt_restart"}

# E4 p2p: checkpoints while a point-to-point operation lacks its counterpart
# (the per-pair clock of the checkpoint cut). Once each, with a checkpoint 1 s
# into the program's 20 s wait; correct when every receiver gets its sender's
# data and no "[NCCL Drain]" line appears.
P2P_CASES=${P2P_CASES:-"p2p_recv_first:1x2 p2p_recv_first_cold:1x2 p2p_send_first:1x2
p2p_chain:1x3 p2p_relay:1x3 p2p_multi:1x4 p2p_create:1x2 p2p_split:1x2 p2p_destroy:1x2
p2p_recv_first:2x1 p2p_send_first:2x1 p2p_chain:2x2 p2p_multi:2x2"}
P2P_MODES=${P2P_MODES:-"ckpt_resume ckpt_restart"}
P2P_ENV="CKPT_DELAY=1 HANG_GRACE_S=60"

EXPERIMENTS_ALL="micro scaling torchsave p2p"

_in() { case " $2 " in *" $1 "*) return 0 ;; esac; return 1; }

# _grid <nodes> <layouts> <benches> <modes> [env...]: every bench x mode at
# each layout with this node count, REPS times, modes interleaved per rep.
_grid() {
    local nn=$1 layouts=$2 benches=$3 modes=$4 l b m r gpn
    shift 4
    for l in $layouts; do
        [ "${l%x*}" = "$nn" ] || continue
        gpn=${l#*x}
        for b in $benches; do
            layout_ok "$b" "$nn" "$gpn" || continue
            for r in $(seq 1 "$REPS"); do
                for m in $modes; do echo "$b $m $nn $gpn $r $*"; done
            done
        done
    done
}

plan_runs() {
    local exp=$1 nn=$2 c b l w ws=
    case "$exp" in
    micro)
        _grid "$nn" "$MICRO_LAYOUTS" "$MICRO_BENCHES" "$MICRO_MODES"
        _grid "$nn" "$MICRO_LAYOUTS" "$MICRO_CKPT_BENCHES" "$MICRO_CKPT_MODES"
        ;;
    scaling)
        _grid "$nn" "$SCALE_LAYOUTS" "$SCALE_BENCHES" "$SCALE_MODES"
        ;;
    torchsave)
        _grid "$nn" "$TS_LAYOUTS" torchsave native
        for w in $TS_WIDTHS; do [ "$w" = "$PT_WIDTH" ] || ws+=" pytorch_w$w"; done
        [ -n "$ws" ] && _grid "$nn" "$TS_LAYOUTS" "$ws" "$TS_CKPT_MODES"
        ;;
    p2p)
        for c in $P2P_CASES; do
            b=${c%%:*} l=${c#*:}
            [ "${l%x*}" = "$nn" ] || continue
            for m in $P2P_MODES; do echo "$b $m $nn ${l#*x} 1 $P2P_ENV"; done
        done
        ;;
    *) echo "unknown experiment $exp (one of: $EXPERIMENTS_ALL)" >&2; return 1 ;;
    esac
}

# plan_nodes <experiments...>: the node counts any of them needs, ascending.
plan_nodes() {
    local e n out=
    for n in 1 2 3 4 8 16; do
        for e in "$@"; do
            if [ -n "$(plan_runs "$e" "$n")" ]; then out+=" $n"; break; fi
        done
    done
    echo $out
}
