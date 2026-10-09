#!/bin/bash
# Per-task wrapper run by srun on every rank.
#
#   launch_rank.sh native           <ckpt_root> -- cmd args...
#   launch_rank.sh dmtcp            <ckpt_root> -- cmd args...
#   launch_rank.sh restart          <ckpt_root>
#   launch_rank.sh netsplit         <ckpt_root> -- cmd args...
#   launch_rank.sh netsplit-restart <ckpt_root>
#
# dmtcp: each rank checkpoints into <ckpt_root>/r<SLURM_PROCID>, so a restart
# can hand every srun task exactly the images it owns.
# netsplit: nccl_launch runs the program as netsplit-lh's upper half, and the
# netsplit plugin puts rank r's images in <ckpt_root>/ckpt_rank_<r>, where
# nccl_restart --restartdir looks for them.
# A restart srun must use the same -N / --ntasks-per-node / -w as the launch
# so ranks land on the same node and GPU. Every mode records where the rank
# runs in <ckpt_root>/hosts/{launch,restart}.<rank>, and run_bench.sh checks
# that a restart put every rank back where it was.
# COORD_HOST / COORD_PORT / PLUGIN / DMTCP_* / NCCL_LAUNCH / NCCL_RESTART come
# from the environment.

mode=$1
ckroot=$2
shift 2
[ "${1:-}" = "--" ] && shift

export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID
export WORLD_SIZE=$SLURM_NTASKS
mydir=$ckroot/r$RANK

case "$mode" in
restart | netsplit-restart) phase=restart ;;
*) phase=launch ;;
esac
mkdir -p "$ckroot/hosts"
echo "$(hostname) local=$LOCAL_RANK gpus=${CUDA_VISIBLE_DEVICES:-}" >"$ckroot/hosts/$phase.$RANK"

case "$mode" in
native)
    exec "$@"
    ;;
dmtcp)
    mkdir -p "$mydir"
    plug=()
    [ -n "$PLUGIN" ] && plug=(--with-plugin "$PLUGIN")
    exec "$DMTCP_LAUNCH" -j -h "$COORD_HOST" -p "$COORD_PORT" --interval 0 --no-gzip \
        --ckptdir "$mydir" "${plug[@]}" "$@"
    ;;
restart)
    shopt -s nullglob
    imgs=("$mydir"/ckpt_*.dmtcp)
    if [ ${#imgs[@]} -eq 0 ]; then
        echo "launch_rank: rank $RANK has no image in $mydir" >&2
        exit 1
    fi
    exec "$DMTCP_RESTART" -j -h "$COORD_HOST" -p "$COORD_PORT" "${imgs[@]}"
    ;;
netsplit | netsplit-restart)
    # netsplit-lh's restore takes the coordinator from the environment only.
    export DMTCP_COORD_HOST=$COORD_HOST DMTCP_COORD_PORT=$COORD_PORT
    if [ "$mode" = netsplit ]; then
        exec "$NCCL_LAUNCH" --interval 0 --ckptdir "$ckroot" "$@"
    fi
    exec "$NCCL_RESTART" --restartdir "$ckroot"
    ;;
*)
    echo "launch_rank: unknown mode $mode" >&2
    exit 2
    ;;
esac
