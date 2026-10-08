#!/bin/bash
# Per-task wrapper run by srun on every rank.
#
#   launch_rank.sh native  <ckpt_root> -- cmd args...
#   launch_rank.sh dmtcp   <ckpt_root> -- cmd args...
#   launch_rank.sh restart <ckpt_root>
#
# Each rank checkpoints into <ckpt_root>/r<SLURM_PROCID>, so a restart can hand
# every srun task exactly the images it owns. A restart srun must use the same
# -N / --ntasks-per-node as the launch so ranks land on the same node and GPU.
# COORD_HOST / COORD_PORT / PLUGIN / DMTCP_* come from the environment.

mode=$1
ckroot=$2
shift 2
[ "${1:-}" = "--" ] && shift

export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID
export WORLD_SIZE=$SLURM_NTASKS
mydir=$ckroot/r$RANK

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
*)
    echo "launch_rank: unknown mode $mode" >&2
    exit 2
    ;;
esac
