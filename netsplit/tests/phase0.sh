#!/bin/bash
# netsplit phase 0a and 0b.  Runs inside an allocation of 2 GPU nodes, as a
# one-node step on the first:
#
#   salloc -A m4128 -q interactive -t 00:45:00 -N 2 -C gpu --gpus-per-node=4 --no-shell
#   srun --jobid=J -N 1 -n 1 --overlap bash netsplit/tests/phase0.sh
#
# 0a  probe_v6 drives aws-ofi-nccl through ncclNet v6 with only the CUDA stub
#     (2 nodes, then 1 node), and with the real CUDA runtime as a control.
# 0b  gdr_ckpt registers GPU memory with libfabric before and after
#     cuda-checkpoint checkpoints and restores the process.
# Output: $OUT/summary.txt and one log per case.

set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
REPO=$(cd "$HERE/.." && pwd)
: "${OUT:=$SCRATCH/netsplit-runs/phase0-${SLURM_JOB_ID:?run inside an allocation}}"
CKPT_TOOL=$REPO/cuda-checkpoint/bin/x86_64_Linux/cuda-checkpoint

module unload darshan vasp >/dev/null 2>&1 || true
module load cudatoolkit/13.2 nccl/2.29.2-cu13 >/dev/null 2>&1
export DARSHAN_DISABLE=1
PLUGIN=$NCCL_HOME/plugin/lib/libnccl-net.so
CUDA_LIBDIR=$CUDA_HOME/lib64

# The driver is a one-node step: give the inner sruns the job's nodes.
JOBNODES=$(squeue -j "$SLURM_JOB_ID" -h -o %N)
NJOBNODES=$(scontrol show hostnames "$JOBNODES" | wc -l)
export SLURM_JOB_NODELIST=$JOBNODES SLURM_NODELIST=$JOBNODES SLURM_NNODES=$NJOBNODES \
    SLURM_JOB_NUM_NODES=$NJOBNODES
unset SLURM_NTASKS SLURM_NPROCS SLURM_TASKS_PER_NODE SLURM_STEP_NODELIST SLURM_STEP_NUM_NODES \
    SLURM_STEP_NUM_TASKS SLURM_STEP_TASKS_PER_NODE SLURM_NTASKS_PER_NODE SLURM_CPUS_PER_TASK \
    SLURM_CPU_BIND SLURM_CPU_BIND_TYPE SLURM_CPU_BIND_LIST

mkdir -p "$OUT"
summary() { echo "$*" | tee -a "$OUT/summary.txt"; }
summary "== $(date -Is) job $SLURM_JOB_ID nodes $JOBNODES"
summary "== plugin $(readlink -f "$PLUGIN")"

# Library path of the lower half: libfabric and the plugin's own directory,
# no CUDA directory (libcuda.so.1 stays reachable through ld.so.cache).
LH_PATH=/opt/cray/libfabric/2.3.1/lib64:$NCCL_HOME/plugin/lib

probe() { # name nodes tasks_per_node use_stub
    local name=$1 nn=$2 tpn=$3 stub=$4 d=$OUT/$1
    rm -rf "$d" && mkdir -p "$d"
    local -a envv=(env LD_LIBRARY_PATH=$LH_PATH)
    local -a args=(-p "$PLUGIN" -d "$d" -s 4194304 -n 200)
    if [ "$stub" = 1 ]; then
        args+=(-c "$HERE/bin/libcudart_stub.so")
    else
        envv=(env LD_LIBRARY_PATH=$LH_PATH:$CUDA_LIBDIR)
    fi
    timeout 300 srun -N "$nn" -n $((nn * tpn)) --ntasks-per-node="$tpn" --gpus-per-node=4 \
        --cpu-bind=none --overlap "${envv[@]}" FI_LOG_LEVEL=warn "$HERE/bin/probe_v6" "${args[@]}" \
        >"$d/log" 2>&1
    local rc=$?
    local pass
    pass=$(grep -c '^RESULT probe_v6 .*PASS' "$d/log")
    summary "$(printf '%-24s rc=%-3s pass=%s/%s  %s' "$name" "$rc" "$pass" $((nn * tpn)) \
        "$(grep -h -E 'MB/s|CUDA-related' "$d/log" | sed -E 's/^(INFO|MAPS) //' | tr '\n' ';' | cut -c1-200)")"
    grep -h '^MAPS' "$d/log" | sort -u | sed 's/^/    /' | tee -a "$OUT/summary.txt" >/dev/null
    grep -h -E 'ptrSupport' "$d/log" | sort -u | head -4 | sed 's/^/    /' | tee -a "$OUT/summary.txt" >/dev/null
}

gdr() { # name FI_HMEM_CUDA_USE_DMABUF
    local name=$1 dmabuf=$2 d=$OUT/$1
    rm -rf "$d" && mkdir -p "$d"
    timeout 300 srun -N 1 -n 1 --gpus-per-node=4 --cpu-bind=none --overlap bash -c '
        export FI_HMEM_CUDA_USE_DMABUF='"$dmabuf"' FI_LOG_LEVEL=warn
        "'"$HERE"'/bin/gdr_ckpt" -d "'"$d"'" &
        pid=$!
        for i in $(seq 600); do [ -f "'"$d"'/ready" ] && break; sleep 0.1; done
        p=$(cat "'"$d"'/ready")
        echo "== checkpoint pid $p: $("'"$CKPT_TOOL"'" --toggle --pid "$p" 2>&1) state=$("'"$CKPT_TOOL"'" --get-state --pid "$p" 2>&1)"
        sleep 2
        echo "== restore pid $p: $("'"$CKPT_TOOL"'" --toggle --pid "$p" 2>&1) state=$("'"$CKPT_TOOL"'" --get-state --pid "$p" 2>&1)"
        touch "'"$d"'/restored"
        wait $pid
        echo "== exit $?"' >"$d/log" 2>&1
    summary "$(printf '%-24s FI_HMEM_CUDA_USE_DMABUF=%s' "$name" "$dmabuf")"
    grep -h -E '^(REG|FAIL|==|INFO domain)' "$d/log" | sed 's/^/    /' | tee -a "$OUT/summary.txt" >/dev/null
}

probe 0a-2node-stub 2 1 1
probe 0a-1node-stub 1 2 1
probe 0a-2node-realcudart 2 1 0
gdr 0b-gdr-iov-peermem 0
gdr 0b-gdr-iov-dmabuf 1
summary "results: $OUT/summary.txt"
