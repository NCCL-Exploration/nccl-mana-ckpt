#!/bin/bash
# Record the hardware/software environment for the report.
#
#   sysinfo.sh <out_dir>
#
# Inside an allocation it also runs node_info on every compute node (one task
# per node), since login nodes have different CPUs and no GPUs.

set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
OUT=$1
mkdir -p "$OUT"

node_info() {
    echo "== host: $(hostname)"
    echo "== os: $(. /etc/os-release && echo "$PRETTY_NAME"); kernel $(uname -r)"
    echo "== cpu"
    lscpu | grep -E '^(Model name|Socket\(s\)|Core\(s\) per socket|Thread\(s\) per core|CPU\(s\)|NUMA node\(s\)|CPU max MHz)'
    echo "== memory"
    grep MemTotal /proc/meminfo
    echo "== gpu"
    nvidia-smi --query-gpu=index,name,memory.total,driver_version,pci.bus_id,clocks.max.sm \
        --format=csv 2>&1
    nvidia-smi -q 2>/dev/null | grep -m1 'CUDA Version'
    echo "== gpu topology"
    nvidia-smi topo -m 2>&1
    echo "== network"
    ls /sys/class/net | grep -E '^hsn' | while read -r n; do
        echo "$n speed=$(cat /sys/class/net/$n/speed 2>/dev/null)Mb/s"
    done
    command -v fi_info >/dev/null && fi_info -p cxi 2>/dev/null | grep -E 'domain|fabric' | sort -u
}

if [ "${1:-}" = --node ]; then node_info; exit 0; fi

{
    echo "== date: $(date -Is)"
    echo "== slurm: job=${SLURM_JOB_ID:-none} nodes=${SLURM_JOB_NODELIST:-none}"
    echo "== repo: $(git -C "$REPO_ROOT" rev-parse HEAD) branch $(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD)" \
        "$(git -C "$REPO_ROOT" diff --quiet HEAD -- src include tests || echo '(dirty)')"
    echo "== dmtcp: $(cat "$DMTCP_ROOT/VERSION") $(git -C "$DMTCP_ROOT" describe --always --tags --dirty 2>/dev/null)"
    echo "== plugin: $(stat -c '%y' "$REPO_ROOT/bin/libncclplugin.so" 2>/dev/null) $(md5sum "$REPO_ROOT/bin/libncclplugin.so" 2>/dev/null | cut -d' ' -f1)"

    (
        load_modules is >/dev/null 2>&1
        echo "== modules (is/nccl_bench)"; module -t list 2>&1 | tr '\n' ' '; echo
        echo "== nvcc: $(nvcc --version | tail -n 2 | tr '\n' ' ')"
        echo "== nccl: $(grep -hE '#define NCCL_(MAJOR|MINOR|PATCH|SUFFIX) ' "$NCCL_HOME/include/nccl.h" | awk '{printf "%s ", $3}') ($NCCL_HOME)"
        echo "== nccl net plugin: $(ls "$NCCL_HOME"/plugin/lib 2>/dev/null | tr '\n' ' ')"
        echo "== gcc: $(gcc --version | head -n 1)"
        echo "== libfabric: $(fi_info --version 2>/dev/null | tr '\n' ' ')"
    )
    echo "== nccl_bench: $(md5sum "$NCCL_BENCH" 2>/dev/null | cut -d' ' -f1); links: $(ldd "$NCCL_BENCH" 2>/dev/null | awk '{print $1}' | tr '\n' ' ')"
    (
        load_modules pytorch >/dev/null 2>&1
        echo "== python/torch: $(python -c 'import sys, torch; print(sys.version.split()[0], "torch", torch.__version__, "cuda", torch.version.cuda, "nccl", ".".join(map(str, torch.cuda.nccl.version())))' 2>&1 | tail -n 1)"
    )
    echo "== filesystem: $RESULTS_ROOT"
    df -hT "$RESULTS_ROOT" | tail -n 1
    command -v lfs >/dev/null && lfs getstripe -d "$RESULTS_ROOT" 2>/dev/null
    echo "== config"
    ( set -o posix; set ) | grep -E '^(NCCL_PROFILE|NET_PROFILE|IS_|PT_|NB_|CKPT_|REPS|CAMPAIGN)='
} >"$OUT/software.txt" 2>&1

if [ -n "${SLURM_JOB_ID:-}" ]; then
    srun -N "$SLURM_JOB_NUM_NODES" -n "$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 --gpus-per-node=4 \
        bash "$REPORT_DIR/sysinfo.sh" --node >"$OUT/nodes.txt" 2>&1
else
    node_info >"$OUT/login_node.txt" 2>&1
fi
echo "sysinfo written to $OUT"
