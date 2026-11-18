#!/bin/bash
# Build everything the experiments need. Safe to run on a login node.
#
#   experiments/build.sh [dmtcp] [plugin] [nas] [ncclbench] [p2ppending] [check]   (default: all)
#
# Binaries go to $RESULTS_ROOT/bin (on $SCRATCH); the plugin to bin/ in the
# repo, DMTCP to dmtcp/bin. Nothing links MPI.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"
STEPS=${*:-dmtcp plugin nas ncclbench p2ppending check}
mkdir -p "$BIN_DIR" "$RESULTS_ROOT/build"

for step in $STEPS; do
    case "$step" in
    dmtcp)
        (
            load_modules nas
            cd "$DMTCP_ROOT"
            [ -f Makefile ] || ./configure
            make -j 16
            # The top-level make does not rebuild the CUDA plugin.
            make -C plugin/cuda
        )
        ;;
    plugin)
        (load_modules nas && make -C "$REPO_ROOT" plugin)
        ;;
    nas)
        (
            load_modules nas
            for b in is cg mg; do
                B=${b^^}
                # Build in a copy so `make clean` never touches the source tree's run files.
                src=$RESULTS_ROOT/build/$b
                rm -rf "$src" && cp -r "$REPO_ROOT/tests/nas-benchmarks/$B" "$src"
                rm -rf "$src"/ckpt_* "$src"/*.log "$src"/*_exe
                for c in $NAS_CLASSES; do
                    make -C "$src" clean >/dev/null
                    make -C "$src" "${b}_nccl" CLASS="$c"
                    cp "$src/${b}_nccl_exe" "$BIN_DIR/${b}_nccl_$c"
                done
            done
        )
        ;;
    ncclbench)
        (
            load_modules nccl
            nvcc -O3 -arch=sm_80 -Xcompiler -Wall -I"$NCCL_HOME/include" -L"$NCCL_HOME/lib" \
                -Xlinker -rpath="$NCCL_HOME/lib" -o "$NCCL_BENCH" \
                "$EXP_DIR/microbench/nccl_bench.cu" -lnccl
            if ldd "$NCCL_BENCH" | grep -qi mpi; then echo "nccl_bench links MPI" >&2; exit 1; fi
        )
        ;;
    p2ppending)
        (
            load_modules nccl
            nvcc -O2 -arch=sm_80 -Xcompiler -Wall -I"$NCCL_HOME/include" -L"$NCCL_HOME/lib" \
                -Xlinker -rpath="$NCCL_HOME/lib" -o "$P2P_PENDING" \
                "$REPO_ROOT/tests/multi/p2p_pending.cu" -lnccl
        )
        ;;
    check)
        ok=1
        for f in "$DMTCP_LAUNCH" "$DMTCP_ROOT/plugin/cuda/libdmtcp_cuda.so" \
            "$REPO_ROOT/bin/libncclplugin.so" "$NCCL_BENCH" "$P2P_PENDING"; do
            [ -e "$f" ] || { echo "MISSING $f"; ok=0; }
        done
        for b in is cg mg; do
            for c in $NAS_CLASSES; do
                [ -x "$BIN_DIR/${b}_nccl_$c" ] || { echo "MISSING ${b}_nccl_$c"; ok=0; }
            done
        done
        (load_modules pytorch && python -c 'import torch; print("torch", torch.__version__)') ||
            { echo "pytorch module unusable"; ok=0; }
        [ $ok = 1 ] && echo "build check: all artifacts present"
        ;;
    *) echo "unknown step $step" >&2; exit 2 ;;
    esac
done
