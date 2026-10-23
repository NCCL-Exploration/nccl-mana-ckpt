#!/bin/bash
# Build everything the campaign needs. Safe to run on a login node.
#
#   tests/report/build.sh [dmtcp] [plugin] [is] [ncclbench] [p2ppending] [check]   (default: all)
#
# Binaries go to $RESULTS_ROOT/bin (on $SCRATCH). Nothing links MPI.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
: "${IS_CLASSES:=B C}"
: "${CG_CLASSES:=B C}"
: "${MG_CLASSES:=B C}"
STEPS=${*:-dmtcp plugin is cg mg ncclbench p2ppending check}
mkdir -p "$BIN_DIR" "$RESULTS_ROOT/build"

for step in $STEPS; do
    case "$step" in
    dmtcp)
        if [ ! -x "$DMTCP_LAUNCH" ] || [ ! -f "$DMTCP_ROOT/plugin/cuda/libdmtcp_cuda.so" ]; then
            (
                load_modules is
                cd "$DMTCP_ROOT"
                [ -f Makefile ] || ./configure
                make -j 16
            )
        fi
        ;;
    plugin)
        (load_modules is && make -C "$REPO_ROOT" plugin)
        ;;
    is)
        (
            load_modules is
            # Build in a copy so `make clean` never touches the source tree's run files.
            src=$RESULTS_ROOT/build/is
            rm -rf "$src" && cp -r "$REPO_ROOT/tests/nas-benchmarks/IS" "$src"
            rm -rf "$src"/ckpt_rank_* "$src"/*.log
            for c in $IS_CLASSES; do
                make -C "$src" clean >/dev/null
                make -C "$src" is_nccl CLASS="$c"
                cp "$src/is_nccl_exe" "$BIN_DIR/is_nccl_$c"
            done
        )
        ;;
    cg)
        (
            load_modules cg
            src=$RESULTS_ROOT/build/cg
            rm -rf "$src" && cp -r "$REPO_ROOT/tests/nas-benchmarks/CG" "$src"
            rm -rf "$src"/ckpt_rank_* "$src"/*.log
            for c in $CG_CLASSES; do
                make -C "$src" clean >/dev/null
                make -C "$src" cg_nccl CLASS="$c"
                cp "$src/cg_nccl_exe" "$BIN_DIR/cg_nccl_$c"
            done
        )
        ;;
    mg)
        (
            load_modules mg
            src=$RESULTS_ROOT/build/mg
            rm -rf "$src" && cp -r "$REPO_ROOT/tests/nas-benchmarks/MG" "$src"
            rm -rf "$src"/ckpt_rank_* "$src"/*.log
            for c in $MG_CLASSES; do
                make -C "$src" clean >/dev/null
                make -C "$src" mg_nccl CLASS="$c"
                cp "$src/mg_nccl_exe" "$BIN_DIR/mg_nccl_$c"
            done
        )
        ;;
    ncclbench)
        (
            load_modules nccl
            nvcc -O3 -arch=sm_80 -Xcompiler -Wall -I"$NCCL_HOME/include" -L"$NCCL_HOME/lib" \
                -Xlinker -rpath="$NCCL_HOME/lib" -o "$NCCL_BENCH" \
                "$REPORT_DIR/nccl_bench/nccl_bench.cu" -lnccl
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
        for c in $IS_CLASSES; do [ -x "$BIN_DIR/is_nccl_$c" ] || { echo "MISSING is_nccl_$c"; ok=0; }; done
        for c in $CG_CLASSES; do [ -x "$BIN_DIR/cg_nccl_$c" ] || { echo "MISSING cg_nccl_$c"; ok=0; }; done
        for c in $MG_CLASSES; do [ -x "$BIN_DIR/mg_nccl_$c" ] || { echo "MISSING mg_nccl_$c"; ok=0; }; done
        (load_modules pytorch && python -c 'import torch; print("torch", torch.__version__)') ||
            { echo "pytorch module unusable"; ok=0; }
        [ $ok = 1 ] && echo "build check: all artifacts present"
        ;;
    *) echo "unknown step $step" >&2; exit 2 ;;
    esac
done
