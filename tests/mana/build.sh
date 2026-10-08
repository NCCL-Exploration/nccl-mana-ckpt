#!/bin/bash
# Build an isolated MANA (cuda-aware branch) + the NCCL plugin against MANA's
# DMTCP + the test program, all under $WORK (default $SCRATCH/mana-nccl).
#
#   tests/mana/build.sh
#
# ~/mana is only read (cloned), never modified or rebuilt.
# The NCCL plugin must be compiled against the DMTCP that will load it
# (dmtcp_kvdb_c.cpp uses DMTCP's internal C++ API), so it is rebuilt here from
# this repo's sources against MANA's DMTCP rather than reusing bin/.

set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$HERE/../.." && pwd)
: "${WORK:=$SCRATCH/mana-nccl}"
: "${MANA_SRC:=$HOME/mana}"
: "${MANA_BRANCH:=cuda-aware}"
: "${CUDA_MODULE:=cudatoolkit/13.2}"
: "${NCCL_MODULE:=nccl/2.29.2-cu13}"

module unload darshan >/dev/null 2>&1 || true
module load "$CUDA_MODULE" "$NCCL_MODULE"
mkdir -p "$WORK"

# 1. MANA, cloned from the local repo (its committed $MANA_BRANCH).
if [ ! -d "$WORK/mana/.git" ]; then
    git clone --branch "$MANA_BRANCH" "$MANA_SRC" "$WORK/mana"
fi
if [ ! -f "$WORK/mana/dmtcp/configure" ]; then
    git -C "$WORK/mana" submodule init
    # Take DMTCP from the local copy instead of GitHub; the commit is the one
    # the branch pins. Local-path submodules need protocol.file.allow.
    git -C "$WORK/mana" config submodule.dmtcp.url "$MANA_SRC/dmtcp"
    git -C "$WORK/mana" -c protocol.file.allow=always submodule update
fi
echo "MANA $(git -C "$WORK/mana" rev-parse --short HEAD), DMTCP $(git -C "$WORK/mana/dmtcp" rev-parse --short HEAD)"
# The cuda-aware branch builds its CUDA forwarding shims (mana_launch --cuda)
# from mpi-proxy-split/cuda-forward/Makefile, but that file is not in the repo:
# the root .gitignore's bare "Makefile" pattern (meant for autoconf output)
# also matched it. Without it, build MANA without the shims; this test does
# not need them (its MPI uses host buffers only, so the lower half never
# touches CUDA) and run.sh then loads the DMTCP CUDA plugin explicitly.
(
    cd "$WORK/mana"
    [ -f Makefile ] || ./configure >configure.log 2>&1
    if [ -f mpi-proxy-split/cuda-forward/Makefile ]; then
        make -j 16 >make.log 2>&1 || { tail -n 40 make.log; exit 1; }
    else
        echo "NOTE: mpi-proxy-split/cuda-forward/Makefile missing; building MANA without --cuda support"
        # With cudatoolkit loaded, Cray's cc/CC silently add libcuda.so.1 to
        # everything they link -- including the lower half, which must not
        # carry a CUDA driver of its own. Only the CUDA plugin needs CUDA.
        (
            module unload "$NCCL_MODULE" cudatoolkit >/dev/null 2>&1
            make -j 16 HAVE_CUDA= >make.log 2>&1
        ) || { tail -n 40 make.log; exit 1; }
        make -C dmtcp/plugin/cuda libdmtcp_cuda.so >>make.log 2>&1 || { tail -n 20 make.log; exit 1; }
        cp -f dmtcp/plugin/cuda/libdmtcp_cuda.so lib/dmtcp/
    fi
)
for f in bin/mana_launch bin/mana_restart bin/lower-half lib/dmtcp/libmana.so \
    lib/dmtcp/libdmtcp_cuda.so; do
    [ -e "$WORK/mana/$f" ] || { echo "MANA build is missing $f"; exit 1; }
done
# One CUDA driver per process (CUDA checkpoint API): the lower half must not load its own.
if [ ! -e "$WORK/mana/lib/dmtcp/cuda-forward/libcuda.so.1" ] &&
    readelf -d "$WORK/mana/bin/lower-half" | grep -q 'NEEDED.*libcuda'; then
    echo "ERROR: bin/lower-half links libcuda.so.1 (a second CUDA driver without --cuda shims)." >&2
    echo "       Rebuild MANA without cudatoolkit loaded: make -C $WORK/mana clean, rerun." >&2
    exit 1
fi

# 2. The NCCL plugin, built against MANA's DMTCP, in its own tree.
rm -rf "$WORK/nccl-plugin"
mkdir -p "$WORK/nccl-plugin"
cp -r "$REPO_ROOT/src" "$REPO_ROOT/include" "$REPO_ROOT/Makefile" "$WORK/nccl-plugin/"
make -C "$WORK/nccl-plugin" DMTCP_ROOT="$WORK/mana/dmtcp" plugin >"$WORK/nccl-plugin.log" 2>&1 ||
    { tail -n 30 "$WORK/nccl-plugin.log"; exit 1; }

# 3. The test. MANA's CUDA forwarding needs the application's cudart shared.
nvcc -O2 -arch=sm_80 -ccbin CC -cudart=shared -I"$NCCL_HOME/include" -L"$NCCL_HOME/lib" \
    -Xlinker -rpath="$NCCL_HOME/lib" -o "$WORK/mana_nccl_test" "$HERE/mana_nccl_test.cu" -lnccl

echo "built: $WORK/mana, $WORK/nccl-plugin/bin/libncclplugin.so, $WORK/mana_nccl_test"
