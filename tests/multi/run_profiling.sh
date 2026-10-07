#!/bin/bash
set -e

echo "=========================================================="
echo " NCCL-CR Overhead Profiler (Perlmutter)"
echo "=========================================================="

if [ -n "$NCCL_DIR" ]; then
    NCCL_FLAGS="-I$NCCL_DIR/include -L$NCCL_DIR/lib -Xlinker -rpath=$NCCL_DIR/lib"
elif [ -n "$NCCL_HOME" ]; then
    NCCL_FLAGS="-I$NCCL_HOME/include -L$NCCL_HOME/lib -Xlinker -rpath=$NCCL_HOME/lib"
else
    NCCL_FLAGS="-L/shared/EL9/explorer/nvidia-hpc-sdk/24.7/Linux_x86_64/24.7/REDIST/comm_libs/12.5/nccl/lib"
fi

# Compile the profiler
echo "Compiling profile_collectives.cpp..."
nvcc -O3 -std=c++11 -I../../src $NCCL_FLAGS -lnccl -lcudart profile_collectives.cpp -o profile_collectives

if [ ! -f ../../bin/libncclplugin.so ]; then
    echo "Warning: libncclplugin.so not found in ../../bin/. Building it now..."
    make -C ../.. plugin
fi

DMTCP_LAUNCH="../../dmtcp/bin/dmtcp_launch"
if [ ! -x "$DMTCP_LAUNCH" ]; then
    echo "Warning: dmtcp_launch not found at $DMTCP_LAUNCH. Please ensure DMTCP is built."
    # We will try to rely on PATH if possible
    DMTCP_LAUNCH="dmtcp_launch"
fi

echo ""
echo "=========================================================="
echo " RUN 1: NATIVE NCCL (NO PLUGIN)"
echo "=========================================================="
./profile_collectives

echo ""
echo "=========================================================="
echo " RUN 2: NCCL-CR PLUGIN INTERPOSITION"
echo "=========================================================="
# We must use dmtcp_launch so that the DMTCP symbols (like dmtcp_plugin_disable_ckpt)
# are loaded into the process. We use --interval 0 to prevent actual checkpoints.
$DMTCP_LAUNCH --interval 0 --no-gzip --with-plugin $(realpath ../../bin/libncclplugin.so) ./profile_collectives

echo ""
echo "=========================================================="
echo " Profiling Complete."
echo " Compare the average runtimes above to calculate overhead."
echo "=========================================================="
