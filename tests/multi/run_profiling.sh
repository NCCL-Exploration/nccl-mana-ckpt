#!/bin/bash
set -e

echo "=========================================================="
echo " NCCL-CR Overhead Profiler (Perlmutter)"
echo "=========================================================="

# Ensure modules are loaded if needed (Perlmutter defaults)
# module load cudatoolkit || true
# module load nccl || true

# Compile the profiler
echo "Compiling profile_collectives.cpp..."
nvcc -O3 -std=c++11 -I../../src -lnccl -lcudart profile_collectives.cpp -o profile_collectives

if [ ! -f ../../bin/libncclplugin.so ]; then
    echo "Warning: libncclplugin.so not found in ../../bin/. Building it now..."
    make -C ../.. plugin
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
LD_PRELOAD=../../bin/libncclplugin.so ./profile_collectives

echo ""
echo "=========================================================="
echo " Profiling Complete."
echo " Compare the average runtimes above to calculate overhead."
echo "=========================================================="
