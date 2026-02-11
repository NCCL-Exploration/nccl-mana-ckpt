#!/bin/bash
echo "Compiling single_gpu_test.cpp..."
nvcc -lnccl -L/shared/EL9/explorer/nvidia-hpc-sdk/24.7/Linux_x86_64/24.7/REDIST/comm_libs/12.5/nccl/lib -o single_gpu_test single_gpu_test.cpp
echo "Compilation finished."
