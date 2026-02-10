#!/bin/bash
echo "Compiling single_gpu_test.cpp..."
nvcc -o single_gpu_test single_gpu_test.cpp -lnccl
echo "Compilation finished."
