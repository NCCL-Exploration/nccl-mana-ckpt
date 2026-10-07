# NCCL Multi-GPU Tests and Profiling

This directory contains test drivers and profiling scripts for NCCL-CR multi-GPU executions.

## Profiling Collective Overhead on Perlmutter

To measure the runtime overhead introduced by the `nccl-mana-ckpt` wrapper on standard collective functions, a dedicated profiling script is provided.

The profiler tests the following collectives:
- `AllReduce`, `Broadcast`, `Reduce`, `AllGather`, `ReduceScatter`, `AlltoAll`, `Gather`, `Scatter`

### How to Run

On a Perlmutter GPU node (or an interactive `salloc` session with 4 GPUs), simply execute:

```bash
./run_profiling.sh
```

### What it does

1. Compiles `profile_collectives.cpp` against the local CUDA and NCCL modules.
2. Initializes a 4-GPU clique (`ncclCommInitAll`).
3. Allocates 4MB (1M floats) of payload per GPU.
4. **Run 1**: Executes 10 iterations of each collective natively (without the plugin).
5. **Run 2**: Re-executes the exact same suite but forcibly injects `LD_PRELOAD=../../bin/libncclplugin.so`.
6. Outputs the end-to-end average runtime in milliseconds for each collective function, allowing you to directly calculate the overhead delta (which should be well under a microsecond per call).
