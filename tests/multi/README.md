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
2. Initializes a 4-GPU clique using `ncclGroupStart()` and `ncclCommInitRank()`.
3. Allocates 4MB (1M floats) of payload per GPU.
4. **Run 1**: Executes 10 iterations of each collective natively (without the plugin).
5. **Run 2**: Re-executes the exact same suite but forcibly injects the plugins via `dmtcp_launch --with-plugin`.
6. Outputs the end-to-end average runtime in milliseconds for each collective function side-by-side, allowing you to directly calculate the overhead delta (which should be minimal).

### Troubleshooting
- **Bus Error / Crash during Warmup**: This is typically caused by issues initializing communicators across multiple GPUs on a single thread. The script uses explicit `ncclGroupStart()` blocks to avoid these lockups and safely navigate the plugin's state tracking.
- **Missing Symbols**: Ensure that the plugin (`libncclplugin.so`) is compiled. The script will attempt to build it automatically if missing.
