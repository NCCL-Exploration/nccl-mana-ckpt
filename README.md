# nccl-mana-ckpt

An NCCL checkpointing plugin for [DMTCP](https://github.com/dmtcp/dmtcp).

The plugin interposes NCCL calls, tracks communicators, CUDA streams, sequence
progress, and in-flight point-to-point messages, then drains or synchronizes
that state before DMTCP checkpoints the process.

## Prerequisites

| Dependency | Notes |
|------------|-------|
| NVIDIA GPU + drivers | Required for CUDA and NCCL |
| CUDA Toolkit (`nvcc`) | Used as `CC` and `CXX` in the Makefiles |
| NCCL | Library paths are hardcoded in Makefiles; update them for your system |
| DMTCP | Defaults to the `dmtcp/` submodule, or set `DMTCP_ROOT` |
| MPI | Needed for multi-process tests and examples |
| Build tools | `make`, `gcc` / `g++`, `git` |

### Local Paths

The root `Makefile`, `tests/single/Makefile`, and `tests/multi/Makefile` contain
hardcoded NCCL `-L` paths. Update those paths before building if NCCL is
installed somewhere else.

By default:

```bash
DMTCP_ROOT=./dmtcp
```

For external installs, pass or export the roots:

```bash
export DMTCP_ROOT=~/dmtcp
```

## Quick Start

1. Clone the repository and initialize the submodules:

```bash
git clone git@github.com:NCCL-Exploration/nccl-mana-ckpt.git
cd nccl-mana-ckpt
git submodule update --init --recursive
```

2. Build DMTCP (which automatically builds its internal plugins):

```bash
cd dmtcp
./configure
make -j
cd ..
```

3. Build the NCCL plugin:

```bash
make
```

This produces the shared library at `bin/libncclplugin.so`.

4. Run the root single-GPU check:

```bash
make nccl_check
```

`nccl_check` builds `tests/single/single_gpu_test` and launches it under DMTCP with `bin/libncclplugin.so`.

## Plugin Loading

The plugin sets `NCCL_RAS_ENABLE=0` in every process it is loaded into, unless
the variable is already set. NCCL's RAS service (2.24+) keeps TCP connections
between processes that outlive every communicator, and DMTCP can deadlock
draining them in a multi-node checkpoint. If you set `NCCL_RAS_ENABLE` to a
non-zero value, the plugin keeps it and prints a warning.

CUDA/NCCL checkpoint and restart flows normally require both plugins:

1. `bin/libncclplugin.so`: this repository's NCCL checkpoint plugin.
2. `dmtcp/plugin/cuda/libdmtcp_cuda.so`: The CUDA plugin for GPU memory
   checkpoint/restore.

DMTCP accepts multiple plugins as a colon-separated list:

```bash
${DMTCP_ROOT}/bin/dmtcp_launch \
  --with-plugin $PWD/bin/libncclplugin.so:${DMTCP_ROOT}/plugin/cuda/libdmtcp_cuda.so \
  ./your_test
```

The focused test Makefiles under `tests/single/` and `tests/multi/` use the
NCCL+CUDA plugin pair. Some root `Makefile` check targets currently launch only
the NCCL plugin; use the focused test Makefiles or adjust `--with-plugin` when
you need full CUDA checkpoint/restart coverage.

## Running Tests

### Single-GPU Driver

```bash
cd tests
make DMTCP_ROOT=../dmtcp
make test_all DMTCP_ROOT=../dmtcp
```

Runnable criteria in `tests/Makefile`:

```bash
make test1 DMTCP_ROOT=../dmtcp  # Basic P2P, no checkpoint
make test2 DMTCP_ROOT=../dmtcp  # Checkpoint after completion
make test3 DMTCP_ROOT=../dmtcp  # Checkpoint before NCCL init
make test4 DMTCP_ROOT=../dmtcp  # In-transit message checkpoint
```

`test_all` depends on the runnable criteria above and restarts from any
checkpoint file produced during each run.

### Manual Single-GPU Launch

```bash
cd tests
nvcc -O3 -g -lnccl -L/path/to/nccl/lib -I../include \
  -o single_gpu_test single_gpu_test.c

${DMTCP_ROOT}/bin/dmtcp_launch --coord-port 7781 --interval 10 \
  --with-plugin $PWD/../bin/libncclplugin.so:${DMTCP_ROOT}/plugin/cuda/libdmtcp_cuda.so \
  ./single_gpu_test
```

Restart from the newest checkpoint:

```bash
CKPT=$(ls -t ckpt_*.dmtcp 2>/dev/null | head -n 1)
${DMTCP_ROOT}/bin/dmtcp_restart --coord-port 7781 "$CKPT"
```

### Multi-GPU Tests

```bash
cd tests/multi
make test_all DMTCP_ROOT=../../dmtcp
```

Useful overrides:

```bash
make test4 MSGS=8 SIZE=4096 NGPUS=4
```

The P2P gather correctness entrypoint compares `ncclGather_p2p` against
`ncclAllGather`:

```bash
make gather_p2p_test GATHER_COUNT=16 GATHER_ROOT=0 NGPUS=4
```

The P2P scatter correctness entrypoint checks `ncclScatter_p2p` against the
deterministic rank-slot input pattern. This does not compare against regular
`ncclScatter` because the NCCL copy used by this project does not provide that
symbol.

```bash
make scatter_p2p_test SCATTER_COUNT=16 SCATTER_ROOT=0 NGPUS=4
```

### Cleanup

```bash
make kill
make clean
```

Run cleanup from the directory containing the Makefile whose tests you ran.

### Running on NERSC Perlmutter

We recommend configuring the following modules (note, other CUDA and NCCL versions may work):

```bash
module load nccl/2.29.2-cu13
```

This will also load in the ``cudatoolkit/13.0`` module. Additionally, unload Perlmutter specific modules:

```bash
module unload darshan
```

To ensure you have configured the environment correctly, you can run the following, which will checkpoint and resume, while providing a DMTCP restart script.

```bash
./tests/nas-benchmarks/IS/run_perlmutter_ckpt.sh
```

### Running on AICR

We recommend configuring the following modules (note, other CUDA and NCCL versions may work):

```bash
module load neu/StdEnv
module load gcc/12.5.0
module load zlib-ng/2.2.4-gcc-12.5.0
module load cuda/13.1.1
module load openmpi/4.1.8-gcc-12.5.0-cuda-13.1.1
module load NCCL/2.30.7
```

Lastly, the user can run the following checkpoint test script,
which automatically configures the above and loads a checkpointing test:
```bash
sbatch $HOME/nccl-mana-ckpt/tests/nas-benchmarks/IS/run_aicr_ckpt.sh
```

## Project Structure

```text
nccl-mana-ckpt/
|-- include/
|   `-- nccl_wrappers.h
|-- src/
|   |-- libncclplugin.c
|   |-- nccl_wrappers.c
|   |-- nccl_iprobe.c
|   |-- dmtcp_kvdb_c.cpp
|   |-- dmtcp_kvdb_c.h
|   |-- inflight.c
|   `-- inflight.h
|-- tests/
|   |-- Makefile
|   |-- Makefile.common
|   |-- single/
|   |   |-- Makefile
|   |   |-- nccl_test_driver.cpp
|   |   |-- single_gpu_test.c
|   |   |-- multiproc.c
|   |   |-- send_recv.c
|   |   |-- seq_algorithm_test.c
|   |   |-- seq_stubs.c
|   |   |-- simple_init_test.cpp
|   |   `-- test_drain.cpp
|   |-- multi/
|   |   |-- Makefile
|   |   |-- nccl_test_driver_multiple.cpp
|   |   |-- nccl_p2p_tests.cpp
|   |   |-- pytorch_ddp_test.py
|   |   |-- run_perlmutter_pytorch.sh
|   |   |-- test_drain.cpp
|   |   |-- cc_simple.c
|   |   |-- cc_complex.c
|   |   |-- drain_send.c
|   |   `-- drain_recv.c
|   `-- nas-benchmarks/
|-- dmtcp/
|-- Makefile
|-- README.md
`-- AGENTS.md
```

## How It Works

`src/libncclplugin.c` registers DMTCP event hooks:

- `DMTCP_EVENT_PRESUSPEND`: mark checkpoint pending, coordinate collective
  sequence progress and point-to-point counters across ranks (a rank that owes
  a peer a send or receive runs until it posts it, so pending transfers
  complete with their real data), synchronize tracked CUDA streams, drain
  whatever is still unmatched, and destroy the communicators.
- `DMTCP_EVENT_RESUME`: clear checkpoint state after a checkpoint and reset
  wrapper/in-flight tracking for continued execution.
- `DMTCP_EVENT_RESTART`: reset runtime state after restart while preserving
  drained messages that must be replayed through later `ncclRecv` calls.

`src/nccl_wrappers.c` interposes NCCL calls with `dlsym(RTLD_NEXT, ...)`.
Current wrappers cover `ncclCommInitRank`, `ncclCommInitAll`, `ncclSend`,
`ncclRecv`, `ncclAllReduce`, `ncclBroadcast`, `ncclReduce`, `ncclAllGather`,
`ncclGather`, `ncclScatter`, `ncclReduceScatter`, `ncclGather_p2p`, and
`ncclScatter_p2p`. Native-symbol availability still depends on the installed
NCCL library; the NCCL copy used here does not export `ncclScatter`.

`src/inflight.c` tracks local pending bytes by source/destination rank and
stores drained host-side messages for replay. `src/dmtcp_kvdb_c.cpp` provides
the local key-value interface used for checkpoint-time coordination.

## Troubleshooting

### `mmap failed: Invalid argument` on Restart

Load both the NCCL plugin and the split-CUDA plugin. Without the CUDA plugin, DMTCP may
try to checkpoint GPU-mapped memory directly.

```bash
${DMTCP_ROOT}/bin/dmtcp_launch \
  --with-plugin $PWD/bin/libncclplugin.so:${DMTCP_ROOT}/plugin/cuda/libdmtcp_cuda.so \
  ./your_test
```

### NCCL Library Not Found

Find `libnccl.so` and update the hardcoded `-L` path in the Makefile you are
using:

```bash
find / -name "libnccl.so*" 2>/dev/null
```

### Coordinator Connection Failures

Stop stale coordinators and confirm the port is free:

```bash
${DMTCP_ROOT}/bin/dmtcp_command --quit --coord-port 7781 2>/dev/null || true
ss -tlnp | grep 7781
```

## Benchmark Results (NERSC Perlmutter)

### Runtime Overhead Without a Checkpoint

Mean ± sample standard deviation over three runs (64 MiB for `nccl_*` kernels). Positive overhead indicates slower execution under DMTCP with NCCL-CR; negative indicates faster execution.

| Benchmark | Layout | Metric | Native | DMTCP | Overhead |
| --- | --- | --- | --- | --- | --- |
| NAS IS (C) | 1x4 | Mop/s | 913.09 ± 11.53 | 886.21 ± 59.66 | +3.0% |
| NAS IS (C) | 2x4 | Mop/s | 1272.32 ± 5.92 | 1272.31 ± 18.32 | +0.0% |
| NAS IS (C) | 4x4 | Mop/s | 2541.85 ± 28.94 | 2576.28 ± 50.73 | -1.3% |
| NAS CG (C) | 1x4 | Mop/s | 7497.19 ± 11.91 | 7441.35 ± 151.56 | +0.8% |
| NAS CG (C) | 2x4 | Mop/s | 12457.92 ± 509.01 | 12655.86 ± 107.82 | -1.6% |
| NAS CG (C) | 4x4 | Mop/s | 19583.89 ± 118.52 | 19132.62 ± 330.69 | +2.4% |
| NAS MG (C) | 1x4 | Mop/s | 29629.81 ± 59.39 | 29611.59 ± 55.23 | +0.1% |
| NAS MG (C) | 2x4 | Mop/s | 51201.96 ± 375.99 | 51173.10 ± 399.99 | +0.1% |
| NAS MG (C) | 4x4 | Mop/s | 91486.62 ± 627.42 | 91307.70 ± 638.46 | +0.2% |
| PyTorch DDP | 1x4 | step/s | 284.61 ± 0.76 | 279.55 ± 0.94 | +1.8% |
| PyTorch DDP | 2x4 | step/s | 22.51 ± 0.53 | 23.53 ± 1.24 | -4.3% |
| PyTorch DDP | 4x4 | step/s | 19.40 ± 1.05 | 20.00 ± 1.67 | -3.0% |
| `nccl_allgather` | 1x2 | µs (64 MiB) | 1338.57 ± 0.27 | 1339.52 ± 0.89 | +0.1% |
| `nccl_allgather` | 2x1 | µs (64 MiB) | 50893.01 ± 430.90 | 49418.49 ± 747.83 | -2.9% |
| `nccl_allreduce` | 1x2 | µs (64 MiB) | 1047.64 ± 0.45 | 1048.19 ± 0.17 | +0.1% |
| `nccl_allreduce` | 2x1 | µs (64 MiB) | 48769.35 ± 748.70 | 49633.25 ± 333.10 | +1.8% |
| `nccl_alltoall` | 1x2 | µs (64 MiB) | 1194.17 ± 3.66 | 1193.06 ± 0.61 | -0.1% |
| `nccl_alltoall` | 2x1 | µs (64 MiB) | 48963.21 ± 726.54 | 53574.39 ± 5797.38 | +9.4% |
| `nccl_bcast` | 1x2 | µs (64 MiB) | 677.79 ± 0.81 | 678.65 ± 0.45 | +0.1% |
| `nccl_bcast` | 2x1 | µs (64 MiB) | 17968.62 ± 71.88 | 18099.41 ± 83.53 | +0.7% |
| `nccl_bibw` | 1x2 | MB/s (64 MiB) | 117684.94 ± 30.62 | 117649.70 ± 86.79 | +0.0% |
| `nccl_bibw` | 2x1 | MB/s (64 MiB) | 2571.13 ± 54.92 | 2616.79 ± 188.00 | -1.7% |
| `nccl_bw` | 1x2 | MB/s (64 MiB) | 78971.70 ± 21.73 | 78960.53 ± 32.70 | +0.0% |
| `nccl_bw` | 2x1 | MB/s (64 MiB) | 2744.00 ± 31.62 | 2728.15 ± 91.06 | +0.6% |
| `nccl_latency` | 1x2 | µs (64 MiB) | 885.35 ± 0.86 | 885.27 ± 0.25 | -0.0% |
| `nccl_latency` | 2x1 | µs (64 MiB) | 24454.47 ± 617.49 | 24333.13 ± 890.08 | -0.5% |
| `nccl_reduce` | 1x2 | µs (64 MiB) | 634.27 ± 0.66 | 635.27 ± 0.51 | +0.2% |
| `nccl_reduce` | 2x1 | µs (64 MiB) | 17791.33 ± 70.81 | 17706.51 ± 21.51 | -0.5% |
| `nccl_reduce_scatter` | 1x2 | µs (64 MiB) | 601.13 ± 0.38 | 601.86 ± 0.09 | +0.1% |
| `nccl_reduce_scatter` | 2x1 | µs (64 MiB) | 25160.09 ± 196.61 | 25143.36 ± 581.57 | -0.1% |

### Checkpoint and Restart Breakdown

Mean ± sample standard deviation across 1x4, 2x4, and 4x4 GPUs (Class C NAS IS, CG, MG; PyTorch DDP widths 4096, 8192, 16384). The final column reports end-to-end restart latency for `ckpt_restart` (of which communicator rebuild takes 0.57–0.90 s) and communicator rebuild time for `ckpt_resume`.

| Benchmark | Layout | Mode | Ckpt (s) | NCCL-CR (s) | Image/rank (MiB) | Rebuild / restart (s) |
| --- | --- | --- | --- | --- | --- | --- |
| NAS IS (C) | 1x4 | `ckpt_restart` | 5.29 ± 0.15 | 0.98 ± 0.16 | 1860 ± 0 | 4.76 ± 0.04 |
| NAS IS (C) | 1x4 | `ckpt_resume` | 4.98 ± 0.15 | 0.83 ± 0.24 | 1860 ± 0 | 0.75 ± 0.14 |
| NAS IS (C) | 2x4 | `ckpt_restart` | 5.02 ± 0.30 | 1.00 ± 0.12 | 1667 ± 0 | 4.82 ± 0.11 |
| NAS IS (C) | 2x4 | `ckpt_resume` | 4.91 ± 0.08 | 0.89 ± 0.08 | 1667 ± 0 | 0.84 ± 0.09 |
| NAS IS (C) | 4x4 | `ckpt_restart` | 4.70 ± 0.08 | 0.86 ± 0.04 | 1573 ± 0 | 4.73 ± 0.03 |
| NAS IS (C) | 4x4 | `ckpt_resume` | 4.95 ± 0.25 | 0.88 ± 0.25 | 1572 ± 0 | 0.94 ± 0.10 |
| NAS CG (C) | 1x4 | `ckpt_restart` | 4.61 ± 0.06 | 0.87 ± 0.07 | 1741 ± 1 | 4.19 ± 0.05 |
| NAS CG (C) | 1x4 | `ckpt_resume` | 4.78 ± 0.14 | 0.92 ± 0.21 | 1742 ± 0 | 0.56 ± 0.08 |
| NAS CG (C) | 2x4 | `ckpt_restart` | 4.64 ± 0.26 | 0.87 ± 0.21 | 1612 ± 0 | 4.51 ± 0.30 |
| NAS CG (C) | 2x4 | `ckpt_resume` | 4.72 ± 0.24 | 0.93 ± 0.18 | 1612 ± 1 | 0.64 ± 0.02 |
| NAS CG (C) | 4x4 | `ckpt_restart` | 4.53 ± 0.10 | 0.87 ± 0.13 | 1545 ± 0 | 4.13 ± 0.00 |
| NAS CG (C) | 4x4 | `ckpt_resume` | 4.70 ± 0.13 | 1.01 ± 0.13 | 1546 ± 0 | 0.72 ± 0.04 |
| NAS MG (C) | 1x4 | `ckpt_restart` | 5.53 ± 0.05 | 0.73 ± 0.11 | 2342 ± 5 | 4.62 ± 0.19 |
| NAS MG (C) | 1x4 | `ckpt_resume` | 5.54 ± 0.11 | 0.74 ± 0.10 | 2342 ± 4 | 0.61 ± 0.02 |
| NAS MG (C) | 2x4 | `ckpt_restart` | 5.16 ± 0.18 | 0.89 ± 0.16 | 1909 ± 0 | 4.38 ± 0.31 |
| NAS MG (C) | 2x4 | `ckpt_resume` | 5.17 ± 0.14 | 0.89 ± 0.04 | 1909 ± 0 | 0.69 ± 0.10 |
| NAS MG (C) | 4x4 | `ckpt_restart` | 5.07 ± 0.14 | 0.92 ± 0.02 | 1694 ± 2 | 4.71 ± 0.01 |
| NAS MG (C) | 4x4 | `ckpt_resume` | 4.97 ± 0.16 | 0.89 ± 0.12 | 1694 ± 2 | 0.74 ± 0.13 |
| PyTorch w4096 | 1x4 | `ckpt_restart` | 10.93 ± 0.15 | 0.90 ± 0.14 | 5557 ± 19 | 15.59 ± 0.35 |
| PyTorch w4096 | 1x4 | `ckpt_resume` | 18.30 ± 12.81 | 1.00 ± 0.27 | 5547 ± 32 | 0.59 ± 0.08 |
| PyTorch w4096 | 2x4 | `ckpt_restart` | 10.78 ± 0.11 | 0.91 ± 0.15 | 5476 ± 0 | 15.86 ± 0.09 |
| PyTorch w4096 | 2x4 | `ckpt_resume` | 26.25 ± 26.66 | 1.04 ± 0.09 | 5476 ± 1 | 0.85 ± 0.07 |
| PyTorch w4096 | 4x4 | `ckpt_restart` | 10.92 ± 0.22 | 0.94 ± 0.19 | 5476 ± 0 | 15.93 ± 0.37 |
| PyTorch w4096 | 4x4 | `ckpt_resume` | 24.87 ± 24.75 | 0.87 ± 0.23 | 5476 ± 0 | 0.74 ± 0.04 |
| PyTorch w8192 | 1x4 | `ckpt_restart` | 13.69 ± 0.43 | 0.93 ± 0.23 | 7389 ± 197 | 15.69 ± 0.30 |
| PyTorch w8192 | 1x4 | `ckpt_resume` | 13.96 ± 0.31 | 1.00 ± 0.21 | 7605 ± 129 | 0.68 ± 0.12 |
| PyTorch w8192 | 2x4 | `ckpt_restart` | 13.16 ± 0.12 | 0.82 ± 0.10 | 7246 ± 1 | 15.93 ± 0.16 |
| PyTorch w8192 | 2x4 | `ckpt_resume` | 13.30 ± 0.15 | 0.96 ± 0.07 | 7245 ± 0 | 0.69 ± 0.03 |
| PyTorch w8192 | 4x4 | `ckpt_restart` | 13.22 ± 0.11 | 0.88 ± 0.01 | 7246 ± 1 | 16.50 ± 0.20 |
| PyTorch w8192 | 4x4 | `ckpt_resume` | 13.33 ± 0.16 | 0.96 ± 0.16 | 7246 ± 0 | 0.86 ± 0.13 |
| PyTorch w16384 | 1x4 | `ckpt_restart` | 25.41 ± 0.59 | 1.05 ± 0.13 | 15839 ± 296 | 17.52 ± 0.05 |
| PyTorch w16384 | 1x4 | `ckpt_resume` | 24.94 ± 0.21 | 0.69 ± 0.04 | 15497 ± 593 | 0.56 ± 0.30 |
| PyTorch w16384 | 2x4 | `ckpt_restart` | 23.62 ± 0.30 | 1.77 ± 0.26 | 14158 ± 0 | 17.45 ± 0.29 |
| PyTorch w16384 | 2x4 | `ckpt_resume` | 23.53 ± 0.29 | 1.71 ± 0.27 | 14159 ± 0 | 0.73 ± 0.25 |
| PyTorch w16384 | 4x4 | `ckpt_restart` | 23.87 ± 0.45 | 2.19 ± 0.32 | 14159 ± 0 | 17.20 ± 0.21 |
| PyTorch w16384 | 4x4 | `ckpt_resume` | 24.40 ± 1.91 | 1.79 ± 0.47 | 14158 ± 1 | 0.85 ± 0.01 |

### Application-Level `torch.save` / `torch.load` vs. Transparent DMTCP Checkpoint/Restart

All-rank `torch.save` / `torch.load` (model weights + Adam optimizer state) vs. transparent DMTCP checkpoint/restart across 1x4, 2x4, and 4x4 GPUs and three PyTorch DDP widths.

| Layout | Width | `torch.save` (s) | DMTCP ckpt (s) | Ratio | `torch.load` (s) | DMTCP restart (s) | Payload/rank | Image/rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1x4 | 4096 | 0.42 ± 0.02 | 14.61 ± 9.06 | 35.2x | 0.30 ± 0.00 | 15.59 ± 0.35 | 256 MiB | 5552 MiB |
| 1x4 | 8192 | 1.62 ± 0.01 | 13.82 ± 0.37 | 8.5x | 1.35 ± 0.01 | 15.69 ± 0.30 | 1025 MiB | 7497 MiB |
| 1x4 | 16384 | 6.49 ± 0.04 | 25.18 ± 0.47 | 3.9x | 5.48 ± 0.12 | 17.52 ± 0.05 | 4098 MiB | 15668 MiB |
| 2x4 | 4096 | 0.44 ± 0.05 | 18.52 ± 18.87 | 42.5x | 0.31 ± 0.00 | 15.86 ± 0.09 | 256 MiB | 5476 MiB |
| 2x4 | 8192 | 1.62 ± 0.01 | 13.23 ± 0.15 | 8.1x | 1.37 ± 0.01 | 15.93 ± 0.16 | 1025 MiB | 7245 MiB |
| 2x4 | 16384 | 6.48 ± 0.04 | 23.58 ± 0.27 | 3.6x | 5.57 ± 0.11 | 17.45 ± 0.29 | 4098 MiB | 14158 MiB |
| 4x4 | 4096 | 0.44 ± 0.06 | 17.90 ± 17.42 | 40.3x | 0.31 ± 0.00 | 15.93 ± 0.37 | 256 MiB | 5476 MiB |
| 4x4 | 8192 | 1.62 ± 0.01 | 13.27 ± 0.14 | 8.2x | 1.37 ± 0.01 | 16.50 ± 0.20 | 1025 MiB | 7246 MiB |
| 4x4 | 16384 | 6.53 ± 0.03 | 24.13 ± 1.28 | 3.7x | 5.59 ± 0.08 | 17.20 ± 0.21 | 4098 MiB | 14158 MiB |

### Unmatched Point-to-Point and Communicator-Lifecycle Convergence (`p2p_pending`)

Triggered 1 s into a 20 s peer wait across 13 scenario-layout pairs in both `ckpt_restart` and `ckpt_resume` modes (26 runs total; zero bytes drained via fallback).

| Scenario | Layout | Verdict | Restart ckpt (s) | Resume ckpt (s) | Rounds (Rs/Rm) | Drained |
| --- | --- | --- | --- | --- | --- | --- |
| `chain` | 1x3 | PASS | 22.241 | 22.280 | 27 / 27 | 0 |
| `chain` | 2x2 | PASS | 22.164 | 22.039 | 27 / 27 | 0 |
| `create` | 1x2 | PASS | 22.080 | 21.931 | 25 / 25 | 0 |
| `destroy` | 1x2 | PASS | 22.106 | 21.752 | 26 / 26 | 0 |
| `multi` | 1x4 | PASS | 23.561 | 23.500 | 27 / 27 | 0 |
| `multi` | 2x2 | PASS | 22.052 | 22.092 | 27 / 27 | 0 |
| `recv-first` | 1x2 | PASS | 22.064 | 21.988 | 27 / 27 | 0 |
| `recv-first` | 2x1 | PASS | 21.429 | 21.278 | 27 / 27 | 0 |
| `recv-first-cold` | 1x2 | PASS | 22.116 | 21.933 | 27 / 27 | 0 |
| `relay` | 1x3 | PASS | 23.451 | 23.608 | 28 / 28 | 0 |
| `send-first` | 1x2 | PASS | 22.104 | 21.741 | 27 / 27 | 0 |
| `send-first` | 2x1 | PASS | 21.252 | 21.156 | 27 / 26 | 0 |
| `split` | 1x2 | PASS | 21.789 | 22.140 | 26 / 26 | 0 |
