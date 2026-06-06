# nccl-mana-ckpt

An NCCL checkpointing plugin for [DMTCP](https://github.com/dmtcp/dmtcp) +
[CRAC](https://github.com/xuyao0127/crac).

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
| CRAC | Defaults to the `crac/` submodule, or set `CRAC_ROOT` |
| MPI | Needed for multi-process tests and examples |
| Build tools | `make`, `gcc` / `g++`, `git` |

### Local Paths

The root `Makefile`, `tests/Makefile`, `single_gpu_tests/Makefile`, and
`multiple_gpu_tests/Makefile` contain hardcoded NCCL `-L` paths. Update those
paths before building if NCCL is installed somewhere else.

By default:

```bash
DMTCP_ROOT=./dmtcp
CRAC_ROOT=./crac
```

For external installs, pass or export the roots:

```bash
export DMTCP_ROOT=~/dmtcp
export CRAC_ROOT=~/crac
```

## Quick Start

```bash
git clone git@github.com:NCCL-Exploration/nccl-mana-ckpt.git
cd nccl-mana-ckpt
git submodule update --init --recursive --remote
```

Build the NCCL plugin:

```bash
make plugin
```

or:

```bash
make
```

Both produce `src/libncclplugin.so`.

Run the root single-GPU check:

```bash
make nccl_check
```

`nccl_check` builds `tests/single_gpu_test` and launches it under DMTCP with
`src/libncclplugin.so`.

## Plugin Loading

CUDA/NCCL checkpoint and restart flows normally require both plugins:

1. `src/libncclplugin.so`: this repository's NCCL checkpoint plugin.
2. `crac/libdmtcp_crac.so`: CRAC's split-CUDA plugin for GPU memory
   checkpoint/restore.

DMTCP accepts multiple plugins as a colon-separated list:

```bash
${DMTCP_ROOT}/bin/dmtcp_launch \
  --with-plugin $PWD/src/libncclplugin.so:${CRAC_ROOT}/libdmtcp_crac.so \
  ./your_test
```

The focused test Makefiles under `tests/` and `single_gpu_tests/` use the
NCCL+CRAC plugin pair. Some root `Makefile` check targets currently launch only
the NCCL plugin; use the focused test Makefiles or adjust `--with-plugin` when
you need full CUDA checkpoint/restart coverage.

## Running Tests

### Single-GPU Driver

```bash
cd tests
make DMTCP_ROOT=../dmtcp CRAC_ROOT=../crac
make test_all DMTCP_ROOT=../dmtcp CRAC_ROOT=../crac
```

Runnable criteria in `tests/Makefile`:

```bash
make test1 DMTCP_ROOT=../dmtcp CRAC_ROOT=../crac  # Basic P2P, no checkpoint
make test2 DMTCP_ROOT=../dmtcp CRAC_ROOT=../crac  # Checkpoint after completion
make test3 DMTCP_ROOT=../dmtcp CRAC_ROOT=../crac  # Checkpoint before NCCL init
make test4 DMTCP_ROOT=../dmtcp CRAC_ROOT=../crac  # In-transit message checkpoint
```

`test_all` depends on the runnable criteria above and restarts from any
checkpoint file produced during each run.

### Manual Single-GPU Launch

```bash
cd tests
nvcc -O3 -g -lnccl -L/path/to/nccl/lib -I../include \
  -o single_gpu_test single_gpu_test.c

${DMTCP_ROOT}/bin/dmtcp_launch --coord-port 7781 --interval 10 \
  --with-plugin $PWD/../src/libncclplugin.so:${CRAC_ROOT}/libdmtcp_crac.so \
  ./single_gpu_test
```

Restart from the newest checkpoint:

```bash
CKPT=$(ls -t ckpt_*.dmtcp 2>/dev/null | head -n 1)
${DMTCP_ROOT}/bin/dmtcp_restart --coord-port 7781 "$CKPT"
```

### Multi-GPU Tests

```bash
cd multiple_gpu_tests
make test_all DMTCP_ROOT=../dmtcp CRAC_ROOT=../crac
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
|   |-- nccl_test_driver.cpp
|   |-- single_gpu_test.c
|   |-- multiproc.c
|   |-- drain_send.c
|   |-- drain_recv.c
|   |-- cc_simple.c
|   `-- cc_complex.c
|-- single_gpu_tests/
|-- multiple_gpu_tests/
|-- crac/
|-- dmtcp/
|-- Makefile
|-- README.md
`-- AGENTS.md
```

## How It Works

`src/libncclplugin.c` registers DMTCP event hooks:

- `DMTCP_EVENT_PRESUSPEND`: mark checkpoint pending, synchronize tracked CUDA
  streams when needed, drain in-flight P2P messages, and coordinate sequence
  progress across ranks before checkpoint.
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

Load both the NCCL plugin and CRAC's split-CUDA plugin. Without CRAC, DMTCP may
try to checkpoint GPU-mapped memory directly.

```bash
${DMTCP_ROOT}/bin/dmtcp_launch \
  --with-plugin $PWD/src/libncclplugin.so:${CRAC_ROOT}/libdmtcp_crac.so \
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
