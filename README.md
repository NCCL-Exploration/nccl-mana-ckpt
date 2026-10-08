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

For the AICR cluster, the user is expected to load the NCCL library themselves.
Note: the AICR cluster has RTX PRO 6000 and B200 GPUs, both of which use the Blackwell architecture,
which is supported from NCCL version 2.25 and on. Thus, we recommend downloading the library like so:
```bash
git clone https://github.com/NVIDIA/nccl.git
cd nccl
git checkout v2.29.2-1
make src.build CUDA_HOME=$CUDA_HOME \
	NVCC_GENCODE="-gencode=arch=compute_100,code=sm_100 -gencode=arch=compute_120,code=sm_120"
```
The above make command builds for specifically the architecture of the B200 and the RTX PRO 6000 GPUs.
After the above, the user would have to reconfigure nccl-mana-ckpt. See the "Quick Start" section.


We recommend configuring the following modules (note, other CUDA and NCCL versions may work):

```bash
module load cuda/13.1.1
module load yale/StdEnv
module load OpenMPI/5.0.3-GCC-13.3.0
```
The above will load many additional modules. As a result, export the following environment variable, which by default conflicts with DMTCP:
```bash
export UCX_MEM_EVENTS=n
```

Furthermore, the user is expected to load the NCCL library themselves. The below assumes it is loaded in the same home directory where nccl-mana-ckpt is:
```bash
export NCCL_HOME=$HOME/nccl-2.29.2/build
export LD_LIBRARY_PATH=$NCCL_HOME/lib:$LD_LIBRARY_PATH
export CPATH=$NCCL_HOME/include:$CPATH
```

Lastly, if the user has ``nccl-2.29.2`` built in their home directory, they can run the following checkpoint test script,
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
