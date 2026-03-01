# nccl-mana-ckpt

An NCCL checkpointing plugin for [DMTCP-CRAC](https://github.com/DMTCP-CRAC/CRAC-early-development).
Intercepts NCCL collective and point-to-point calls, tracks CUDA streams and
in-flight messages, and drains outstanding operations before a DMTCP checkpoint
is taken.

## Prerequisites

| Dependency | Notes |
|------------|-------|
| **NVIDIA GPU + drivers** | Required for CUDA / NCCL |
| **CUDA Toolkit** (`nvcc`) | Used as the compiler (`CC` / `CXX` in the Makefiles) |
| **NCCL** | Link path is hardcoded — see [NCCL Path](#nccl-library-path) below |
| **MPI** | Only needed for `tests/multiproc.c` |
| **Standard build tools** | `make`, `gcc` / `g++`, `git` |

### NCCL Library Path

Both `Makefile` (root) and `tests/Makefile` contain a hardcoded `-L` path for
the NCCL library:

```
/shared/EL9/explorer/nvidia-hpc-sdk/24.7/Linux_x86_64/24.7/REDIST/comm_libs/12.5/nccl/lib
```

If your NCCL is installed elsewhere, update the path in **both** files before
building.

---

## Quick Start

### 1. Clone the Repository

```bash
git clone --recursive git@github.com:NCCL-Exploration/nccl-mana-ckpt.git
cd nccl-mana-ckpt
```

> The `--recursive` flag automatically clones the DMTCP-CRAC submodule into
> `./dmtcp`.

If you already cloned without `--recursive`:

```bash
git submodule update --init --recursive
```

### 2. Build DMTCP-CRAC

```bash
cd dmtcp
./configure
make -j$(nproc)
cd ..
```

Verify the build produced the expected binaries:

```bash
ls dmtcp/bin/dmtcp_launch dmtcp/bin/dmtcp_command dmtcp/bin/dmtcp_restart
```

### 3. Build the Plugin

```bash
make clean
make
```

This produces `src/libncclplugin.so`.

---

## Running Tests

All tests are in the `tests/` directory and are orchestrated by
`tests/Makefile`. The primary test binary is `nccl_test_driver`, which accepts
a *mode* argument controlling the checkpoint scenario.

### Run All Tests

```bash
cd tests
make test_all
```

This runs tests 1–5 sequentially. Each test:
1. Launches `nccl_test_driver` under `dmtcp_launch` with the plugin
2. Triggers a checkpoint (for modes 2–5) after a delay
3. Waits for the process to finish
4. Restarts from the checkpoint (if one was created) via `dmtcp_restart`
5. Verifies data integrity

### Run Individual Tests

```bash
cd tests

make test1   # Criteria 1: Basic P2P — no checkpoint, verify NCCL works
make test2   # Criteria 2: Checkpoint AFTER all messages complete
make test3   # Criteria 3: Checkpoint BEFORE NCCL is initialized
make test4   # Criteria 4: Checkpoint with 1 message in-transit
make test5   # Criteria 5: Checkpoint with N messages in-transit
```

### Customize Test Parameters

```bash
# Change message count and size (in KB):
make test5 MSGS=10 SIZE=2048
```

### Cleanup

```bash
# Kill the DMTCP coordinator and stale processes:
make quit

# Remove build artifacts and checkpoint files:
make clean
```

### Additional Test Executables

These are standalone tests (not part of `make test_all`):

| File | Description | How to Run |
|------|-------------|------------|
| `single_gpu_test.cpp` | Simple AllReduce on 1 GPU + sleep for manual checkpoint | `make nccl_check` (from root) |
| `hello_world.c` | Minimal AllReduce example, no checkpoint | Compile manually with `nvcc` |
| `multiproc.c` | Multi-process MPI + NCCL AllReduce | Requires MPI — see below |
| `test_drain.cpp` | Conceptual drain sketch (**does not compile**) | N/A |

**Running `multiproc.c`:**
```bash
# Compile:
nvcc -o tests/multiproc tests/multiproc.c -lnccl -lmpi -I include

# Run under DMTCP:
dmtcp/bin/dmtcp_launch --coord-port 7781 \
  --with-plugin $PWD/src/libncclplugin.so \
  mpirun -np 2 ./tests/multiproc
```

---

## Project Structure

```
nccl-mana-ckpt/
├── dmtcp/                    # DMTCP-CRAC submodule (checkpoint infrastructure)
├── include/
│   └── nccl_wrappers.h       # Shared declarations for wrapper globals/helpers
├── src/
│   ├── libncclplugin.c       # DMTCP plugin: event hooks, stream drain, memory skip
│   ├── nccl_wrappers.c       # NCCL interposition (Send/Recv/AllReduce/CommInit)
│   ├── nccl_iprobe.c         # ncclIprobe: query pending messages via KVDB
│   ├── dmtcp_kvdb_c.cpp      # Local key-value store (replaces DMTCP KVDB for CRAC)
│   └── dmtcp_kvdb_c.h        # C-linkage header for the KV store
├── tests/
│   ├── Makefile              # Test orchestration (test1–5, checkpoint, restart)
│   ├── nccl_test_driver.cpp  # Main test driver with 5 checkpoint modes
│   ├── single_gpu_test.cpp   # Simple single-GPU AllReduce test
│   ├── hello_world.c         # Minimal NCCL example
│   ├── multiproc.c           # Multi-rank MPI+NCCL test
│   └── test_drain.cpp        # In-flight drain sketch (not runnable)
├── Makefile                  # Root build: plugin .so + sanity checks
└── README.md
```

---

## Troubleshooting

### `mmap failed: Invalid argument` on Restart

The plugin skips NVIDIA device memory regions during checkpointing
(`dmtcp_skip_memory_region_ckpting` in `libncclplugin.c`). If you still hit
this, CRAC's split-CUDA plugin (`contrib/split-cuda/` inside the CRAC repo)
may need to be loaded alongside the NCCL plugin.

### Coordinator Connection Failures

```bash
# Kill stale coordinators:
cd tests && make quit
# or directly:
dmtcp/bin/dmtcp_command --quit --coord-port 7781 2>/dev/null

# Check if port 7781 is already in use:
ss -tlnp | grep 7781
```

### NCCL Library Not Found

```bash
# Locate your NCCL library:
find / -name "libnccl.so*" 2>/dev/null

# Update -L path in Makefile (line 18) and tests/Makefile (line 2).
```
