# nccl-mana-ckpt: Setup & Testing Guide (with CRAC)

This document provides step-by-step instructions for building and running the
**nccl-mana-ckpt** test suite using the
[DMTCP-CRAC (CRAC-early-development)](https://github.com/DMTCP-CRAC/CRAC-early-development)
library instead of the standard DMTCP submodule.

---

## Table of Contents

1. [Repository Overview](#repository-overview)
2. [Prerequisites](#prerequisites)
3. [Step 1 — Replace the DMTCP Submodule with CRAC](#step-1--replace-the-dmtcp-submodule-with-crac)
4. [Step 2 — Build CRAC](#step-2--build-crac)
5. [Step 3 — Handle the `kvdb.h` Compatibility Issue](#step-3--handle-the-kvdbh-compatibility-issue)
6. [Step 4 — Build the NCCL Plugin](#step-4--build-the-nccl-plugin)
7. [Step 5 — Run the Test Suite](#step-5--run-the-test-suite)
8. [Test Case Reference](#test-case-reference)
9. [Troubleshooting](#troubleshooting)

---

## Repository Overview

```
nccl-mana-ckpt/
├── dmtcp/                    # DMTCP submodule (to be REPLACED with CRAC)
├── include/
│   └── nccl_wrappers.h       # Wrapper function declarations
├── src/
│   ├── libncclplugin.c       # Main DMTCP plugin (event hooks, drain logic)
│   ├── nccl_wrappers.c       # NCCL function wrappers (Send/Recv/AllReduce/CommInit)
│   ├── nccl_iprobe.c         # ncclIprobe implementation (probe via KVDB)
│   ├── dmtcp_kvdb_c.cpp      # C wrapper around DMTCP's C++ KVDB API
│   └── dmtcp_kvdb_c.h        # Header for the KVDB C wrapper
├── tests/
│   ├── Makefile              # Test orchestration (checkpoint, restart, test1-5)
│   ├── nccl_test_driver.cpp  # Main test driver (5 test modes)
│   ├── single_gpu_test.cpp   # Simple single-GPU NCCL AllReduce test
│   ├── hello_world.c         # Minimal NCCL AllReduce example
│   ├── multiproc.c           # Multi-process MPI+NCCL test
│   └── test_drain.cpp        # In-flight message drain test (sketch, won't compile as-is)
├── Makefile                  # Root-level Makefile
└── README.md
```

**What the plugin does:** It intercepts NCCL calls (`ncclSend`, `ncclRecv`,
`ncclAllReduce`, `ncclCommInitRank`, `ncclCommInitAll`) via wrappers, tracks
CUDA streams and in-flight messages, and uses DMTCP's KVDB (key-value
database) to coordinate message draining across ranks before checkpointing.

---

## Prerequisites

On the remote machine, you need:

- **NVIDIA GPU** with CUDA drivers installed
- **CUDA Toolkit** (nvcc compiler) — the Makefiles use `nvcc` as `CC` and `CXX`
- **NCCL library** installed — the Makefiles link `-lnccl` and use an explicit library path:
  `/shared/EL9/explorer/nvidia-hpc-sdk/24.7/Linux_x86_64/24.7/REDIST/comm_libs/12.5/nccl/lib`
  > ⚠️ If your NCCL is installed elsewhere, update the `-L` path in **both**
  > `Makefile` (line 18) and `tests/Makefile` (line 2).
- **MPI** (only needed for `multiproc.c`)
- **Git**
- Standard build tools (`make`, `gcc`/`g++`)

---

## Step 1 — Replace the DMTCP Submodule with CRAC

The repo currently has a git submodule pointing to the standard DMTCP
(`https://github.com/dmtcp/dmtcp`). You need to replace it with CRAC.

```bash
cd /path/to/nccl-mana-ckpt

# 1. Remove the existing DMTCP submodule
git submodule deinit -f dmtcp
rm -rf .git/modules/dmtcp
git rm -f dmtcp

# 2. Remove the .gitmodules entry (or it will be auto-cleaned by git rm)
rm -f .gitmodules

# 3. Clone CRAC into the same ./dmtcp directory
git clone https://github.com/DMTCP-CRAC/CRAC-early-development.git dmtcp
```

> **Why this path matters:** Both `/Makefile` and `/tests/Makefile` hardcode
> `DMTCP_ROOT=./dmtcp` (or `../dmtcp`). By cloning CRAC into `./dmtcp`, all
> existing paths (`${DMTCP_ROOT}/bin/dmtcp_launch`, `${DMTCP_ROOT}/include/`,
> etc.) keep working.

---

## Step 2 — Build CRAC

```bash
cd dmtcp
./configure
make -j$(nproc)
```

Verify the build produced the expected binaries:

```bash
ls -la bin/dmtcp_launch bin/dmtcp_command bin/dmtcp_restart bin/dmtcp_coordinator
```

All four should exist. Then return to the project root:

```bash
cd ..
```

---

## Step 3 — KVDB Compatibility (Already Resolved)

The standard DMTCP repo includes `kvdb.h`, a coordinator-based key-value
database API. **CRAC does not include this header or implementation** (KVDB
was added to mainline DMTCP after CRAC forked).

The plugin's `src/dmtcp_kvdb_c.cpp` originally depended on `kvdb.h`. This has
already been fixed: the DMTCP KVDB dependency was replaced with a local
in-process `std::map`-based key-value store that provides the exact same API.
No other source files were changed.

This local store is functionally equivalent for all single-process tests
(tests 1–5) since they run with a single rank. For future multi-node work,
this would need to be extended to a distributed store.

> **No action needed** — this step is already done in the codebase.

---

## Step 4 — Build the NCCL Plugin

From the project root:

```bash
# Ensure NCCL library path is correct for your system.
# Edit Makefile line 18 and tests/Makefile line 2 if needed.

# Build the plugin
make clean
make
```

This produces `src/libncclplugin.so`.

If you get an error about missing headers, verify:
```bash
ls dmtcp/include/dmtcp.h    # Must exist (provided by CRAC)
```

---

## Step 5 — Run the Test Suite

### Quick Sanity Check (Plugin + Basic DMTCP)

```bash
# From the project root — runs DMTCP's own dmtcp1 test with the plugin loaded
make check
```

### NCCL Single-GPU Sanity Check

```bash
# From the project root
make nccl_check
```

This builds `tests/single_gpu_test` and runs it under `dmtcp_launch` with the
plugin. It performs an AllReduce on a single GPU and sleeps 40s (to allow
manual checkpoint triggering).

### Full Automated Test Suite (Tests 1–5)

```bash
cd tests

# Build the test driver
make all

# Run ALL tests sequentially (each test checkpoints and restarts):
make test_all

# Or run individual tests:
make test1   # Criteria 1: Basic P2P Complete
make test2   # Criteria 2: Post-Completion Checkpoint
make test3   # Criteria 3: Pre-Init Checkpoint
make test4   # Criteria 4: Single Message In-Transit
make test5   # Criteria 5: Multi Message In-Transit
```

### Customizing Test Parameters

```bash
# Adjust message count and size (in KB):
make test5 MSGS=10 SIZE=2048
```

### Cleanup

```bash
# Kill the coordinator and running processes:
make quit

# Remove all build artifacts and checkpoint files:
make clean
```

---

## Test Case Reference

| Test | Makefile Target | Mode | What It Verifies |
|------|----------------|------|-----------------|
| **Test 1** | `make test1` | Basic P2P | `ncclSend`/`ncclRecv` complete. No checkpoint taken. Verifies data integrity. |
| **Test 2** | `make test2` | Post-Completion | Messages finish, then sleeps 20s. Checkpoint triggers AFTER all communication is done. Verifies checkpoint of idle NCCL state. |
| **Test 3** | `make test3` | Pre-Init | Sleeps 10s BEFORE `ncclCommInitAll`. Checkpoint triggers before NCCL is initialized. Verifies checkpoint of pre-NCCL state. |
| **Test 4** | `make test4` | Single In-Transit | Sends 1 message, sleeps 20s (checkpoint window), then receives. Tests drain logic for a single in-flight message. |
| **Test 5** | `make test5` | Multi In-Transit | Sends N messages, sleeps 20s (checkpoint window), then receives all. Tests drain logic for multiple in-flight messages. |

**Each test performs:**
1. Launch the test driver under `dmtcp_launch` with the plugin
2. (For modes 2–5) Wait, then trigger a checkpoint
3. Wait for the process to complete
4. If a checkpoint file was created, restart from it using `dmtcp_restart`
5. Verify the restart succeeds and data integrity is maintained

---

## Troubleshooting

### `mmap failed: Invalid argument` on Restart

This was encountered in previous testing. The plugin has a
`dmtcp_skip_memory_region_ckpting()` function in `libncclplugin.c` that skips
NVIDIA device mappings (`/dev/nvidia*`, `/dev/uverbs*`, etc.). If you still
hit this:

- CRAC's split-process architecture (`contrib/split-cuda/`) is specifically
  designed to handle CUDA memory correctly. Make sure CRAC was built with CUDA
  support if available.
- Check if CRAC requires the split-cuda plugin to be loaded alongside your
  NCCL plugin.

### NCCL Library Not Found

If you get linker errors about `-lnccl`:

```bash
# Find your NCCL library:
find / -name "libnccl.so*" 2>/dev/null

# Update the -L path in both Makefiles to match.
```

### `kvdb.h: No such file or directory`

You skipped Step 3. Go back and either copy the header or stub out the KVDB
calls.

### Coordinator Connection Failures

```bash
# Make sure no stale coordinator is running:
make quit   # (in tests/ directory)
# or
dmtcp/bin/dmtcp_command --quit --coord-port 7781

# Check if port 7781 is in use:
ss -tlnp | grep 7781
```

### `test_drain.cpp` Won't Compile

This file has a typo (`ncclCommt_t` instead of `ncclComm_t`) and uses
undefined variables (`ncclId`). It appears to be a conceptual sketch, not a
runnable test. The actual drain testing is done via `nccl_test_driver.cpp`
modes 4 and 5. You can safely ignore `test_drain.cpp`.

### `multiproc.c` Requires MPI

This test uses MPI (`#include <mpi.h>`). To compile and run:

```bash
# Compile with MPI wrapper:
mpicc -o multiproc multiproc.c -lnccl -lcudart -L/path/to/nccl/lib

# Run with multiple ranks:
dmtcp/bin/dmtcp_launch --coord-port 7781 \
  --with-plugin $PWD/src/libncclplugin.so \
  mpirun -np 2 ./tests/multiproc
```

---

## Summary of Steps

```bash
# 1. Replace DMTCP with CRAC
git submodule deinit -f dmtcp && rm -rf .git/modules/dmtcp && git rm -f dmtcp
git clone https://github.com/DMTCP-CRAC/CRAC-early-development.git dmtcp

# 2. Build CRAC
cd dmtcp && ./configure && make -j$(nproc) && cd ..

# 3. kvdb.h compatibility — already resolved in src/dmtcp_kvdb_c.cpp (no action needed)

# 4. Build the plugin
make clean && make

# 5. Run tests
cd tests && make test_all
```
