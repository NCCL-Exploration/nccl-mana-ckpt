# nccl-mana-ckpt

An NCCL checkpointing plugin for
[DMTCP-CRAC](https://github.com/DMTCP-CRAC/CRAC-early-development).
Intercepts NCCL collective and point-to-point calls, tracks CUDA streams and
in-flight messages, and drains outstanding operations before a DMTCP checkpoint
is taken.

## Prerequisites

| Dependency | Notes |
|------------|-------|
| **NVIDIA GPU + drivers** | Required for CUDA / NCCL |
| **CUDA Toolkit** (`nvcc`) | Used as the compiler (`CC` / `CXX` in the Makefiles) |
| **NCCL** | Link path is hardcoded — see [NCCL Path](#nccl-library-path) below |
| **DMTCP-CRAC** | Must be built separately — see [Building CRAC](#1-build-dmtcp-crac) |
| **MPI** | Only needed for `tests/multiproc.c` |
| Standard build tools | `make`, `gcc` / `g++`, `git` |

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

### 1. Build DMTCP-CRAC

Clone and build CRAC separately (outside this repo):

```bash
cd ~
git clone https://github.com/DMTCP-CRAC/CRAC-early-development.git crac
cd crac
./configure
make -j$(nproc)
```

This produces:
- `~/crac/bin/dmtcp_launch`, `dmtcp_command`, `dmtcp_restart` — the DMTCP
  binaries
- `~/crac/lib/dmtcp/libdmtcp_crac.so` — the **CRAC split-CUDA plugin**
  (required for GPU checkpointing)

> **Note:** If the CRAC build fails with `_STAT_VER` errors on newer glibc
> (2.33+), add the following before the `#undef` block in
> `plugin/pathvirt/pathvirt.cpp`:
> ```c
> #ifndef _STAT_VER
> # if defined(__x86_64__) || defined(__aarch64__)
> #  define _STAT_VER 1
> # elif defined(__i386__)
> #  define _STAT_VER 3
> # else
> #  define _STAT_VER 0
> # endif
> #endif
> ```

### 2. Clone This Repository

```bash
git clone git@github.com:NCCL-Exploration/nccl-mana-ckpt.git
cd nccl-mana-ckpt
```

### 3. Build the NCCL Plugin

Set `DMTCP_ROOT` to your CRAC installation (for the include headers):

```bash
make clean
make DMTCP_ROOT=~/crac
```

This produces `src/libncclplugin.so`.

---

## Running Tests

### Important: Loading Both Plugins

When running under DMTCP-CRAC with CUDA/NCCL, you must load **two** plugins:

1. **`libncclplugin.so`** — this project's NCCL checkpoint plugin
2. **`libdmtcp_crac.so`** — CRAC's split-CUDA plugin (handles GPU memory
   checkpoint/restore)

They are specified together with a colon separator in the `--with-plugin` flag.

### Checkpoint the Single-GPU Test

```bash
cd tests

# Build the test
nvcc -O3 -g -lnccl -L/path/to/nccl/lib -o single_gpu_test single_gpu_test.cpp

# Launch under DMTCP-CRAC with both plugins
~/crac/bin/dmtcp_launch --interval 10 \
  --with-plugin $PWD/../src/libncclplugin.so:~/crac/lib/dmtcp/libdmtcp_crac.so \
  ./single_gpu_test
```

The `--interval 10` flag tells DMTCP to checkpoint automatically every 10
seconds. The test sleeps for 40 seconds, so a checkpoint will be taken during
execution.

### Restart from Checkpoint

```bash
# Find the latest checkpoint file
CKPT=$(ls -t ckpt_*.dmtcp 2>/dev/null | head -n 1)

# Restart
~/crac/bin/dmtcp_restart --coord-port 7781 $CKPT
```

### Run the Automated Test Suite

```bash
cd tests

# Build the test driver
make DMTCP_ROOT=~/crac

# Run all 5 test criteria:
make test_all DMTCP_ROOT=~/crac

# Or individual tests:
make test1 DMTCP_ROOT=~/crac   # Basic P2P — no checkpoint
make test2 DMTCP_ROOT=~/crac   # Checkpoint after messages complete
make test3 DMTCP_ROOT=~/crac   # Checkpoint before NCCL init
make test4 DMTCP_ROOT=~/crac   # Checkpoint with 1 message in-transit
make test5 DMTCP_ROOT=~/crac   # Checkpoint with N messages in-transit
```

> **Note:** The tests Makefile currently does not load `libdmtcp_crac.so`
> automatically. You may need to update the `run_test` macro in
> `tests/Makefile` to include both plugins in the `--with-plugin` flag.

### Cleanup

```bash
# Kill the DMTCP coordinator and stale processes:
make quit

# Remove build artifacts and checkpoint files:
make clean
```

---

## Project Structure

```
nccl-mana-ckpt/
├── include/
│   └── nccl_wrappers.h       # Shared declarations for wrapper globals/helpers
├── src/
│   ├── libncclplugin.c       # DMTCP plugin: event hooks, stream drain, memory skip
│   ├── nccl_wrappers.c       # NCCL interposition (Send/Recv/AllReduce/CommInit)
│   ├── nccl_iprobe.c         # ncclIprobe: query pending messages via KVDB
│   ├── dmtcp_kvdb_c.cpp      # Local key-value store for message tracking
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

### How It Works

The plugin hooks into DMTCP's event system:

- **`DMTCP_EVENT_PRESUSPEND`** — Called before GPU threads are suspended.
  The plugin drains all in-flight NCCL operations and synchronizes CUDA
  streams at this point, **before** the CRAC plugin suspends GPU state.
  This ordering is critical: the CRAC CUDA API needs active GPU threads to
  dump GPU state to CPU memory.

- **`DMTCP_EVENT_RESUME`** / **`DMTCP_EVENT_RESTART`** — Reset plugin state
  after checkpoint or restart.

NCCL functions (`ncclSend`, `ncclRecv`, `ncclAllReduce`, `ncclCommInitRank`,
`ncclCommInitAll`) are intercepted via wrapper functions that track
communicator rank info and active CUDA streams.

---

## Troubleshooting

### `mmap failed: Invalid argument` on Restart

Make sure you are loading **both** plugins. The CRAC split-CUDA plugin
(`libdmtcp_crac.so`) is essential for handling GPU memory regions during
checkpoint/restart. Without it, DMTCP will try to checkpoint GPU-mapped
memory and fail.

```bash
# Correct — two plugins, colon-separated:
~/crac/bin/dmtcp_launch \
  --with-plugin $PWD/src/libncclplugin.so:~/crac/lib/dmtcp/libdmtcp_crac.so \
  ./your_test

# Wrong — only the NCCL plugin:
~/crac/bin/dmtcp_launch \
  --with-plugin $PWD/src/libncclplugin.so \
  ./your_test
```

### NCCL Library Not Found

```bash
# Locate your NCCL library:
find / -name "libnccl.so*" 2>/dev/null

# Update -L path in Makefile (line 18) and tests/Makefile (line 2).
```

### CRAC Build Fails (`_STAT_VER` or `JASSERT` errors)

These are known compatibility issues with CRAC on newer Linux distributions
(glibc 2.33+). See the build note in [Step 1](#1-build-dmtcp-crac) above.

### Coordinator Connection Failures

```bash
# Kill stale coordinators:
~/crac/bin/dmtcp_command --quit --coord-port 7781 2>/dev/null

# Check if port 7781 is already in use:
ss -tlnp | grep 7781
```
