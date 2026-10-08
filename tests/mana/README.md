# MANA + NCCL plugin compatibility test

The question: can MANA (MPI checkpointing, `cuda-aware` branch) and this repo's NCCL plugin checkpoint and restart an MPI+NCCL program together? And can they do it when NCCL uses CUDA IPC between GPUs of a node?

```bash
tests/mana/build.sh                     # login node; everything goes to $SCRATCH/mana-nccl
sbatch -A m4128 tests/mana/run.sh       # 1 node, debug QOS, 2 ranks on 2 GPUs
cat $SCRATCH/mana-nccl/runs/<jobid>/summary.txt
```

`build.sh` clones `~/mana` (it never modifies or rebuilds it) and builds MANA, with its CUDA forwarding shims when they can be built (see "Two modes"). It then rebuilds the NCCL plugin against MANA's DMTCP. This rebuild is required: the plugin's KVDB shim uses DMTCP's internal C++ API, so it must be compiled against the DMTCP that will load it. The `cuda-aware` branch pins the same DMTCP commit as this repo, so no plugin code changes are needed.

## Two modes

- **`plain`** (what builds today). MANA without `--cuda`. `run.sh` loads the NCCL plugin and the DMTCP CUDA plugin explicitly, in that order. This is enough for this test: its MPI touches only host buffers, so the lower half never uses CUDA. NCCL's IPC happens in the upper half, on the application's own CUDA driver. MANA is built with `cudatoolkit` unloaded, because Cray's `cc` adds `libcuda.so.1` to everything it links while `cudatoolkit` is loaded. `build.sh` fails if `lower-half` still links it.
- **`cuda`**: `mana_launch --cuda`, adding MANA's lower-half forwarding shims and its VMM allocator. This mode is picked automatically once `lib/dmtcp/cuda-forward/libcuda.so.1` exists. It can't be built from the `cuda-aware` branch as committed: `mpi-proxy-split/cuda-forward/Makefile` was never committed, because the root `.gitignore` entry `Makefile` matches it. Once that file is committed (`git add -f`), `build.sh` uses it. The `p2p_legacy_novmm` variant runs only in this mode.

## Program

`mana_nccl_test.cu` uses MPI the way MPI+NCCL codes do:
- It bootstraps NCCL with `MPI_Bcast` of the uniqueId.
- It does a host `MPI_Allreduce` through MANA's lower half.

The NCCL work is a 4 MiB `ncclAllReduce` plus a ring of `ncclSend`/`ncclRecv` (grouped). Every iteration checks every result exactly, and a counter in GPU memory is checked against the iteration number.

## Variants (`VARIANTS=...`)

| Variant | NCCL transport between the 2 GPUs | What it answers |
|---|---|---|
| `native` | P2P (cuMem) | Sanity check, no MANA |
| `no_ipc` | NET, i.e. TCP sockets (`P2P_DISABLE`, `SHM_DISABLE`) | Do MANA and the plugin coexist at all? (two plugin stacks, event order, coordinator key-value store) |
| `shm` | SHM (host shared memory) | Shared-memory transport |
| `p2p_cumem` | P2P via cuMem/VMM POSIX-fd IPC (NCCL ≥ 2.19 default) | CUDA IPC, VMM flavor |
| `p2p_legacy` | P2P via legacy `cudaIpc*` (`NCCL_CUMEM_ENABLE=0`) | CUDA IPC, legacy flavor |
| `p2p_legacy_novmm` | Same, with MANA's VMM allocator off | MANA's `libmana_cuda_vmm.so` turns `cudaMalloc` into VMM memory, which legacy IPC cannot export |

In every variant `NCCL_NET_PLUGIN=none`, so the libfabric/CXI plugin is never loaded. With `NCCL_DEBUG=INFO`, NCCL logs the transport it actually picked; `summary.txt` reports it for each run.

Each MANA run goes through these steps:
1. `mana_launch --cuda --with-plugin libncclplugin.so`
2. Blocking checkpoint, after which the job must keep making progress (resume)
3. A second checkpoint
4. Kill
5. `mana_restart --cuda`, after which the job must run to `PASS` on every rank

## Expected behaviour and why

The NCCL plugin destroys every communicator before a checkpoint and rebuilds it afterwards. This should release NCCL's IPC exports, imports and shared-memory segments before `cuCheckpointProcessCheckpoint` runs. If it does, the IPC variants pass without needing MANA's virtual-IPC layer, which covers only the lower half's MPI. A failure in `p2p_legacy` but not in `p2p_cumem` would point to something that outlives the communicator, such as an IPC export cached by NCCL or a VMM-allocator interaction. MANA's plan notes that `cuCheckpoint` fails with error 304 on any live legacy IPC export or import. Look at `errors.txt` and the logs in the variant's directory.

## Status (2026-10-07): parked

Runs: jobs 59524095, 59524160 and 59524265 (`$SCRATCH/mana-nccl/runs/<job>/`), all in `plain` mode with 2 ranks on 2 A100s.

- `native` passes. NCCL uses `P2P/CUMEM`, and the IPC variants really select `P2P/IPC`, `P2P/CUMEM` and `SHM`.
- **Every MANA variant crashes at the first checkpoint, including the `cuda_only` control (no NCCL, no NCCL plugin).** So the blocker is MANA plus the DMTCP CUDA plugin, not the NCCL plugin. The NCCL plugin's own checkpoint work (drain, consensus, teardown) completes on both ranks every time. The CUDA IPC question remains unanswered: no run got past the first checkpoint.
- **Root cause, from the cores:**
  - `cuCheckpointProcessLock` and `cuCheckpointProcessCheckpoint` succeed.
  - The CUDA plugin then re-reserves the released `/dev/nvidia*` ranges with `mmap(MAP_FIXED)` (`plugin/cuda/cuda-ckpt.cpp`).
  - MANA's `__mmap_wrapper` (`mpi-proxy-split/lower-half/mem-wrapper.cpp`) treats a `MAP_FIXED` request above `max_allocated_addr` as "the arena picks the address" and relocates it. Seen for `/dev/nvidia3` at `0x7ffff3cef000`, above the arena top of `0x1000a000000`.
  - The plugin's `JASSERT(ret == addr)` fails, and MANA segfaults inside `JAssert::Print`, so no message is printed.
- **Tried: `MAP_FIXED_NOREPLACE` in the CUDA plugin.** It still fails, now on the first area (`/dev/nvidiactl`, `0x9c45a000`, free in the core). The cause is unknown: the real `mmap` runs in the lower half, whose libc has its own `errno`, so the upper-half caller never sees the error. This edit exists only in `$SCRATCH/mana-nccl/mana/dmtcp`.
- **Suggested next step:** keep `MAP_FIXED` in the plugin, and make MANA's wrapper never relocate a `MAP_FIXED` request. Two related MANA issues to fix as well:
  - Make `errno` visible to upper-half callers of the wrapped `mmap`/`munmap`.
  - Commit `mpi-proxy-split/cuda-forward/Makefile` with `git add -f`.
