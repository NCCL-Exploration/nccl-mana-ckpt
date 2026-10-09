# netsplit: NCCL's network back end in a lower half

netsplit checkpoints NCCL jobs with NCCL's production network, the AWS-OFI
plugin on Slingshot (CXI), including restarts on other nodes. It reuses
nccl-mana-ckpt's checkpoint logic (`../src`, `bin/libncclplugin.so`) unchanged
and adds one thing: the network back end runs in a lower half that is never
checkpointed and is started fresh at restart.

```
one process per GPU
┌ upper half: saved in the image ────────────────────────────────────────┐
│ application · NCCL (libnccl.so.2) · CUDA runtime + driver · DMTCP      │
│ nccl-mana-ckpt plugin (virtual comms, drain, teardown, rebuild)        │
│ net proxy: lib/libnccl-net-netsplit.so (NCCL's net plugin, v6)         │
└──────────────────────── ncclNet v6: 16 calls ──────────────────────────┘
┌ lower half: bin/netsplit-lh, never saved, new at restart ──────────────┐
│ aws-ofi-nccl 1.6.0 · libfabric 2.3.1 (CXI) · libcxi                     │
│ CUDA stub: lib/libcudart_stub.so (4 calls, static answers)             │
│ own ld.so and libc · parked threads that lend their TLS                │
└────────────────────────────────────────────────────────────────────────┘
```

Why: the single-process plugin cannot restart a multi-node job over OFI
(libfabric's CXI state is restored from the image: `NET/OFI Couldn't enable
endpoint ... No space left on device`), and NCCL in MANA's lower half needs CUDA
forwarding shims that keep needing patches. Cutting at NCCL's net plugin API moves only the network out
of the image, and no live CUDA call crosses the split. The design, with a
diagram and per-library handling, is the "NCCL Net-Split Prototype" page.

## Build

```bash
module load cudatoolkit/13.2 nccl/2.29.2-cu13; module unload darshan vasp
make                  # in the repo root: bin/libncclplugin.so (and DMTCP, see ../README.md)
make -C netsplit      # bin/netsplit-lh, lib/libnccl-net-netsplit.so, lib/libcudart_stub.so,
                      # bin/nccl_launch, bin/nccl_restart, phase-0 probes
```

## Run

```bash
export DMTCP_COORD_HOST=$(hostname) DMTCP_COORD_PORT=7781
dmtcp/bin/dmtcp_coordinator --daemon -p $DMTCP_COORD_PORT
srun -N 2 -n 4 --gpus-per-node=4 netsplit/bin/nccl_launch ./app args   # in the ckpt dir
dmtcp/bin/dmtcp_command -c                                             # images: ckpt_rank_<r>/
srun -N 2 -n 4 --gpus-per-node=4 netsplit/bin/nccl_restart --restartdir DIR
```

`nccl_launch` loads the plugins in this order: netsplit, nccl-mana-ckpt, any
`--with-plugin`, DMTCP's CUDA plugin. It sets `NCCL_NET_PLUGIN=netsplit`, puts
`netsplit/lib` first on `LD_LIBRARY_PATH` and takes off any other directory
with a `libnccl-net.so`: after a checkpoint NCCL opens the default name
`libnccl-net.so` for the rebuild, and `lib/libnccl-net.so` links to the proxy.
The lower half loads the real plugin by its full path (`NETSPLIT_BACKEND`,
default `$NCCL_HOME/plugin/lib/libnccl-net.so`).

Two data paths:

- **Host staging (the default).** The proxy reports host memory only,
  whatever `NCCL_NET_GDR_LEVEL` says. The back end needs no CUDA:
  - the stub answers aws-ofi-nccl's four calls with fixed values;
  - libfabric's CUDA support is off (`FI_HMEM=system`).
- **GPUDirect RDMA (`NETSPLIT_GDR=1`).** The proxy passes the back end's
  pointer support through, and NCCL registers its GPU buffers with the NIC.
  libfabric gets CUDA from three stand-ins that `backend_load()` loads before
  the back end (`stub/cuda_fwd.c`): `libcuda.so.1`, `libcudart.so` and
  `libnvidia-ml.so.1`.
  - Each call switches to the calling NCCL thread's own TLS, found in the
    proxy's table of borrowed TLS, and runs the upper half's driver: there is
    still one CUDA driver per process.
  - libfabric's runtime calls are rebuilt on the driver API.
  - Its copies between host and GPU memory go through a pinned bounce buffer
    on a private stream, which the proxy makes before the back end starts.
    So they never wait for NCCL's kernels, nor for the driver's pageable-copy
    staging.
  - Its host-memory pinning is refused: that memory is the lower half's, which
    a restart replaces.
  - Two libfabric settings change, unless already set:
    - GDRCopy is off (`FI_HMEM_CUDA_USE_GDRCOPY=0`), because it cannot pin
      the memory of a restored process (RESULTS.md, "GDR after a
      checkpoint");
    - rendezvous messages carry no eager data (`FI_CXI_RDZV_EAGER_SIZE=0`),
      so an early message leaves nothing to copy into GPU memory.
  - `NETSPLIT_FWD_VERBOSE=1` (or `2`, for all calls) traces the forwarded
    calls.

## Layout

| Path | What |
|---|---|
| `include/net_v6.h` | NCCL's net plugin API v6 |
| `include/netsplit-api.h` | the contract between the halves (`NetsplitInfo`) |
| `upper/netproxy.c` | the net proxy: borrowed lower-half TLS per thread, epochs, handle counts |
| `upper/netsplit_dmtcp.cpp` | DMTCP plugin: upper-half mmap into the arena, region filter, signals, per-rank ckpt dirs, restart |
| `lower/` | `netsplit-lh`: MANA's kernel loader without MPI (`netsplit-lh.cpp`), back end and TLS pool (`backend.cpp`) |
| `stub/` | the lower half's `libcudart.so.13` (`cudart_stub.c`); libfabric's CUDA stand-ins for GDR (`cuda_fwd.c`, `cuda_fwd.h`) |
| `scripts/` | `nccl_launch`, `nccl_restart` |
| `tests/` | `probe_v6.c`, `gdr_ckpt.c` (phase 0), `counter.c`, `run_ns.sh` (phase 1 driver), `gdr_probe.c`/`.sh`, `gdr_pin.c`, `fi_cudart_shim.c` (GDR after a checkpoint) |

## Status (2026-10-08)

- Phase 0: aws-ofi-nccl runs through ncclNet v6 with only the stub (21.6 GB/s,
  the same as with the real CUDA runtime). Phase 0b concluded that GPUDirect
  RDMA cannot survive a CUDA checkpoint, and phase 2 was dropped. A later
  probe showed that only GDRCopy's pin fails (RESULTS.md, "GDR after a
  checkpoint").
- Phase 1: over OFI at 2x1 and 2x2, checkpoint-resume and checkpoint-restart
  pass for `nccl_bench` (allreduce, alltoall, bw) and NAS IS. A restart on
  another node passes, and so does a second checkpoint and restart. Without a
  checkpoint, netsplit is within 4% of NCCL with GDR off. All of the cost
  against production NCCL (GDR on) comes from host staging.
- Stress test (2 nodes, same-node restarts): host staging passed 16 of 16
  cases (`nccl_bench` collectives, bw, NAS IS and PyTorch DDP), with 64
  checkpoints and 24 restarts.
- GPUDirect RDMA (`NETSPLIT_GDR=1`) is experimental. PyTorch DDP deadlocks in
  it (RESULTS.md, "Stress test"). Otherwise:
  - bw 2x1 and allreduce, alltoall and IS at 2x2 pass resume and restart
    with GDRDMA, and so does a restart on another node;
  - without a checkpoint it runs at native GDR speed, except for
    point-to-point messages of 2 KiB or less.
- RESULTS.md has the numbers, what was fixed and what is still open. Logs are
  in `$SCRATCH/netsplit-runs/`.
