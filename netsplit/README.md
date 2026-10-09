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

Network traffic is staged through host memory: the proxy reports host memory
only, whatever `NCCL_NET_GDR_LEVEL` says. GPU memory cannot be registered with
the NIC after a CUDA checkpoint on this system (`tests/gdr_ckpt.c`, phase 0b);
`NETSPLIT_GDR=1` passes the back end's GPU support through for experiments.

## Layout

| Path | What |
|---|---|
| `include/net_v6.h` | NCCL's net plugin API v6 |
| `include/netsplit-api.h` | the contract between the halves (`NetsplitInfo`) |
| `upper/netproxy.c` | the net proxy: borrowed lower-half TLS per thread, epochs, handle counts |
| `upper/netsplit_dmtcp.cpp` | DMTCP plugin: upper-half mmap into the arena, region filter, signals, per-rank ckpt dirs, restart |
| `lower/` | `netsplit-lh`: MANA's kernel loader without MPI (`netsplit-lh.cpp`), back end and TLS pool (`backend.cpp`) |
| `stub/` | the lower half's `libcudart.so.13` |
| `scripts/` | `nccl_launch`, `nccl_restart` |
| `tests/` | `probe_v6.c`, `gdr_ckpt.c` (phase 0), `counter.c`, `run_ns.sh` (phase 1 driver) |

## Status (2026-10-08)

- Phase 0: aws-ofi-nccl runs through ncclNet v6 with only the stub (21.6 GB/s,
  the same as with the real CUDA runtime). GPUDirect RDMA cannot survive a
  CUDA checkpoint (driver 595.91.07, CXI), so phase 2 is dropped.
- Phase 1: over OFI at 2x1 and 2x2, checkpoint-resume and checkpoint-restart
  pass for `nccl_bench` (allreduce, alltoall, bw) and NAS IS. A restart on
  another node passes, and so does a second checkpoint and restart. Without a
  checkpoint, netsplit is within 4% of NCCL with GDR off. All of the cost
  against production NCCL (GDR on) comes from host staging.
- RESULTS.md has the numbers, what was fixed and what is still open. Logs are
  in `$SCRATCH/netsplit-runs/`.
