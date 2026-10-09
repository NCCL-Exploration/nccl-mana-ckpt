# netsplit results (2026-10-08)

netsplit runs NCCL's network back end in a lower half that is never
checkpointed (see README.md). These are the prototype's first results, on
Perlmutter, against NCCL without checkpointing.

**Setup.** Two GPU nodes per job: 4 A100s and 4 Slingshot-11 NICs (CXI) each.

- Software: NCCL 2.29.2-cu13, aws-ofi-nccl 1.6.0 through net API v6,
  libfabric 2.3.1, CUDA 13.2, nccl-mana-ckpt 6b1c912 plus `netsplit/`.
- Driver: not recorded for jobs 59571523 to 59576795. Node nid200437 (job
  59579060) runs 580.178.04 with CUDA 13.2's compatibility `libcuda`. An
  earlier version of this file gave 595.91.07, which is the version of the
  `cuda-checkpoint` tool, not of the driver.
- Programs: `nccl_bench` (no MPI) and NAS IS class B over NCCL.
- Layouts are NODESxGPUS-PER-NODE with one rank per GPU.
- Jobs:
  - 59571523 (phase 0);
  - 59574333 and 59576795 (phase 1);
  - 59579060, 59582340 and 59585324 (GDR probes);
  - 59583126 and 59584050 (GDR in netsplit, and the host-staged regression).
- Raw logs: `$SCRATCH/netsplit-runs/<job>-<tag>/`, one `summary.txt` per
  driver run (`tests/run_ns.sh`).

## Summary

- **Checkpointing works on the production network.** Checkpoint-resume and
  checkpoint-restart pass for allreduce, alltoall, bw and IS over OFI at 2x1
  and 2x2. A restart on another node passes too, and so does a second
  checkpoint and restart of a restarted job. At each restart, the back end
  starts fresh in the new lower half (epoch 2, then 3), and each rank maps one
  CUDA driver.
- **The split costs nothing measurable.** Without a checkpoint, netsplit is
  within 4% of NCCL with GPUDirect RDMA (GDR) off at every message size, from
  8 B to 64 MiB.
- **All of the cost against production NCCL is host staging** (the default
  mode). Against NCCL with GDR:
  - up to 64 KiB: allreduce is 4–16% slower (up to 6 µs). bw is 1–13%
    faster and alltoall is 2–17% faster, because GDR is slower there;
  - from 1 MiB: 78–83% of bandwidth;
  - for a 64 MiB message: about twice the time for allreduce (2.1x) and
    alltoall (1.9x).
- **The stress test passed with host staging:** 16 of 16 cases, 64
  checkpoints and 24 restarts. That includes PyTorch DDP at 2x4 (see "Stress
  test").
- **GDR mode (`NETSPLIT_GDR=1`) matches native GDR and checkpoints, but not
  with PyTorch.** PyTorch DDP deadlocks between NCCL's proxy thread and the
  application thread inside the CUDA driver, with or without a checkpoint.
  - Through CUDA forwarding in the lower half, bw 2x1 and allreduce, alltoall
    and IS at 2x2 pass resume and restart with GDRDMA, and so does a restart
    on another node.
  - Without a checkpoint it runs at native GDR speed (0.94–1.03), except for
    point-to-point messages of 2 KiB or less (0.54–0.65; see "GDR in
    netsplit").

## Phase 0: can the back end run alone?

| Test | Result |
|---|---|
| 0a: aws-ofi-nccl driven through ncclNet v6 by a CUDA-free program, with only the CUDA stub | Passes on 1 and 2 nodes: 200 x 4 MiB at 21.6 GB/s, the same as with the real CUDA runtime. The only CUDA library mapped is the stub. |
| 0b: register GPU memory with the NIC after `cuCheckpointProcessRestore` | Fails for dma-buf and peer memory. Phase 2 (GDR) was dropped on this result, but the failing step is GDRCopy's pin, not the NIC's (see "GDR after a checkpoint"). |

## Checkpoint, resume and restart

Every run from the phase-1 driver on jobs 59574333 and 59576795 is counted,
plus the two-restart test, which was run by hand. That includes runs made
before the fixes below, except that restarts on another node count only after
the bootstrap fix. The 2x layouts use OFI between the
nodes, and the 1x2 layouts use P2P/cuMem unless the network is forced
(`NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1`).

- A checkpoint time is from `dmtcp_command -c` until the coordinator writes
  the restart script.
- A restart time is from `srun nccl_restart` until the first progress line.

| Mode | Program, layout | Passed | Checkpoint, median (s) | Restart, median (s) | Failures |
|---|---|---|---|---|---|
| resume | bw 2x1 | 2 of 2 | 3.0 | | |
| restart | bw 2x1 | 6 of 8 | 3.1 | 9.4 | 2x CUDA restore 304 |
| resume | allreduce 2x2 | 2 of 2 | 4.1 | | |
| restart | allreduce 2x2 | 8 of 8 | 4.1 | 3.9 | |
| resume | alltoall 2x2 | 5 of 6 | 4.1 | | 1 hang (first run) |
| restart | alltoall 2x2 | 3 of 5 | 4.1 | 4.2 | 1 exit 99 (first run); 1 checkpoint request refused by the coordinator |
| resume | IS 2x2 | 2 of 2 | 4.2 | | |
| restart | IS 2x2 | 3 of 3 | 4.3 | 3.9 | |
| resume, restart | allreduce 1x2 | 2 of 2 | 3.7 | 5.2 | |
| resume, restart | alltoall 2x1 | 2 of 2 | 3.4 | 4.4 | |
| restart | IS 1x2 | 1 of 1 | 4.1 | 3.9 | |
| restart, other node | allreduce 1x2, network forced | 3 of 3 | 3.8 | 5.2 | |
| restart, other node | allreduce 1x2 | 1 of 1 | 3.6 | 7.5 | |
| restart, other node | alltoall 1x2 | 1 of 1 | 4.1 | 6.0 | |
| restart, other node | IS 1x2 | 3 of 4 | 4.3 | 7.0 | 1x CUDA restore 304 |
| two restarts in a row | allreduce 2x2 | 1 of 2 | | | 1 segfault at the second checkpoint, before the nscd fix |

In total, 13 of 14 resumes, 23 of 27 restarts on the same nodes and 8 of 9
restarts on another node passed. After each restart, every rank logged the
back end starting in the new lower half (epoch 2; epoch 3 after the second
restart).

- Most checkpoints took 3–4 s, and most restarts 3–10 s, for images of about
  2 GB per rank on `/pscratch`.
- A few runs on job 59576795 took 15–32 s for one or the other, presumably
  because of Lustre load. The table's medians include them.
- The "two restarts in a row" run is launch, checkpoint, restart, checkpoint,
  restart. A one-rank run of the same sequence, with the nscd fix, also
  passed.

## Performance without a checkpoint

Each column is one run of `nccl_bench -T 1` (1 s per size) on job 59574333:

- **NCCL, GDR**: NCCL's default (GDR on).
- **NCCL, GDR off**: `NCCL_NET_GDR_LEVEL=LOC`, the data path that netsplit
  uses.
- **netsplit**: `nccl_launch` (DMTCP, nccl-mana-ckpt and the CUDA plugin
  loaded), no checkpoint.

In the ratio columns, more than 1 means netsplit is faster.

`bw` 2x1 (MB/s)

| Size | NCCL, GDR | NCCL, GDR off | netsplit | netsplit vs GDR off | netsplit vs GDR |
|---:|---:|---:|---:|---:|---:|
| 8 B | 0.5 | 0.6 | 0.5 | 0.96 | 1.08 |
| 1 KiB | 64.7 | 67.9 | 67.1 | 0.99 | 1.04 |
| 64 KiB | 2179.7 | 2242.5 | 2195.3 | 0.98 | 1.01 |
| 1 MiB | 14461.4 | 12061.0 | 12044.5 | 1.00 | 0.83 |
| 4 MiB | 20255.6 | 16119.1 | 16127.4 | 1.00 | 0.80 |
| 16 MiB | 22172.1 | 17636.9 | 17600.4 | 1.00 | 0.79 |
| 64 MiB | 22709.1 | 17889.4 | 17777.9 | 0.99 | 0.78 |

`allreduce` 2x2 (µs)

| Size | NCCL, GDR | NCCL, GDR off | netsplit | netsplit vs GDR off | netsplit vs GDR |
|---:|---:|---:|---:|---:|---:|
| 8 B | 27.1 | 28.7 | 29.3 | 0.98 | 0.92 |
| 1 KiB | 28.8 | 31.7 | 32.3 | 0.98 | 0.89 |
| 64 KiB | 65.5 | 66.9 | 69.1 | 0.97 | 0.95 |
| 1 MiB | 194.4 | 257.7 | 258.8 | 1.00 | 0.75 |
| 4 MiB | 246.6 | 387.6 | 384.3 | 1.01 | 0.64 |
| 16 MiB | 722.1 | 1474.4 | 1454.0 | 1.01 | 0.50 |
| 64 MiB | 2696.7 | 5859.3 | 5746.3 | 1.02 | 0.47 |

`alltoall` 2x2 (µs)

| Size | NCCL, GDR | NCCL, GDR off | netsplit | netsplit vs GDR off | netsplit vs GDR |
|---:|---:|---:|---:|---:|---:|
| 8 B | 46.2 | 39.1 | 39.9 | 0.98 | 1.16 |
| 1 KiB | 44.7 | 39.6 | 40.1 | 0.99 | 1.12 |
| 64 KiB | 67.2 | 65.1 | 65.7 | 0.99 | 1.02 |
| 1 MiB | 177.2 | 281.5 | 269.0 | 1.05 | 0.66 |
| 4 MiB | 515.3 | 932.7 | 912.8 | 1.02 | 0.56 |
| 16 MiB | 1839.0 | 3545.9 | 3448.9 | 1.03 | 0.53 |
| 64 MiB | 7067.9 | 13865.5 | 13694.5 | 1.01 | 0.52 |

- **The split's own cost** is the gap between netsplit and NCCL with GDR off.
  It is at most 4% slower, for bw at 8 B, where it moves 0.5 MB/s:
  - allreduce: at most 3% slower (at 64 KiB, 69.1 µs against 66.9 µs);
  - alltoall: at most 2% slower (at 8 B, 39.9 µs against 39.1 µs).

  At large sizes netsplit is up to 5% faster. Each number is a single run, so
  these gaps are within run-to-run noise.
- **Host staging** is what costs against production NCCL. It matters from
  about 256 KiB.
- **Small alltoall messages are slower with GDR.** At 8 B, NCCL with GDR
  takes 46 µs, against 39 µs with GDR off, so netsplit is ahead there.

## GDR after a checkpoint

`tests/gdr_probe.c` (driver `tests/gdr_probe.sh`) and `tests/gdr_pin.c` were
run on job 59579060 (one node, nid200437). They separate what the earlier
probe lumped together.

- Each probe is a plain CUDA program that `cuda-checkpoint --toggle`
  checkpoints and restores in place: no NCCL and no DMTCP.
- After the restore, the probe registers GPU memory with libfabric and checks
  each successful registration with a loopback transfer into the buffer. It
  tries:
  - a buffer allocated before the checkpoint;
  - new `cudaMalloc` and `cuMemCreate` (RDMA-capable) buffers;
  - a buffer in a new context;
  - memory owned by a helper process that is never checkpointed.
- libfabric's registration cache is off (`FI_MR_CACHE_MAX_COUNT=0`), so every
  result is a real registration.

| After the restore | libfabric default | `FI_HMEM_CUDA_USE_GDRCOPY=0` |
|---|---|---|
| Register any GPU memory (peer memory or dma-buf) | `EINVAL`: `gdr_pin_buffer` fails | ok, and the NIC's writes land correctly |
| The helper's own memory, in its own domain | ok | ok |
| GDRCopy pin (`gdr_pin`), handle from before or new | `EINVAL`, for old and new buffers | |

- **The NIC can use GPU memory after a CUDA checkpoint.** What fails is
  GDRCopy's pin, which libfabric makes for every CUDA registration. It fails
  for any memory of a restored process, even with a GDRCopy handle opened
  after the restore. This is the error NCCL reported in all three designs
  ("NET/OFI Unable to register memory ... RC: 22").
- **Why the pin fails: NVIDIA's P2P pin API, not GDRCopy** (jobs 59585324,
  59606626 and 59606939; sources: gdrcopy v2.5, open-gpu-kernel-modules
  580.178.04, HPE shs-cxi-driver):
  - gdrdrv pins with `nvidia_p2p_get_pages_persistent()`
    (`GDR_ATTR_USE_PERSISTENT_MAPPING` = 1). The `EINVAL` comes from that call:
    gdrdrv's only other `EINVAL` is a page-size check that cannot fail, since
    the driver always reports 64 KiB pages.
  - Given no P2P token, the driver finds the memory through its third-party
    P2P registry (`RmP2PGetInfoWithoutToken`, `src/nvidia/src/kernel/gpu/bus/p2p.c`):
    an `NV50_THIRD_PARTY_P2P` object whose creator's PID is the caller's, a VA
    space registered in it, and a registered video-memory range containing
    the address (`NV503C_CTRL_CMD_REGISTER_VA_SPACE`, `..._REGISTER_VIDMEM`).
    CUDA's user-mode driver creates and fills these. A miss is
    `NV_ERR_OBJECT_NOT_FOUND`, which `nv-p2p.c` maps to `EINVAL`.
  - The non-persistent API without a token (`nvidia_p2p_get_pages(0, 0, ...)`)
    does the same lookup. So a gdrdrv loaded with `use_persistent_mapping=0`
    would most likely fail the same way. The legacy token query
    (`cuPointerGetAttribute(CU_POINTER_ATTRIBUTE_P2P_TOKENS)`) succeeds after
    the restore but returns token 0, and the pins still fail (`gdr_pin -t`).
  - The NIC fails the same way when it uses this API. The CXI driver pins
    with `nvidia_p2p_get_pages_persistent()` too (`nv_p2p_persistent=Y`).
    With libfabric's dma-buf use off (`FI_HMEM_CUDA_USE_DMABUF=0`), every
    registration of the restored process fails with `EINVAL`, old and new
    buffers alike. The NIC registrations that worked after a restore all
    went through dma-buf, which is exported from CUDA's own memory handles
    and never consults the registry.
  - What crosses the checkpoint does not matter (`tests/gdr_pin.sh`,
    `gdr_pin -m`). Every pin is released before the checkpoint, and after the
    restore every pin fails, for buffers allocated before and after the
    restore (`cudaMalloc` and `cuMemCreate`):
    - `keep`: the GDRCopy handle stays open across the checkpoint;
    - `closed`: it is closed before the checkpoint and reopened after;
    - `fresh`: GDRCopy is first opened after the restore, as in a restarted
      netsplit lower half.
  - So after `cuCheckpointProcessRestore`, the process's memory is missing
    from the third-party P2P registry, even memory allocated after the
    restore. The kernel module has no checkpoint-specific code; the restore
    is the closed user-mode driver's, and it evidently does not re-create
    that registration. NVIDIA lists RDMA among the CUDA checkpoint's
    unsupported features.
  - Ways around it: dma-buf, which works after a restore for the NIC; for
    CPU mappings, GDRCopy's dma-buf mmap backend (no gdrdrv), which needs
    CUDA driver 13.3 or newer, so not on Perlmutter (see the next item); or a
    fix in NVIDIA's restore path.
- **No dma-buf stand-in for GDRCopy.** `mmap()` of a CUDA dma-buf
  (`cuMemGetHandleForAddressRange`) fails with error 524 (ENOTSUPP),
  read-write or read-only:
  - for `cudaMalloc` and `cuMemCreate` memory alike;
  - before the checkpoint as well as after.

  NVIDIA's exporter does not support CPU mappings.
- **With the registration cache on**, a buffer registered before the
  checkpoint seems to register again after the restore, but the NIC's writes
  land elsewhere. That is silent data corruption.
- **libfabric needs GDRCopy at run time.** Without it, a small message that
  arrives before its receive is copied into GPU memory with `cudaMemcpy` plus
  `cudaDeviceSynchronize` (`hmem_cuda.c`). That waits for NCCL's kernel, which
  waits for the proxy thread doing the copy. NCCL with GDR and
  `FI_HMEM_CUDA_USE_GDRCOPY=0` hangs even natively (stack captured).
- **`tests/fi_cudart_shim.c` removes the deadlock natively.** It is a
  preloaded `libcudart.so` that, for callers inside libfabric only, copies on
  a private non-blocking stream and waits for that stream.
  - With it, the 1x2 allreduce over OFI with GDR and no GDRCopy passes
    natively: 121 µs at 1 MiB, the same as with GDRCopy.
  - Under DMTCP it hung at first, with or without a checkpoint. The cause is
    DMTCP's dl plugin (`dmtcp/src/plugin/dl/dlwrappers.cpp`). Its `dlopen`
    wrapper turns libfabric's `dlopen("libcudart.so")` into full paths built
    from `LD_LIBRARY_PATH`, which bypasses the lookup by SONAME that finds the
    preloaded shim. So libfabric got the real runtime, and its
    `cudaDeviceSynchronize`, back. `dmtcp_launch --disable-dl-plugin` fixes
    it; netsplit's `nccl_launch` already passes that option.
- **With `FI_HMEM=system`**, as in netsplit's lower half, libfabric cannot
  register GPU memory at all, not even a dma-buf.

**GDR across a checkpoint works with the single-process plugin.** Job 59582340
(nid200276, driver 580 with CUDA 13.2's compatibility `libcuda`) ran
`tests/report`'s `nccl_allreduce` at 1x2 with the network forced and these
settings:

- `NCCL_PROFILE=gdr NET_PROFILE=ofi`;
- `FI_HMEM_CUDA_USE_GDRCOPY=0`;
- the shim in `LD_PRELOAD`;
- `dmtcp_launch --disable-dl-plugin` (through a copy of `launch_rank.sh`).

NCCL logs `via NET/AWS Libfabric/N/GDRDMA` before the checkpoint and again
after the rebuild.

| Mode | Passed | us_at_max (1 MiB), GDR | Same, host staging (no GDR, no shim) |
|---|---|---|---|
| checkpoint-resume | 4 of 4 | 279–328 | 616 |
| checkpoint-restart | 4 of 4 | 280 | 535 |

- us_at_max is the benchmark's average latency at its largest size, in a run
  that includes the checkpoint. Compare it only within this table.
- These are restarts on the same node. The single-process plugin still
  cannot restart a multi-node job over OFI.

For netsplit this means GDR needs libfabric's CUDA support in the lower half.
That is the runtime calls listed in `fi_cudart_shim.c` plus the driver calls
in libfabric's CUDA layer, forwarded to the upper half's single CUDA driver,
with the private-stream copy and GDRCopy off. Those calls run only on NCCL
threads that entered through the proxy.

## GDR in netsplit

`NETSPLIT_GDR=1` turns it on:

- The lower half loads `stub/cuda_fwd.c`, built as `libcuda.so.1`,
  `libcudart.so` and `libnvidia-ml.so.1`. Each call runs the upper half's
  driver on the calling NCCL thread's own TLS, found in the proxy's table.
- libfabric's runtime calls are rebuilt on the driver API. Its host-memory
  pinning is refused.
- Its copies between host and GPU memory go through a pinned bounce buffer, on
  a private stream. The proxy makes both before the back end starts.
- The stub forwards aws-ofi-nccl's pointer query and flush.
- Two libfabric settings change: GDRCopy is off, and rendezvous messages carry
  no eager data (`FI_CXI_RDZV_EAGER_SIZE=0`).

**Checkpoints.** Job 59584050 (nid200360 and nid200361), with the final
settings:

| Mode | Program, layout | Passed |
|---|---|---|
| resume, restart | bw 2x1 | 4 of 4, 4 of 4 |
| resume, restart | allreduce 2x2 | 2 of 2, 2 of 2 |
| resume, restart | alltoall 2x2 | 2 of 2, 2 of 2 |
| resume, restart | IS 2x2 | 1 of 1, 2 of 2 |
| restart, other node | allreduce 1x2, network forced | 2 of 2 |

These counts include the runs made after the bounce-buffer fix, before
`FI_CXI_RDZV_EAGER_SIZE=0` became the default. NCCL logs
`via NET/AWS Libfabric/N/GDRDMA` before every checkpoint and again after every
rebuild.

**The bw checkpoint hang, fixed.** The first version hung every bw checkpoint
(job 59583126).

- **What was stuck.** Stacks with every library's symbols (DMTCP's
  `load-symbols-library-all`, then addresses resolved through the process's
  maps) showed three threads:
  - the checkpoint thread in `cudaStreamSynchronize`, waiting for NCCL's
    queued work;
  - the application's main thread in a synchronous `cudaMemcpy` into pageable
    host memory, waiting for NCCL's kernel;
  - on one rank, NCCL's proxy thread blocked on a futex in the same part of
    the driver, inside a forwarded copy of libfabric's pageable buffer.
- **The cause.** The driver serializes pageable copies through its staging
  buffers, so the proxy's copy waited behind the main thread's. The main
  thread waited for the kernel, which waited for the proxy.
- **The fix.** The forwarders copy through a pinned bounce buffer. Making that
  buffer lazily, in the middle of NCCL's traffic, blocked the same way (a
  plain launch hung), so the proxy makes it, with the private stream, before
  `backend_load()`.

**Tuning: no eager data on rendezvous messages.** A message that arrives
before its receive is posted leaves its eager part in a host buffer. Without
GDRCopy, that part is copied into GPU memory with a DMA and a stream sync
instead of CPU stores. Perlmutter's module sets `FI_CXI_RDZV_THRESHOLD=0`,
which libfabric raises to 192 bytes, so almost every message is rendezvous
with an eager part. Netsplit GDR launches with each setting, at 64 MiB (job
59584050, `tune`):

| Setting | bw 2x1 | alltoall 2x2 | allreduce 2x2 |
|---|---|---|---|
| none (before) | 8.4 GB/s | 15.7 ms | 2.71 ms |
| `FI_CXI_RDZV_EAGER_SIZE=0` | 23.4 GB/s | 7.4 ms | 2.79 ms |
| `NCCL_P2P_NET_CHUNKSIZE=2097152` | 23.4 GB/s | 7.1 ms | |
| both | 23.5 GB/s | 7.7 ms | |
| `NCCL_BUFFSIZE=16777216` | | | 2.68 ms |

- At 1 MiB, bw is 8.5, 16.1, 12.6 and 19.4 GB/s with these settings.
- At 64 KiB, allreduce takes 77 µs and 57 µs, before and with eager size 0.
- The lower half now sets `FI_CXI_RDZV_EAGER_SIZE=0` in GDR mode.

**Performance without a checkpoint**, with the final settings, against native
NCCL with GDR on the same nodes (job 59584050, `perf2`, one run each):

`bw` 2x1 (MB/s)

| Size | NCCL, GDR | netsplit, GDR | netsplit vs NCCL | netsplit, host staging (job 59574333) |
|---:|---:|---:|---:|---:|
| 8 B | 0.6 | 0.3 | 0.57 | 0.5 |
| 1 KiB | 75.0 | 43.2 | 0.58 | 67.1 |
| 64 KiB | 2611.3 | 2645.8 | 1.01 | 2195.3 |
| 1 MiB | 15958.4 | 16016.6 | 1.00 | 12044.5 |
| 4 MiB | 21040.2 | 21133.1 | 1.00 | 16127.4 |
| 16 MiB | 22800.6 | 22907.2 | 1.00 | 17600.4 |
| 64 MiB | 23343.1 | 23396.7 | 1.00 | 17777.9 |


`allreduce` 2x2 (µs)

| Size | NCCL, GDR | netsplit, GDR | netsplit vs NCCL | netsplit, host staging (job 59574333) |
|---:|---:|---:|---:|---:|
| 8 B | 24.7 | 25.3 | 0.98 | 29.3 |
| 1 KiB | 26.4 | 26.7 | 0.99 | 32.3 |
| 64 KiB | 55.0 | 56.6 | 0.97 | 69.1 |
| 1 MiB | 193.5 | 195.6 | 0.99 | 258.8 |
| 4 MiB | 232.9 | 232.7 | 1.00 | 384.3 |
| 16 MiB | 720.5 | 739.4 | 0.97 | 1454.0 |
| 64 MiB | 2707.6 | 2760.6 | 0.98 | 5746.3 |


`alltoall` 2x2 (µs)

| Size | NCCL, GDR | netsplit, GDR | netsplit vs NCCL | netsplit, host staging (job 59574333) |
|---:|---:|---:|---:|---:|
| 8 B | 45.0 | 70.2 | 0.64 | 39.9 |
| 1 KiB | 43.3 | 45.1 | 0.96 | 40.1 |
| 64 KiB | 62.7 | 63.1 | 0.99 | 65.7 |
| 1 MiB | 173.4 | 179.2 | 0.97 | 269.0 |
| 4 MiB | 510.4 | 527.4 | 0.97 | 912.8 |
| 16 MiB | 1800.3 | 1908.7 | 0.94 | 3448.9 |
| 64 MiB | 6981.9 | 7352.2 | 0.95 | 13694.5 |

- **netsplit with GDR matches native GDR**, except for small point-to-point
  messages:
  - bw: 0.99–1.02 from 4 KiB;
  - allreduce: 0.96–1.03 at every size;
  - alltoall: 0.94–1.00 from 256-byte blocks.
- **Small point-to-point messages are slower**: bw 0.54–0.58 up to 2 KiB, and
  alltoall 0.62–0.65 up to 128-byte blocks. Messages under 192 bytes are
  always eager, so an early one still costs a DMA and a stream sync where
  GDRCopy would take CPU stores.
- **The host-staged column** is from job 59574333, other nodes.

**Also found:** `FI_MR_CUDA_CACHE_MONITOR_ENABLED=0` (no cache of CUDA
registrations) hangs NCCL with aws-ofi-nccl across nodes, natively too. The
lower half leaves the cache on.

## Stress test (2 nodes)

Job 59585527 (nid200388 and nid200389), with `tests/run_ns.sh` and every
restart on the nodes that took the checkpoint. Each workload ran twice:
- `resume5`: 5 checkpoints in one run, each followed by progress;
- `restart3`: 3 generations of checkpoint, kill and restart.

The workloads were `nccl_bench` allreduce, alltoall, allgather,
reduce_scatter and bcast at 2x4, bw at 2x1, NAS IS class B at 2x4, and
PyTorch DDP at 2x4 (`pytorch_minimal_test.py`, launched by Slurm with no MPI
or torchrun; `tests/rank_env.sh` sets its rank variables).

| Data path | Cases passed | Checkpoints | Restarts |
|---|---|---|---|
| Host staging | 16 of 16 | 64 of 64 | 24 of 24 |
| GDR | 13 of 16 | 56 of 56 | 20 of 21 |

- Checkpoints took 3.8–4.8 s for `nccl_bench` and IS at 2x4, 2.8–3.5 s for
  bw at 2x1, and 8.9–10.7 s for PyTorch.
- Restarts took 2.6–9.2 s.

**GDR failures:**
- **bw 2x1, third restart of a chain:** CUDA restore error 304 (see "Open").
- **PyTorch, both cases: a deadlock.** The forwarders cannot fix it.
  - It happens before any checkpoint too: one reproduction hung in the first
    training step.
  - Stacks: on one rank, NCCL's proxy thread is in libfabric's copy of an
    early small message into GPU memory, which goes through the forwarders'
    `cudaMemcpy` to `cuMemcpyHtoDAsync`. Inside the driver it is blocked on
    `pthread_rwlock_rdlock`.
  - Meanwhile Python's main thread is inside a driver call, waiting for
    NCCL's kernel. That kernel waits for the proxy.
  - A device-to-device copy from a GPU-mapped bounce buffer blocks the same
    way, so the lock covers more than host-to-device copies. The code keeps
    the pinned host-to-device copy.
  - GDRCopy avoids the problem because its copies are CPU stores that never
    call the driver. Until GDRCopy works after a restore (see "GDR after a
    checkpoint"), GDR mode is not safe for PyTorch.

Host staging passed every case, PyTorch included: its back end makes no CUDA
call at all.

## Fixed during phase 1

- **Restart on another node.** NCCL computes some host state once per process
  and keeps it in the image:
  - the bootstrap interface (`bootstrapNetIfAddr`, guarded by
    `bootstrapNetInitDone`, which only `ncclInit()`'s `call_once` sets);
  - the host hash (`getHostHash()::once`).

  Without the fix, the rebuild's `ncclGetUniqueId()` binds the old node's
  address. At RESTART, `upper/netsplit_dmtcp.cpp` finds these local symbols in
  libnccl's `.symtab`, clears both flags and calls `bootstrapNetInit()` again.
  `NETSPLIT_NCCL_RESET=0` turns this off.
- **A second checkpoint after a restart segfaulted** in glibc's nscd client
  (`__nscd_get_nl_timestamp`), called from NCCL's teardown.
  - DMTCP saves an nscd database mapping as one zero-page header with no
    parent (`dmtcp/src/writeckpt.cpp`).
  - The restore code, copied from `mtcp_restart.c`, assumes that every
    zero-page header has a parent that already mapped the range. So it never
    mapped the area, and glibc's handle pointed to unmapped memory.
  - `lower/netsplit-lh.cpp` now maps such an area.
  - DMTCP's `src/mtcp/mtcp_restart.c` and MANA's `lower-half/lower-half.cpp`
    have the same code.
- **`tests/run_ns.sh` collects DMTCP's errors.** `nccl_launch` and
  `nccl_restart` pass `-q -q`, which hides JASSERT messages. The driver now
  sets `JALIB_STDERR_PATH` to `dmtcp-errors.log` in each case directory.

## Open

- **`cuCheckpointProcessRestore` sometimes fails** with 304
  (`CUDA_ERROR_OPERATING_SYSTEM`) in DMTCP's CUDA plugin (`cuda-ckpt.cpp:287`).
  - It happened in 5 of about 70 restarts, always on one rank only: bw 2x1
    three times, IS 1x2 on another node, and a one-rank restart.
  - Not an address collision with the new lower half: with
    `NETSPLIT_SAVE_MAPS=1`, the failing rank's `/proc/self/maps` at
    DMTCP_EVENT_RESTART is identical to a passing run's, apart from DMTCP's
    shared-area file names.
  - By its summary, none of the 16 restarts in the single-process ablation of
    2026-10-07 failed this way. So it may come from something that only
    netsplit does before the restore.
  - Next steps:
    - run the single-process plugin through the same driver, to compare the
      failure rate;
    - wait longer between the kill and the restart;
    - log the driver's fds at restore.
- **alltoall 2x2 sometimes hangs after the rebuild** with host staging: all
  ranks finish rebuilding, then make no progress.
  - It happened in the first run (a resume), before any of today's fixes,
    and again in the regression run of job 59584050 (a restart). One more
    restart in the first run exited with code 99, before DMTCP's errors were
    logged.
  - That is 3 of about 15 alltoall 2x2 runs; no stacks yet.
- **The coordinator refused one checkpoint request** ("computation not in
  running state") in the final alltoall 2x2 restart run. The program then ran
  to `validation PASSED`.
- **Every restart test kills the old job.** After its checkpoint, the job is
  killed while nccl-mana-ckpt publishes rebuild ids. As a result, every case's
  `dmtcp-errors.log` holds one `kvdbRequest` JASSERT from the killed job. This
  is expected.
- **Not yet run:**
  - 2x4 and 4x4;
  - the `latency` benchmark;
  - moving a multi-node job to other nodes (needs 4 nodes);
  - several communicators, communicator churn and `ncclCommSplit`;
  - PyTorch DDP;
  - repeated performance runs.
- **GDR in netsplit:**
  - PyTorch DDP deadlocks in GDR mode, because the proxy thread's CUDA
    copies wait on a driver lock that an application thread holds (see
    "Stress test"); fixing it needs GDRCopy or a copy that bypasses the
    driver;
  - small point-to-point messages are slower without GDRCopy. GDRCopy
    fails because a restored process is missing from NVIDIA's third-party
    P2P registry ("GDR after a checkpoint"), which a gdrdrv module setting
    cannot fix; GDRCopy's dma-buf backend (CUDA driver 13.3 or newer) or a
    fix in NVIDIA's restore path could;
  - 2x4 and 4x4 have not been run with GDR yet.

  `NETSPLIT_GDR=1` passes the back end's pointer support through, but on its
  own it cannot work: the lower half's libfabric has no CUDA.
