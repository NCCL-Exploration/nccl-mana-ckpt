# netsplit results (2026-10-08)

netsplit runs NCCL's network back end in a lower half that is never
checkpointed (see README.md). These are the prototype's first results, on
Perlmutter, against NCCL without checkpointing.

**Setup.** Two GPU nodes per job: 4 A100s and 4 Slingshot-11 NICs (CXI) each.

- Software: NCCL 2.29.2-cu13, aws-ofi-nccl 1.6.0 through net API v6,
  libfabric 2.3.1, CUDA 13.2, driver 595.91.07, nccl-mana-ckpt 6b1c912 plus
  `netsplit/`.
- Programs: `nccl_bench` (no MPI) and NAS IS class B over NCCL.
- Layouts are NODESxGPUS-PER-NODE with one rank per GPU.
- Jobs: 59571523 (phase 0), 59574333 and 59576795 (phase 1).
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
- **All of the cost against production NCCL is host staging.** GDR cannot
  survive a CUDA checkpoint on this system (phase 0b), so traffic is staged
  through host memory. Against NCCL with GDR:
  - up to 64 KiB: allreduce is 4–16% slower (up to 6 µs). bw is 1–13%
    faster and alltoall is 2–17% faster, because GDR is slower there;
  - from 1 MiB: 78–83% of bandwidth;
  - for a 64 MiB message: about twice the time for allreduce (2.1x) and
    alltoall (1.9x).

## Phase 0: can the back end run alone?

| Test | Result |
|---|---|
| 0a: aws-ofi-nccl driven through ncclNet v6 by a CUDA-free program, with only the CUDA stub | Passes on 1 and 2 nodes: 200 x 4 MiB at 21.6 GB/s, the same as with the real CUDA runtime. The only CUDA library mapped is the stub. |
| 0b: register GPU memory with the NIC after `cuCheckpointProcessRestore` | Fails for dma-buf and peer memory (`cxil_map` returns -22). Phase 2 (GDR) is dropped. |

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
  - It happened in 4 of about 40 restarts, always on one rank only: bw 2x1
    twice, IS 1x2 on another node, and a one-rank restart.
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
- **The first alltoall 2x2 runs failed** before any of today's fixes: 1 resume
  hung and 1 restart exited with code 99. Neither was reproduced in later
  runs, and DMTCP's error log was not collected then.
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
- **GDR before the first checkpoint.** The proxy could pass GDR support
  through until the first checkpoint (`NETSPLIT_GDR=1` does it for a whole
  run). That would only help jobs that run long before they first checkpoint.
