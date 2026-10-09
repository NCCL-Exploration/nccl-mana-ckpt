# Test plan: netsplit, tech-report evaluation

This plan measures netsplit: NCCL checkpointing with NCCL's network back end
(aws-ofi-nccl, libfabric) in a lower half that is never checkpointed. It
follows `tests/report/TEST_PLAN.md`, the plan for the single-process plugin,
with these changes:
- one scaling series, with full nodes whenever more than one node is used;
- the point-to-point microbenchmarks (`latency`, `bw`) run only at 1x2 and
  2x1;
- checkpoint and restart times are measured only for NAS IS and PyTorch;
- every restart runs on the nodes the job was checkpointed on;
- this round uses host staging only, with no GPUDirect RDMA (§5);
- every phase of every run has a timeout, so a hung run is recorded and the
  campaign continues (§7).

**No MPI anywhere.** netsplit supports NCCL only. Every process is a plain
`srun` task, one per GPU:
- Ranks come from `SLURM_PROCID`, `SLURM_NTASKS` and `SLURM_LOCALID`.
- PyTorch gets `RANK`, `WORLD_SIZE` and `LOCAL_RANK` from `tests/rank_env.sh`.
- `nccl_bench` and IS exchange their `ncclUniqueId` through a file on
  `$SCRATCH`, and PyTorch uses a FileStore there.

## 1. Questions the evaluation answers

1. **Correctness.** Is the result verified after a NAS IS or PyTorch DDP job
   is checkpointed mid-run and then resumed, or restarted on the same nodes?
2. **Runtime overhead.** How much slower is NCCL under netsplit (DMTCP,
   nccl-mana-ckpt, the CUDA plugin and the split network back end) than
   production NCCL, with no checkpoint?
3. **Checkpoint cost** (IS and PyTorch only):
   - How long does a checkpoint take?
   - How is that time split between NCCL's drain and teardown, and the CUDA
     checkpoint plus the image write?
   - How large are the images?
4. **Restart cost** (IS and PyTorch only):
   - How long until every rank is restored and its communicators rebuilt?
   - How long until the job completes?
5. **Scaling.** How do 2–4 change from 2 GPUs (collectives) or 1 full node
   (IS, PyTorch) to 4 full nodes?

## 2. Scaling series

| Layout (nodes x GPUs per node) | GPUs | Used by |
|---|---|---|
| 1x2 | 2 | collectives; `latency` and `bw` (intra-node) |
| 1x4 | 4 | collectives, NAS IS, PyTorch |
| 2x4 | 8 | collectives, NAS IS, PyTorch |
| 4x4 | 16 | collectives, NAS IS, PyTorch |
| 2x1 | 2 | `latency` and `bw` only (inter-node) |

- **Multi-node runs use every GPU** (4 per node). The only exception is 2x1,
  for the point-to-point microbenchmarks.
- **4x4 is the top of the series.** The interactive QOS covers it (up to 4
  nodes).
- **NAS IS and PyTorch run at 1x4, 2x4 and 4x4.** IS class D needs at least
  4 ranks (§3), and PyTorch uses the same layouts so the two can be compared
  point by point. So all checkpoint and restart measurements are at full
  nodes.
- **No 1x1.** `nccl_bench` collectives mean nothing with one rank, and IS and
  PyTorch start at 1x4.
- **Single-node layouts (1x2, 1x4) never use the network.** NCCL uses
  NVLink (P2P/cuMem) or SHM within a node, so netsplit's lower half is
  loaded but idle. These points measure netsplit without the network, and
  2x4 and 4x4 measure it with the network.

## 3. Workloads

| Benchmark | Layouts | What it exercises | Correctness check | Metric |
|---|---|---|---|---|
| `nccl_bench -b latency\|bw` | 1x2 (intra-node), 2x1 (inter-node) | NCCL point-to-point between 2 ranks | `validation PASSED` | latency (µs) / bandwidth (MB/s), 8 B – 64 MiB |
| `nccl_bench -b allreduce\|allgather\|reduce_scatter\|bcast\|reduce\|alltoall` | 1x2, 1x4, 2x4, 4x4 | NCCL collectives | exact fp32 values per size | avg/min/max latency (µs), 8 B – 64 MiB |
| NAS IS, class D | 1x4, 2x4, 4x4 | `ncclAllReduce`, all-to-all point-to-point, `ncclReduce` | NPB `Verification = SUCCESSFUL` | Mop/s |
| PyTorch DDP (`tests/multi/pytorch_minimal_test.py`) | 1x4, 2x4, 4x4 | `ProcessGroupNCCL`, bucketed allreduce | `PASS pytorch_minimal` on every rank, plus an exact `all_reduce` each step | steps/s of the slowest rank |

- **Benchmark details:** `nccl_bench`, run lengths in seconds rather than
  iterations, and the PyTorch workarounds (`pyshim`, `USE_LIBUV=0`,
  `--no-gzip`) are all as in `tests/report/TEST_PLAN.md` §3.
- **IS class D** (`$SCRATCH/nccl-ckpt-report/bin/is_nccl_D`):
  - It sorts 2^31 keys in total at every layout (strong scaling).
  - It needs at least 4 ranks (`MIN_PROCS 4` in `is_nccl.cu`, to avoid
    32-bit overflow).
  - Its keys live in pinned host memory (`cudaMallocHost`), about 6 GiB per
    rank at 1x4, so each checkpoint writes tens of GB at every layout. The
    smoke test records the actual image size.
- **Microbenchmarks are performance runs only.** Their correctness under
  checkpointing comes from the stress test (§9).

## 4. Modes

| Mode | What runs | Workloads | Purpose |
|---|---|---|---|
| `native` | `srun`, production NCCL: module defaults (OFI with GDR, P2P, SHM) | all | reference |
| `ns` | `nccl_launch`, host-staged network, no checkpoint | all | overhead of netsplit |
| `ns_resume` | `ns`, one checkpoint mid-run, then the job continues | IS, PyTorch | checkpoint cost, resume rebuild, correctness |
| `ns_restart` | `ns`, one checkpoint, kill, `nccl_restart` on the same nodes, run to the end | IS, PyTorch | restart cost, correctness |

- **`native` runs NCCL as users run it.** Unlike the single-process plan,
  there is no checkpoint-safe NCCL profile: netsplit checkpoints with
  production settings, so overhead is measured against production NCCL. On
  2x4 and 4x4, `native` uses GPUDirect RDMA and `ns` stages through host
  memory, so the network part of the overhead includes the cost of host
  staging.
- **Modes are interleaved** within each repetition, so drift over a job
  affects all of them equally.

## 5. Not in this round

- **GPUDirect RDMA (`NETSPLIT_GDR=1`).** It matches native GDR speed for
  `nccl_bench`, but PyTorch DDP deadlocks in it (RESULTS.md, "Stress test"),
  and GDRCopy cannot pin restored memory on this system.
- **Restarting on other nodes.**
- **8x4 and larger.**

## 6. Same-node restarts

- A restart uses the same allocation, the same `srun` geometry (`-N`,
  `--ntasks-per-node`), and the same node list (`-w`) as the launch. So
  every rank comes back on the node and GPU it was checkpointed on.
- Each run records the rank-to-host map at launch and at restart, and fails
  if they differ.

## 7. Timeouts

Each phase of a run has its own limit. On any timeout, the driver:
1. records the phase as the run's status;
2. kills the job (`dmtcp_command --kill`, then SIGTERM to its `srun`, then
   SIGKILL after 10 s);
3. stops the coordinator;
4. keeps the logs and the first error lines (`dmtcp-errors.log`);
5. moves on to the next run.

Hung runs count as failures in the success rate.

| Phase | Limit (default) | Status if exceeded |
|---|---|---|
| Startup: first progress marker | 600 s (`STARTUP_TIMEOUT`; PyTorch's import and NCCL init under DMTCP) | `hang_startup` |
| Checkpoint: coordinator writes the restart script | 300 s (`CKPT_TIMEOUT`; IS class D writes tens of GB) | `hang_ckpt` |
| Progress after a resume | 120 s (`PROGRESS_TIMEOUT`) | `hang_after_resume` |
| Restart: every rank prints `Rebuild overhead` | 600 s (`RESTART_TIMEOUT`) | `hang_restart` |
| Completion after the last checkpoint or restart | run target + 300 s | `hang_run` |
| Any single `srun` step | 1800 s (`RUN_TIMEOUT`, hard cap) | `timeout` |

Two more guards:
- **Between runs,** the driver kills any rank (`netsplit-lh`) or coordinator
  left on the job's nodes, so a hung run cannot disturb the next one.
- **Before each run,** the driver compares the job's remaining time with the
  run's worst case (all limits above). If the job would end mid-run, it
  skips the remaining runs and says so, rather than being killed by Slurm
  without a result.

`tests/run_ns.sh` has the per-phase limits for the stress test, and the
report harness has them for every mode (`tests/report/common.sh` holds the
defaults). Runs that the time guard skips are listed in the campaign's
`skipped.txt` and in the summary.

## 8. Metrics

| Metric | Definition | Source |
|---|---|---|
| Runtime overhead | `ns` vs `native` on the application metric (per size for `nccl_bench`); positive means slower | performance runs |
| Checkpoint time | `dmtcp_command -c` until the coordinator writes the restart script, i.e. every image is on disk | driver timer |
| NCCL drain + teardown | `[NCCL PLUGIN] Checkpoint overhead`, max over ranks | nccl-mana-ckpt log |
| CUDA + image write | checkpoint time − NCCL drain (derived) | report |
| Image size | sum of `ckpt_*.dmtcp` (`--no-gzip`), total and per rank | `ls` of the image directories |
| Write bandwidth | image bytes / checkpoint time | derived |
| Resume rebuild | `[NCCL PLUGIN] Rebuild overhead` after a resume, max over ranks | nccl-mana-ckpt log |
| Restart latency | `srun nccl_restart` until every rank has printed `Rebuild overhead` | driver timer + log |
| Restart total | `srun nccl_restart` until the job completes | driver timer |
| Success rate | `ok` runs / runs, per configuration; failures and hangs are kept and listed with their status and first error | driver status |

## 9. Robustness (already run, report as is)

The stress test of 2026-10-08 (job 59585527, 2 nodes), host staging,
same-node restarts:
- **Runs:**
  - `nccl_bench` allreduce, alltoall, allgather, reduce_scatter and bcast at
    2x4, `bw` at 2x1, NAS IS (class B) at 2x4 and PyTorch DDP at 2x4;
  - each with 5 checkpoints in one run and with 3 checkpoint, kill and
    restart generations.
- **Result:** 16 of 16 cases passed, with 64 checkpoints and 24 restarts.

`tests/run_ns.sh` reproduces it (the `resumeN` and `restartN` modes).

## 10. Procedure

1. **Extend the report harness for netsplit** (`tests/report`, done).
   `PLAN=netsplit` selects this plan:
   - `common.sh`: the layouts of §2, `IS_CLASS=D`, production NCCL
     settings (`NCCL_PROFILE=default NET_PROFILE=ofi`), the limits of §7,
     and campaigns named `ns-<date>`;
   - `launch_rank.sh`: `netsplit` (`nccl_launch --ckptdir`) and
     `netsplit-restart` (`nccl_restart --restartdir`) modes;
   - `run_bench.sh`: the modes of §4 (`native`, `ns`, `ns_resume`,
     `ns_restart`), the limits and kills of §7, and the rank-to-host check
     of §6 (`srun -w`, status `fail_hostmap`);
   - `campaign.sbatch`: each layout's benchmarks and modes, the cleanup
     between runs and the time guard;
   - `summarize.py`: the new modes, a success-rate table that counts hangs,
     and each failed run's first error line.

   All results land in one `results.csv`, as in the single-process plan.
2. **Smoke test:** 1x4 and 2x4, one repetition of every benchmark and mode.
   It also records IS class D's image size and checkpoint time, to confirm
   the 300 s checkpoint limit.

   ```bash
   PLAN=netsplit SMOKE=1 INTERACTIVE=1 tests/report/submit_all.sh
   ```
3. **Campaign:** one job per node count (1, 2 and 4 nodes), `REPS=3`.

   ```bash
   PLAN=netsplit tests/report/submit_all.sh                  # sbatch, QOS=regular
   PLAN=netsplit INTERACTIVE=1 tests/report/submit_all.sh    # interactive QOS, one job at a time
   ```
4. **Tables:** `summarize.py`, as in `tests/report/TEST_PLAN.md` §6.

**Budget** (rough, 3 repetitions; about 1 minute per performance run, 3–4
minutes per PyTorch checkpoint run, 4–5 per IS class D checkpoint run):
- 1 node (1x2 and 1x4, plus the 1x2 point-to-point runs): about 3
  node-hours.
- 2 nodes (2x4, plus the 2x1 point-to-point runs): about 4 node-hours.
- 4 nodes (4x4): about 7 node-hours.
- Total about 14 node-hours. Each job fits the interactive QOS (4 nodes, 4
  hours).

## 11. Decisions (2026-10-08)

- No GPUDirect RDMA this round.
- No 8x4; the series ends at 4x4.
- The point-to-point microbenchmarks are `latency` and `bw`, not `bibw`.
- NAS IS uses class D throughout, so it runs at 1x4, 2x4 and 4x4.
- PyTorch runs at the same layouts as IS: 1x4, 2x4 and 4x4.
- Every phase of every run has a timeout (§7).

## 12. Known risks

- **Intermittent CUDA restore error 304** (`cuCheckpointProcessRestore`, in
  DMTCP's CUDA plugin): about 1 in 45 same-node restarts in the stress
  test, and 0 of 24 with host staging. Failures are kept in the success
  rate, not hidden by retries.
- **IS class D images** are much larger than the class B images of the
  stress test. Lustre write time dominates its checkpoint time and varies
  with load. Report the spread across repetitions.
- **Intermittent hang after the rebuild** in host-staged alltoall 2x2: 3 of
  about 15 runs, none in the stress test. IS uses all-to-all exchanges, so
  watch for it in IS runs (the §7 timeouts record it as
  `hang_after_resume` or `hang_restart`).
