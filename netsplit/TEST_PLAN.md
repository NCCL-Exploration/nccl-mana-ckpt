# Test plan: netsplit, round 2 (multi-clock cut)

This plan measures netsplit: NCCL checkpointing with NCCL's network back end
(aws-ofi-nccl, libfabric) in a lower half that is never checkpointed. Round 2
evaluates the current implementation, whose checkpoint cut runs a collective
clock per communicator and a point-to-point clock per pair of ranks, and puts
communicator creation, split and destruction in the cut (commits 57477b5,
c9274f3, 5d6f2c2). Its results replace round 1's (campaign `ns-20261009`,
which used the plugin of 2026-10-07) in the thesis chapter.

**Changes from round 1:**
- **Baseline.** Runtime overhead is measured against NCCL with GPUDirect RDMA
  off (`native`, §4), the data path netsplit uses. Production NCCL (GDR on)
  runs only the `latency` and `bw` microbenchmarks across nodes
  (`native_ref`), for the limitations.
- **Checkpoint timing.** Every checkpoint run is done with the checkpoint 5 s
  and 10 s after the first progress report (§4). Round 1 used 10 s, where IS
  class D at 2x4 failed, and passed at 5 s in a follow-up.
- **Repeated checkpoints.** The stress test (5 checkpoints in one run; 3
  generations of checkpoint, kill and restart) is part of the campaign (§5),
  instead of a separate run on an earlier build.
- **Hardware.** Jobs take the default GPU nodes (`-C gpu`), as before. The
  node type is recorded per job and reported (§9).
- **Restart time is split** into the communicator rebuild and the rest (§8).

**No MPI anywhere.** netsplit supports NCCL only. Every process is a plain
`srun` task, one per GPU:
- Ranks come from `SLURM_PROCID`, `SLURM_NTASKS` and `SLURM_LOCALID`.
- PyTorch gets `RANK`, `WORLD_SIZE` and `LOCAL_RANK` from `tests/rank_env.sh`.
- `nccl_bench` and IS exchange their `ncclUniqueId` through a file on
  `$SCRATCH`, and PyTorch uses a FileStore there.

## 1. Questions the evaluation answers

1. **Correctness.** Is the result verified after a NAS IS or PyTorch DDP job
   is checkpointed mid-run and then resumed, or restarted on the same nodes,
   once and repeatedly? Does it depend on when the checkpoint is taken?
2. **Runtime overhead.** How much slower is NCCL under netsplit (DMTCP,
   nccl-mana-ckpt, the CUDA plugin and the split network back end) than NCCL
   on the same data path, with no checkpoint?
3. **Checkpoint cost** (IS and PyTorch):
   - How long does a checkpoint take?
   - How is that time split between NCCL (consensus, drain, teardown) and the
     CUDA checkpoint plus the image write?
   - How large are the images?
4. **Restart cost** (IS and PyTorch): how long until every rank is restored
   and its communicators rebuilt, and how much of that is the rebuild?
5. **Scaling.** How do 2–4 change from 2 GPUs (collectives) or 1 full node
   (IS, PyTorch) to 4 full nodes?
6. **Limitation.** How far are netsplit's point-to-point latency and
   bandwidth across nodes from production NCCL with GDR?

## 2. Scaling series

| Layout (nodes x GPUs per node) | GPUs | Used by |
|---|---|---|
| 1x2 | 2 | collectives; `latency` and `bw` (intra-node) |
| 1x4 | 4 | collectives, NAS IS, PyTorch |
| 2x4 | 8 | collectives, NAS IS, PyTorch; stress test |
| 4x4 | 16 | collectives, NAS IS, PyTorch; stress test (IS, PyTorch) |
| 2x1 | 2 | `latency` and `bw` (inter-node); stress test (`bw`) |

- **Multi-node runs use every GPU** (4 per node), except 2x1 for the
  point-to-point microbenchmarks.
- **4x4 is the top of the series.**
- **NAS IS and PyTorch run at 1x4, 2x4 and 4x4.** IS class D needs at least
  4 ranks, and PyTorch uses the same layouts so the two can be compared point
  by point.
- **Single-node layouts (1x2, 1x4) never use the network.** NCCL uses NVLink
  (P2P/cuMem) or SHM within a node, so netsplit's lower half is loaded but
  idle, and GDR settings change nothing. These points measure netsplit
  without the network; 2x1, 2x4 and 4x4 measure it with the network.

## 3. Workloads

| Benchmark | Layouts | What it exercises | Correctness check | Metric |
|---|---|---|---|---|
| `nccl_bench -b latency\|bw` | 1x2, 2x1 | NCCL point-to-point between 2 ranks | `validation PASSED` | latency (µs) / bandwidth (MB/s), 4 B – 64 MiB |
| `nccl_bench -b allreduce\|allgather\|reduce_scatter\|bcast\|reduce\|alltoall` | 1x2, 1x4, 2x4, 4x4 | NCCL collectives | exact fp32 values per size | avg/min/max latency (µs), 4 B – 64 MiB |
| NAS IS, class D | 1x4, 2x4, 4x4 | `ncclAllReduce`, all-to-all point-to-point, `ncclReduce` | NPB `Verification = SUCCESSFUL` | Mop/s |
| PyTorch DDP (`tests/multi/pytorch_minimal_test.py`) | 1x4, 2x4, 4x4 | `ProcessGroupNCCL`, bucketed allreduce | `PASS pytorch_minimal` on every rank, plus an exact `all_reduce` each step | steps/s of the slowest rank |

- **Benchmark details:** `nccl_bench`, run lengths in seconds rather than
  iterations, and the PyTorch workarounds (`pyshim`, `USE_LIBUV=0`,
  `--no-gzip`) are all as in `tests/report/TEST_PLAN.md` §3.
- **IS class D** (`$SCRATCH/nccl-ckpt-report/bin/is_nccl_D`) sorts 2^31 keys
  at every layout (strong scaling). Its keys live in pinned host memory, so
  each checkpoint writes 30–47 GiB (round 1: 7.6 GiB per rank at 1x4, 3.0 GiB
  per rank at 4x4).
- **PyTorch DDP:** 2 x 4096^2 fp32 parameters, batch 64, about 134 MB of
  gradients per step; images of about 5.5 GiB per rank.

## 4. Modes

All modes run with the same NCCL settings, NCCL's defaults with GDR off
(`NCCL_PROFILE=nogdr`: `NCCL_NET_GDR_LEVEL=0`, `NCCL_DMABUF_ENABLE=0`), over
OFI (`NET_PROFILE=ofi`), with NCCL RAS off. The one exception is
`native_ref`.

| Mode | What runs | Workloads | Purpose |
|---|---|---|---|
| `native` | `srun`, NCCL, GDR off | all | baseline: NCCL on netsplit's data path |
| `ns` | `nccl_launch`, no checkpoint | all | overhead of netsplit |
| `ns_resume` | `ns`, one checkpoint, then the job continues | IS, PyTorch | checkpoint cost, resume rebuild, correctness |
| `ns_restart` | `ns`, one checkpoint, kill, `nccl_restart` on the same nodes, run to the end | IS, PyTorch | restart cost, correctness |
| `ns_resume5` | `ns`, 5 checkpoints in one run, each followed by progress | stress set (§5) | repeated checkpoints |
| `ns_restart3` | `ns`, 3 generations of checkpoint, kill, restart | stress set (§5) | repeated restarts |
| `native_ref` | `srun`, production NCCL (module defaults: GDR on) | `latency`, `bw` at 2x1 | limitations only |

- **`ns` runs under the same profile.** netsplit ignores the GDR settings;
  the profile keeps every other NCCL variable identical to `native`'s.
- **Why GDR off is the baseline.** netsplit's proxy reports host memory only,
  so NCCL stages every network transfer through host buffers, whatever
  `NCCL_NET_GDR_LEVEL` says. NCCL with GDR off takes the same data path, so
  `ns` against `native` isolates netsplit's own cost: DMTCP, the plugin's
  wrappers and gate, and the proxy's crossing into the lower half. Within a
  node, `native` is also production NCCL, since no network is used.
- **Checkpoint delays.** `ns_resume` and `ns_restart` run at
  `CKPT_DELAY` = 5 and 10 s, the time from the application's first progress
  report to `dmtcp_command -c`. Each delay counts as its own configuration in
  the success rate, so a timing dependence shows as a difference between the
  two.
- **Modes are interleaved** within each repetition, so drift over a job
  affects all of them equally.
- **Repetitions:** 3 for every configuration except the stress test, which
  runs once per case (each case already takes 3 or 5 checkpoints).

## 5. Stress test (repeated checkpoints)

The same set as round 1's separate stress test (job 59585527), run inside
the campaign with the round-2 build:

| Workload | Layout | Modes |
|---|---|---|
| `nccl_bench` allreduce, alltoall, allgather, reduce_scatter, bcast (4 MiB) | 2x4 | `ns_resume5`, `ns_restart3` |
| `nccl_bench` bw (4 MiB) | 2x1 | `ns_resume5`, `ns_restart3` |
| NAS IS class D | 2x4, 4x4 | `ns_resume5`, `ns_restart3` |
| PyTorch DDP | 2x4, 4x4 | `ns_resume5`, `ns_restart3` |

- **20 cases**: 80 checkpoints and 30 restarts if all pass.
- **Timing:** each checkpoint lands `STRESS_CKPT_DELAY` = 5 s after the
  program reports progress: first at launch, then again after each resume or
  restart. Each run is `STRESS_CYCLE_S` = 20 s longer for every checkpoint
  after the first.
- **Order:** each job runs its stress cases before the rest of their layout.
  Their worst case is the longest (about 67 minutes for an IS `ns_restart3`
  with every limit of §7 reached), so the time guard would otherwise skip
  them near the end of the job.
- **A case passes** if every checkpoint completes, the program makes progress
  after each, and it verifies at the end (`validation PASSED`, NPB
  `SUCCESSFUL`, `PASS pytorch_minimal` on every rank).
- **Recorded per checkpoint** (`ckpts.csv`): checkpoint time, NCCL share,
  image size, consensus rounds, drained messages, the rebuild after a resume;
  per restart: restart latency and rebuild. Each restart is also checked
  against the launch's rank-to-host map (§6).

## 6. Same-node restarts

- A restart uses the same allocation, the same `srun` geometry (`-N`,
  `--ntasks-per-node`), and the same node list (`-w`) as the launch. So
  every rank comes back on the node and GPU it was checkpointed on.
- Each run records the rank-to-host map at launch and at every restart, and
  fails if they differ.
- **Not in this round:** restarting on other nodes, GDR mode
  (`NETSPLIT_GDR=1`), 8x4 and larger.

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
| Startup: first progress marker | 600 s (`STARTUP_TIMEOUT`) | `hang_startup` |
| Checkpoint: coordinator writes the restart script | 300 s (`CKPT_TIMEOUT`), per checkpoint | `hang_ckpt` |
| Progress after a resume | 120 s (`PROGRESS_TIMEOUT`), per checkpoint | `hang_after_resume` |
| Restart: every rank prints `Rebuild overhead` | 600 s (`RESTART_TIMEOUT`), per restart | `hang_restart` |
| Completion after the last checkpoint or restart | run target + 300 s | `hang_run` |
| Any single `srun` step | 1800 s (`RUN_TIMEOUT`, hard cap) | `timeout` |

Two more guards:
- **Between runs,** the driver kills any rank (`netsplit-lh`) or coordinator
  left on the job's nodes, so a hung run cannot disturb the next one.
- **Before each run,** the driver compares the job's remaining time with the
  run's worst case (all limits above, times the number of checkpoints). If
  the job would end mid-run, it skips the remaining runs and lists them in
  `skipped.txt` and in the summary.

## 8. Metrics

| Metric | Definition | Source |
|---|---|---|
| Runtime overhead | `ns` vs `native` (GDR off) on the application metric, per size for `nccl_bench`; positive means slower | performance runs |
| Gap to production NCCL | `ns` vs `native_ref` for `latency` and `bw` at 2x1 | performance runs |
| Checkpoint time | `dmtcp_command -c` until the coordinator writes the restart script, i.e. every image is on disk | driver timer |
| NCCL share | `[NCCL PLUGIN] Checkpoint overhead`, max over ranks: consensus, drain, teardown | nccl-mana-ckpt log |
| CUDA + image write | checkpoint time − NCCL share | derived |
| Consensus rounds, drained messages | `sequence consensus converged in N round(s)`; `[NCCL Drain]` lines | nccl-mana-ckpt log |
| Image size | sum of `ckpt_*.dmtcp` (`--no-gzip`), total and per rank | `ls` of the image directories |
| Write rate | image bytes / checkpoint time, total and per node | derived |
| Resume rebuild | `[NCCL PLUGIN] Rebuild overhead` after a resume, max over ranks | nccl-mana-ckpt log |
| Restart latency | `srun nccl_restart` until every rank has printed `Rebuild overhead` | driver timer + log |
| Restart rebuild | `Rebuild overhead` after the restart, max over ranks; restart latency − rebuild is the rest (lower half, image read, GPU restore) | nccl-mana-ckpt log |
| Restart total | `srun nccl_restart` until the job completes | driver timer |
| Success rate | `ok` runs / runs, per configuration (benchmark, layout, mode, checkpoint delay); failures and hangs are kept and listed with their status and first error | driver status |

For restart failures, the first error must come from the restart's own logs
(`restart.log`, the restart's `dmtcp-errors.log`), not from the killed
original job, whose `kvdbRequest` JASSERT (exit code 99) is expected.

## 9. Hardware record

Jobs use the default GPU nodes of the regular QOS (`-C gpu`); round 1 got
A100-SXM4 40 GB nodes this way. For every job, `tests/report/sysinfo.sh`
already records, per node:
- GPU model, memory, driver, PCI IDs, maximum SM clock, CUDA version and the
  `nvidia-smi topo -m` matrix;
- CPU model and counts, memory size, OS and kernel;
- the Slingshot interfaces and their speed, and the CXI devices from `fi_info`;

and once per job: the repo commit, the DMTCP version, the plugin's checksum,
the modules, NCCL, libfabric, PyTorch, the filesystem and its striping, and
the campaign settings. The summary gains a table of the node types per job,
and each run's `meta.env` keeps its node list, so every result can be traced
to its GPUs.

## 10. Procedure

1. **Harness changes** (`tests/report`, done):
   - `common.sh`: the netsplit plan defaults to `NCCL_PROFILE=nogdr` and
     `CKPT_DELAYS="5 10"`, and its campaigns are named `ns2-<date>`.
     `mode_info` reads a checkpoint count from the mode's name
     (`ns_resume5`, `ns_restart3`), and `run_worst_case_s` covers every
     checkpoint and restart of a run.
   - `campaign.sbatch`: `native_ref` runs only `REF_BENCHES`
     (`nccl_latency nccl_bw`) at `REF_LAYOUTS` (`2x1`); checkpoint modes run
     once per delay in `CKPT_DELAYS`; `STRESS_CASES` (`bench:layout`) run once
     in each of `STRESS_MODES`, first.
   - `run_bench.sh`: N checkpoints or restart generations per run, each with
     the limits of §7; status `hang_after_restart` when a restored job makes
     no progress before its next checkpoint, and `ckpt_missed` when a run
     ends before its last checkpoint. Measurements read only what the logs
     gained since each checkpoint. The checkpoint delay is in the run
     directory (`…/<mode>/d<delay>/rep<r>`) and in `results.csv`
     (`ckpt_delay_s`, with `n_ckpt`, `n_restart`, `rounds_max`, `drained`);
     one row per checkpoint goes to `ckpts.csv`. A failed restart's first
     error comes from the restarts' logs, not the killed job's.
   - `summarize.py`: the delay as part of each configuration, `restart other`
     (restart latency − rebuild), a table of the repeated-checkpoint runs, and
     the node types of every job.
   - `submit_all.sh`: the wall limits below, and the smoke test of step 2.
2. **Smoke test** on the interactive QOS, one 2-node job, `REPS=1`: every
   mode at 2x1, 1x4 and 2x4, both delays, and the stress cases `nccl_bw:2x1`
   and `pytorch:2x4` (58 runs), to check the harness changes and the run
   lengths:

   ```bash
   PLAN=netsplit SMOKE=1 INTERACTIVE=1 tests/report/submit_all.sh
   ```
3. **Campaign:** one job per node count, `REPS=3`, regular QOS:

   ```bash
   PLAN=netsplit tests/report/submit_all.sh
   ```

4. **Tables and figures:** `summarize.py`; the thesis chapter's
   `scripts/snapshot.py` and `scripts/make_figures.py`.

**Budget** (from round 1's run times: a job takes about 1.12 times the sum of
its runs; mean run 26 s for `nccl_bench`, 79 s for IS, 26 s for PyTorch,
about 98 s and 66 s for IS and PyTorch with a checkpoint, about 4 minutes per
stress case):

| Job | Runs | Estimate | Wall limit |
|---|---|---|---|
| 1 node (1x2, 1x4) | 96 performance, 24 checkpoint | 1.5 h | 2:30 |
| 2 nodes (2x1, 2x4) | 66 performance, 24 checkpoint, 16 stress | 2.5 h | 3:30 |
| 4 nodes (4x4) | 48 performance, 24 checkpoint, 4 stress | 1.4 h | 2:30 |

About 12 node-hours in total.

## 11. Decisions (2026-10-09)

- netsplit only, the round-2 (multi-clock) build; no other checkpointing
  design is measured.
- Baseline: NCCL with GDR off. Production NCCL (GDR on) only for `latency`
  and `bw` across nodes, reported as a limitation.
- Checkpoint delays 5 and 10 s for every checkpoint run.
- Default GPU nodes, with the hardware recorded per job.
- The point-to-point catch-up and lifecycle scenarios
  (`tests/multi/p2p_pending.cu`) are not part of the campaign.
- Repeated checkpoints and restarts are part of the campaign.
- No GPUDirect RDMA mode, no restarts on other nodes, no 8x4: the series ends
  at 4x4.
- Every phase of every run has a timeout (§7).

## 12. Known risks

- **CUDA restore error 304** (`cuCheckpointProcessRestore`, in DMTCP's CUDA
  plugin): 13 of 174 netsplit restarts in round 1 and earlier, concentrated
  in IS class D at 4x4 (5 of 6 restarts, 3–10 of 16 ranks). No failing rank
  was ever on the job's first node, where the coordinator runs. Failures are
  kept in the success rate, not hidden by retries.
- **IS class D at 2x4** failed in the rebuild after the GPU restore
  (`unspecified launch failure` in `ncclCommInitRank`) with a 10 s delay
  (round 1: 0 of 6; 80 GB nodes: 1 of 6), and passed 6 of 6 with 5 s. The two
  delays of §4 measure this.
- **IS class D images** are 30–47 GiB per checkpoint, written to Lustre;
  write time dominates the checkpoint and varies with load. Report the
  spread across repetitions.
- **A startup hang** (1 of 240 runs in round 1, allgather 4x4, no output in
  600 s). The startup limit records it.
- **Same-node restarts may read the images from the Lustre client cache**,
  since the nodes just wrote them; the restart latency is a lower bound for a
  restart on fresh nodes.
