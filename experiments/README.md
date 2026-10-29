# Experiments

The evaluation of transparent NCCL checkpointing (DMTCP + `libncclplugin.so` +
DMTCP's CUDA plugin) on Perlmutter, for the paper. Four experiments, one
harness, one command to turn the results into tables:

```bash
experiments/build.sh                 # once, on a login node
experiments/submit.sh                # all experiments: 3 jobs (1, 2 and 4 nodes)
experiments/collect.sh               # any time; tables land in experiments/results/<campaign>/
```

## What is evaluated

The system checkpoints an unmodified NCCL program. At a checkpoint the plugin
runs a **multi-clock cut**: every rank publishes, per communicator, how many
collectives it has issued (a collective clock per communicator), and per peer,
how many sends and receives it has posted (a point-to-point clock per pair of
ranks). NCCL has no wildcard receive and a pair's operations match in posting
order, so these counters say exactly which operations still lack a
counterpart. A rank behind its peers is let run until it catches up; then
every communicator is drained, finalized and destroyed, the CUDA plugin saves
GPU state, and DMTCP writes one image per process. On resume or restart the
communicators are rebuilt. Communicator creation, split and destruction are
part of the cut. (Commits 57477b5, c9274f3, 5d6f2c2, 7d721ca.)

| Question | Experiment |
|---|---|
| How much does running under the system slow NCCL down, operation by operation? | **E1 micro** |
| Do NCCL operations still produce correct results across a checkpoint? | E1 micro (ckpt modes), E4 |
| What does a checkpoint and a restart of a whole application cost, and how does it scale from 1 to 4 full nodes? | **E2 scaling** |
| How does that compare with what applications do today, `torch.save` of the model weights? | **E3 torchsave** |
| Is the cut correct when a checkpoint arrives while a send or receive is still unmatched? | **E4 p2p** |

## Setup

- **Hardware.** Perlmutter GPU nodes: 4x NVIDIA A100 per node (NVLink within
  a node), Slingshot-11 between nodes, `$SCRATCH` (Lustre, all-flash) for
  every checkpoint image and every `torch.save` file. Each job records its
  nodes, GPUs, drivers, NCCL, CUDA, PyTorch and the repo/DMTCP commits in
  `raw/sysinfo/` and `raw/campaign.txt`.
- **Software.** `cudatoolkit/13.2`, `nccl/2.29.2-cu13`, `pytorch/2.13.0`;
  DMTCP is the `dmtcp` submodule (4.2-dev); Darshan is unloaded (its
  `LD_PRELOAD` wrappers conflict with DMTCP's).
- **No MPI.** Every rank is one `srun` task bound to one GPU; ranks exchange
  the `ncclUniqueId` through a file on `$SCRATCH` (PyTorch: a `FileStore`).
- **Layouts** (`NODES x GPUs per node`):

| Layout | GPUs | Used by |
|---|---|---|
| 1x2 | 2 | E1 (intra-node), E4 |
| 2x1 | 2 | E1 (inter-node), E4 |
| 1x4 | 4 | E2, E3, E4 |
| 2x4 | 8 | E2, E3, E4 (2x2) |
| 4x4 | 16 | E2, E3 |

- **NCCL settings** (`NCCL_PROFILE=conservative`, `NET_PROFILE=socket`),
  identical in every mode except `native_ref`: P2P/SHM/NVLS/cuMem/DMABUF off,
  GPUDirect RDMA off, NCCL's socket transport between nodes, NCCL RAS off.
  These are the settings the plugin checkpoints reliably with; `native_ref`
  (NCCL's defaults: NVLink P2P, AWS-OFI on CXI, GDR) shows what they cost.

## Modes

| Mode | What runs |
|---|---|
| `native` | the program under `srun`, no DMTCP: **the baseline** |
| `native_ref` | as `native`, with NCCL's production defaults (E1 only) |
| `dmtcp` | under `dmtcp_launch` with the NCCL and CUDA plugins, no checkpoint |
| `ckpt_resume` | `dmtcp`, one checkpoint `CKPT_DELAY` (10) s after the program's first progress report, then the same processes continue to the end |
| `ckpt_restart` | `dmtcp`, one checkpoint, every process killed, `dmtcp_restart` on the same nodes and GPUs, run to the end |

Run lengths are targets in seconds, not iteration counts: each program times
its first iterations, agrees on the slowest rank, and sizes the rest of the
run to fit (`*_TARGET_S` = 20 s for timing runs, `*_CKPT_TARGET_S` = 45 s for
checkpoint runs), so a run takes about as long at 2 GPUs as at 16. Each
configuration is repeated `REPS` = 3 times, the modes interleaved within a
repetition; tables report mean ± sample standard deviation.

## E1 micro: NCCL microbenchmarks (1x2, 2x1)

`microbench/nccl_bench.cu`, OSU-style (warm-up, host timing around
stream-synchronized operations, every size validated once, untimed):

| Benchmark | Measures |
|---|---|
| `latency` | ping-pong `ncclSend`/`ncclRecv`, half round trip (µs) |
| `bw`, `bibw` | windowed (64 messages) uni-/bidirectional bandwidth (MB/s) |
| `allreduce`, `allgather`, `reduce_scatter`, `bcast`, `reduce`, `alltoall` | per-operation latency, avg over ranks (µs) |

Message sizes 4 B to 64 MiB, 0.5 s per size. Modes `native`, `native_ref`,
`dmtcp`; `bw`, `allreduce` and `alltoall` also run through a checkpoint
(`ckpt_resume`, `ckpt_restart`, at 4 MiB for 45 s) and must still validate.
**Tables:** `micro_sizes.csv` (per size: native, dmtcp, overhead,
native_ref), `overhead.csv` (largest size), `ckpt.csv`.

## E2 scaling: full transparent checkpoint (1x4, 2x4, 4x4)

Every GPU of 1, 2 and 4 nodes:

| Benchmark | Communication | Verified by | Metric |
|---|---|---|---|
| NAS IS, class C | `ncclAllReduce`, all-to-all send/recv, `ncclReduce` | NPB `Verification = SUCCESSFUL` | Mop/s |
| NAS CG, class C | send/recv exchanges (`ncclGroupStart/End`), reductions | NPB verification (zeta) | Mop/s |
| NAS MG, class C | halo exchange send/recv, `ncclAllReduce` | NPB verification (L2 norm) | Mop/s |
| PyTorch DDP (`tests/multi/pytorch_minimal_test.py`) | `ProcessGroupNCCL`, bucketed allreduce | `PASS` on every rank + an exact `all_reduce` every step | steps/s, slowest rank |

PyTorch: 2 x 4096^2 fp32 parameters (134 MB), batch 64, SGD with momentum.
Modes `native`, `dmtcp`, `ckpt_resume`, `ckpt_restart`. A run counts only if
the application verifies **after** the checkpoint (resume) or restart.

Per checkpoint (`ckpt.csv`, raw per checkpoint in `raw/ckpts.csv`):

| Column | Meaning |
|---|---|
| `ckpt_s` | `dmtcp_command -c` until the coordinator has written the restart script, i.e. every image is on disk: the time the job is stopped |
| `nccl_drain_s` | the NCCL plugin's share of it (consensus rounds, drain, finalize/destroy), max over ranks |
| `rounds_max` | consensus rounds of the multi-clock cut |
| `image_bytes` | all images of the checkpoint (uncompressed; `--no-gzip`) |
| `resume_rebuild_s` | communicator re-creation after a checkpoint the job survives, max over ranks |
| `restart_latency_s` | `dmtcp_restart` until every rank has rebuilt its communicators |
| `restart_rebuild_s` | the rebuild's share of it |
| `restart_total_s` | `dmtcp_restart` until the job has completed |

**Tables:** `overhead.csv` (runtime overhead), `ckpt.csv`; LaTeX
`tex/overhead.tex`, `tex/ckpt.tex`.

## E3 torchsave: torch.save as the baseline (1x4, 2x4, 4x4)

Applications checkpoint today by saving their model (and optimizer) with
`torch.save`. `torchsave/torch_save_bench.py` measures exactly that for the
model of E2, on the same nodes, GPUs and filesystem, at widths
`TS_WIDTHS` = 4096, 8192, 16384 (134 MB, 537 MB, 2.1 GB of parameters; with
SGD momentum the saved state is twice that). Per save, the slowest rank's:

| Column | Meaning |
|---|---|
| `d2h_s` | copy of the state dict (model + optimizer) from GPU to host |
| `write_s` | `torch.save` of the host copy (serialization + write) |
| `fsync_s` | `fsync`, until the data has left the page cache |
| `job_save_s` | barrier to barrier: **the time the job is stopped** |
| `job_load_s` | `torch.load` to the GPU + `load_state_dict` (page cache dropped first), barrier to barrier; the restored parameters are checked |

Two policies: `all` (every rank saves its replica: one file per GPU, as DMTCP
writes one image per process) and `rank0` (only rank 0 saves, the usual DDP
practice since replicas are identical). Each width is also run as a
checkpointed PyTorch job (`pytorch_w8192`, `pytorch_w16384`; width 4096 is
E2's `pytorch`), so every `torch.save` point has a DMTCP twin.
**Tables:** `torchsave.csv`, and `vs_torchsave.csv` / `tex/vs_torchsave.tex`:
per layout and width, `torch.save` vs. DMTCP checkpoint time, `torch.load` vs.
restart latency, and saved payload vs. image size.

How to read the comparison: `torch.save` writes only the tensors the
programmer chose and needs code in the application (and a restart path that
rebuilds everything else: process group, data loader position, RNG state).
The transparent checkpoint needs no code and saves the whole process (CUDA
context, Python heap, NCCL, optimizer, everything in host memory), so its
images are larger; `image / payload` and `ckpt / save` quantify that price.
`torch.load` is timed in a running process, so it excludes interpreter
start-up, `import torch` and process-group creation, all of which a real
restart pays and `restart_latency_s` includes.

## E4 p2p: checkpoints during unmatched point-to-point operations

`tests/multi/p2p_pending.cu`: one side of a transfer waits 20 s before posting
its operation, and the checkpoint lands 1 s into that window, so the other
side's send or receive is pending. Scenarios: `recv-first`, `recv-first-cold`
(a pair's first contact), `send-first`, `chain` and `relay` (3 ranks),
`multi` (two communicators), `create`/`split`/`destroy` (communicator
lifecycle inside the window), at 1x2–1x4 and across nodes (2x1, 2x2), each
with resume and restart. **PASS** = every receiver got its sender's real data
and no `[NCCL Drain]` line appeared (the checkpoint waited for the missing
operation instead of completing the other side with a dummy message).
**Tables:** `p2p.csv`, `tex/p2p.tex`.

## Running

```bash
experiments/build.sh                    # DMTCP + CUDA plugin, NCCL plugin, NAS IS/CG/MG
                                        # (classes B, C), nccl_bench, p2p_pending; then checks
SMOKE=1 experiments/submit.sh micro     # optional: a 30-min debug job, short runs, 1 rep
DRY_RUN=1 experiments/submit.sh         # print every run of the plan, submit nothing
experiments/submit.sh                   # everything: one job per node count (1, 2, 4)
experiments/submit.sh scaling torchsave # or a subset: micro scaling torchsave p2p
```

Interactive (an allocation you already hold; the driver runs as a one-node
step and uses every node of the allocation it needs):

```bash
salloc -A m4128 -q interactive -C gpu -N 2 --gpus-per-node=4 -t 02:00:00
module load cudatoolkit nccl && module unload darshan
experiments/submit.sh --here micro p2p
```

All jobs of one `submit.sh` share a campaign (default: today's date); its raw
data accumulate in `$SCRATCH/nccl-ckpt-paper/<campaign>/` (`results.csv`, one
row per run; `ckpts.csv`; `torchsave.csv`; `runs/<bench>/<layout>/<mode>/rep<N>/`
with every log, `meta.env` and `result.env`). Each job skips runs that could
not finish in its remaining time and lists them in `skipped.txt`.

## Getting the results

```bash
experiments/collect.sh                  # newest campaign
experiments/collect.sh 20261030         # a given one
NAME=final experiments/collect.sh 20261030 20261102   # merge a rerun into one set
```

writes `experiments/results/<name>/`:

| File | Content | Paper |
|---|---|---|
| `summary.md` | every table below, readable | start here |
| `overhead.csv`, `tex/overhead.tex` | native vs. dmtcp, no checkpoint (E1 largest size, E2) | runtime overhead |
| `micro_sizes.csv` | E1 per message size | latency / bandwidth plots |
| `ckpt.csv`, `tex/ckpt.tex` | checkpoint, resume, restart (E1, E2, E3) | checkpoint cost, scaling plots |
| `torchsave.csv` | E3 per width, policy, layout | torch.save breakdown |
| `vs_torchsave.csv`, `tex/vs_torchsave.tex` | torch.save vs. transparent checkpoint, same model | the comparison |
| `p2p.csv`, `tex/p2p.tex` | E4 verdicts | correctness |
| `failures.csv` | every run that did not pass, its status and first error line | |
| `raw/` | the campaign's raw CSVs, `campaign.txt`, `skipped.txt`, `sysinfo/` | artifact |

It runs on a login node with no modules, copies only small files (logs and
images stay on `$SCRATCH`), and can be run while jobs are still going. Commit
`experiments/results/<name>/` and `\input` the `.tex` tables, or plot from
the CSVs. A failed run's status names the phase that did not complete
(`hang_ckpt`, `hang_restart`, `fail_verify`, ...) and its run directory has
the logs; rerun just those with a smaller plan (e.g.
`SCALE_BENCHES=cg SCALE_LAYOUTS=4x4 CAMPAIGN=rerun1 experiments/submit.sh scaling`)
and merge with `collect.sh`.

## Knobs

Every setting is an environment variable read by `lib/common.sh` (defaults
there) or `plan.sh` (which runs make up each experiment), and is recorded per
run in `meta.env`. The common ones: `REPS`, `CKPT_DELAY`, `IS_CLASS` /
`CG_CLASS` / `MG_CLASS`, `PT_WIDTH`, `TS_WIDTHS`, `TS_REPS`, `TS_DIR` (save
`torch.save` files elsewhere, e.g. node-local), `NCCL_PROFILE`,
`NET_PROFILE`, `CKPT_STRIPE_COUNT`, `KEEP_IMAGES=1`, `WALLTIME_<nodes>`,
`ACCOUNT`, `QOS`, and the plan lists `MICRO_*`, `SCALE_*`, `TS_LAYOUTS`,
`P2P_CASES`.

## Files

| Path | Role |
|---|---|
| `build.sh` | builds every binary (into `$SCRATCH/nccl-ckpt-paper/bin`) |
| `plan.sh` | the experiments as lists of runs |
| `submit.sh` | one Slurm job per node count, or `--here` |
| `campaign.sbatch` | a job: runs its node count's part of the plan |
| `collect.sh` | results to `results/<name>/` |
| `lib/run_bench.sh` | one run: launch, checkpoint, restart, verify, measure |
| `lib/launch_rank.sh` | per-rank wrapper (native / `dmtcp_launch` / `dmtcp_restart`) |
| `lib/common.sh` | settings, NCCL profiles, time limits |
| `lib/summarize.py` | the tables |
| `lib/sysinfo.sh` | hardware/software record |
| `microbench/nccl_bench.cu` | E1 microbenchmarks |
| `torchsave/torch_save_bench.py` | E3 baseline |

The harness derives from `tests/report` on the `netsplit` branch, without
netsplit's lower half: everything here is the single-process plugin.
