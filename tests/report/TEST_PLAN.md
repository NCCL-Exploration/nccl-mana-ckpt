# Test plan: NCCL checkpoint plugin, tech-report evaluation

All scripts for this plan are in this directory. Everything runs on NERSC Perlmutter GPU nodes. Run directories, checkpoint images and results all go to `$SCRATCH` (Lustre).

**No MPI anywhere.** The plugin supports NCCL only, so every process is a plain `srun` task, with one task per GPU. Each program takes its rank from `SLURM_PROCID`, `SLURM_NTASKS` and `SLURM_LOCALID`, and exchanges its `ncclUniqueId` through a file on `$SCRATCH`.

## 1. Questions the evaluation answers

1. **Correctness.** Can a job that uses NCCL point-to-point, NCCL collectives, or PyTorch DDP be checkpointed in the middle of a run, then resumed or restarted, and still finish with a verified result?
2. **Runtime overhead.** How much slower is a job under `dmtcp_launch` with the NCCL and CUDA plugins when no checkpoint is taken?
3. **Checkpoint cost.** How long does a checkpoint take, and how is that time split between the NCCL drain, the CUDA checkpoint and the image write? How large are the images?
4. **Restart cost.** How long does it take from `dmtcp_restart` until every communicator has been rebuilt, and until the job completes?
5. **Scaling.** How do items 2–4 change as the job grows from 2 GPUs to 32 GPUs, both within a node and across nodes?

## 2. Testbed (recorded automatically)

Every job runs `sysinfo.sh`, which writes `<campaign>/sysinfo/job<ID>/`:

| Item | Source | File |
|---|---|---|
| CPU model, sockets, cores, threads, NUMA, memory | `lscpu`, `/proc/meminfo` on each compute node | `nodes.txt` |
| GPU model, memory, driver, PCI IDs, max SM clock | `nvidia-smi --query-gpu` | `nodes.txt` |
| GPU interconnect topology (NVLink, NICs) | `nvidia-smi topo -m` | `nodes.txt` |
| NICs (Slingshot `hsn*`, link speed, CXI domains) | `/sys/class/net`, `fi_info -p cxi` | `nodes.txt` |
| OS, kernel | `/etc/os-release`, `uname` | `nodes.txt` |
| CUDA toolkit, driver CUDA version | `nvcc --version`, `nvidia-smi -q` | `software.txt`, `nodes.txt` |
| NCCL version and net plugin | `nccl.h` of `$NCCL_HOME`, `plugin/lib` | `software.txt` |
| libfabric, GCC, `nccl_bench` checksum and linked libraries (proves no MPI) | `fi_info --version`, `ldd` | `software.txt` |
| Python / PyTorch / PyTorch's NCCL | `torch.__version__`, `torch.cuda.nccl.version()` | `software.txt` |
| DMTCP version, plugin commit, dirty flag, `.so` checksum | git, `md5sum` | `software.txt` |
| Filesystem and Lustre stripe of the results directory | `df -hT`, `lfs getstripe` | `software.txt` |
| All experiment knobs | `common.sh` variables | `software.txt`, and `meta.env` in each run |

Expected configuration on Perlmutter: 1× AMD EPYC 7763 and 4× NVIDIA A100 per node, NVLink 3 between the GPUs, and 4× Slingshot-11 NICs. The modules are `cudatoolkit/13.2`, `nccl/2.29.2-cu13` and `pytorch/2.13.0` (which uses the same NCCL module). Quote the recorded files in the report, not this list.

## 3. Workloads

| Benchmark | What it exercises | Correctness check | Application metric |
|---|---|---|---|
| `nccl_bench -b latency\|bw\|bibw` | NCCL point-to-point (`ncclSend`/`ncclRecv`) between 2 ranks, intra-node (1×2) and inter-node (2×1). `bw`/`bibw` post a group of 64 messages, which leaves many operations in flight at the checkpoint. | Each message size validated once (untimed); `validation PASSED` | Latency (µs) or bandwidth (MB/s) per message size, 4 B – 64 MiB |
| `nccl_bench -b allreduce\|allgather\|bcast\|reduce\|reduce_scatter\|alltoall` | NCCL collectives; `alltoall` is a grouped send/recv | Same, with exact fp32 expected values | Avg, min and max latency (µs) per message size |
| NAS IS (`tests/nas-benchmarks/IS`, class C; D at scale) | Integer sort: `ncclAllReduce`, all-to-all P2P exchange, `ncclReduce` | NPB `Verification = SUCCESSFUL` | NPB Mop/s total (the time is set by the duration target) |
| PyTorch DDP (`tests/multi/pytorch_minimal_test.py`) | `ProcessGroupNCCL`: group-wrapped communicator creation, bucketed allreduce | `PASS pytorch_minimal` on every rank, plus an exact `all_reduce` check every step | Steps/s of the slowest rank (width 4096, about 134 MB of gradients per step) |

### `nccl_bench` (in `nccl_bench/`)

`nccl_bench` is our own MPI-free replacement for OSU's `osu_xccl_*` benchmarks, which need MPI for bootstrap and timing. It follows the OSU 7.x methodology:
- Warm-up iterations, then host wall-clock timing around stream-synchronized operations. With `-T SECONDS` (used by the campaign), the warm-up is timed, every rank agrees on the slowest per-iteration time, and the timed iterations are chosen to fill the budget. Without it, the defaults are 100 warm-up and 1000 timed iterations, or 10 and 100 above 1 MiB.
- Ping-pong latency is half the round-trip time.
- Bandwidth uses a window of 64 messages followed by an acknowledgement.
- Collectives are timed one iteration at a time, with an untimed barrier between iterations. Results are reported as avg, min and max over ranks.

Barriers and the combining of timings across ranks are small `ncclAllReduce` calls on the same communicator, made outside the timed region. The uniqueId file carries the Slurm job and step ID, so a stale file can never be used. Run `nccl_bench -h` for the options; `-p N` prints a progress marker, which the checkpoint runs use to know the benchmark is running.

### Run length: targets in seconds, not iteration counts

A fixed iteration count that suits 2 GPUs finishes in seconds on 32 GPUs, while large `alltoall` messages at scale can take far longer. So each benchmark sizes itself to a target duration at run time:
- **IS:** `IS_TARGET_SECONDS` times the first 10-iteration NPB pass and agrees on it across ranks (allreduce max). It then repeats the pass R times (`IS_REPEAT`). Every pass ranks identical data, so NPB verification stays valid, with the expected count and Mop/s scaled by R.
- **`nccl_bench`:** `-T` sets a budget per message size.
- **PyTorch:** `--duration` times the first steps, takes the max across ranks with `all_reduce`, and picks the step count. Steps/s excludes the warm-up steps.

Because every rank computes the same count from the agreed time, the collectives stay matched.

| Knob (`common.sh`) | Default | Smoke test |
|---|---|---|
| `IS_TARGET_S`, `PT_TARGET_S`: timing runs | 20 s | 10 s |
| `NB_SIZE_BUDGET_S`: `nccl_bench` timing runs, per size (25 sizes) | 0.5 s | 0.2 s |
| `IS_CKPT_TARGET_S`, `PT_CKPT_TARGET_S`, `NB_CKPT_TARGET_S`: checkpoint runs | 45 s | 25 s |
| `CKPT_DELAY`: checkpoint taken this long after the first progress | 10 s | 5 s |

Use class C for IS up to a few nodes, and class D (16× larger, at least 4 ranks) for the multi-node points (`IS_CLASS=D`).

## 4. Configurations

**Modes.** Each benchmark runs in each mode, implemented in `run_bench.sh`:

| Mode | DMTCP | NCCL settings | Purpose |
|---|---|---|---|
| `native` | no | checkpoint-safe profile (below) | Baseline for overhead |
| `dmtcp` | `dmtcp_launch` with the NCCL and CUDA plugins, no checkpoint | same | Runtime overhead = `dmtcp` vs `native` |
| `native_ref` | no | module defaults (AWS-OFI/CXI, P2P, SHM, NVLS) | Production reference; shows what the safe NCCL settings cost |
| `ckpt_resume` | yes, one blocking checkpoint mid-run, then the job continues | same as `native` | Checkpoint latency, image size, rebuild after resume, correctness |
| `ckpt_restart` | yes, one checkpoint, kill, `dmtcp_restart`, run to completion | same as `native` | Restart latency, correctness after restore |

**NCCL profile.** `native`, `dmtcp` and both checkpoint modes always use the same NCCL environment, so the overhead numbers isolate DMTCP and the plugin.
- `NCCL_PROFILE=conservative` (default) disables P2P, SHM, cuMem, DMABUF, NVLS and GDR. This is the set that `run_perlmutter_ckpt.sh` is known to checkpoint with.
- `NET_PROFILE=socket` (default) uses NCCL's built-in TCP transport over `hsn*`, with `NCCL_NET_PLUGIN=none`. The AWS-OFI plugin holds `/dev/cxi` resources across communicator destruction, and DMTCP cannot checkpoint those.
- If the smoke test (§6) passes with `NCCL_PROFILE=default` and/or `NET_PROFILE=ofi`, rerun with those settings and report both. That is the stronger result.

**Scales.** `submit_all.sh` submits one job per node count:

| Nodes × GPUs/node | Ranks | Notes |
|---|---|---|
| 1×2 | 2 | intra-node; point-to-point benchmarks |
| 1×4 | 4 | full node |
| 2×1 | 2 | inter-node point-to-point |
| 2×4 | 8 | |
| 4×4 | 16 | |
| 8×4 | 32 | optional (`NODES_LIST="1 2 4 8"` includes it) |

IS requires a power-of-two rank count. The point-to-point benchmarks run only at 2 ranks. One rank is mapped to each GPU, using `LOCAL_RANK = SLURM_LOCALID`.

**Repetitions.** `REPS=3` by default. Modes are interleaved within each repetition, so drift over the course of a job affects all modes equally. Use 5 repetitions for the final numbers if the budget allows.

## 5. Metrics and how they are measured

| Metric | Definition | Where it comes from |
|---|---|---|
| Runtime overhead (app) | `dmtcp` vs `native` on the application's own metric; positive = slower. For `nccl_bench` this is also given per message size (`sizes.csv`). | `summarize.py` |
| Runtime overhead (wall) | Same, on the wall time of the whole `srun` step, which includes DMTCP launch and the PyTorch import | `run_bench.sh` timers |
| Checkpoint time | From `dmtcp_command -c` until the coordinator writes the restart script, i.e. every image is on disk. Covers drain, teardown, CUDA checkpoint and image write. (`-bc` cannot be used: in DMTCP 4.2 it does not block.) | `ckpt_s` |
| NCCL drain time | `[NCCL PLUGIN] Checkpoint overhead`, max over ranks: epoch vote, sequence consensus, P2P drain, stream sync, communicator teardown | `nccl_drain_s` |
| CUDA + write time | `ckpt_s − nccl_drain_s` (derived) | report |
| Image size | Sum of `ckpt_*.dmtcp` (uncompressed, `--no-gzip`), total and per rank | `image_bytes` |
| Effective write bandwidth | image bytes / `ckpt_s` (aggregate to Lustre) | `write_MBps` |
| Resume rebuild | `[NCCL PLUGIN] Rebuild overhead` after a checkpoint the job survives, max over ranks | `resume_rebuild_s` |
| Restart latency | From `dmtcp_restart` (via `srun`) until every rank has printed `Rebuild overhead`, i.e. memory, CUDA and communicators restored | `restart_latency_s` |
| Restart rebuild | NCCL communicator re-creation share of the restart | `restart_rebuild_s` |
| Restart total | From `dmtcp_restart` until the job completes | `restart_total_s` |
| Success rate | Number of `ok` runs / runs, per configuration. Failures are kept and listed. | `status` |

The plugin's timing output comes from `src/libncclplugin.c`. The `Checkpoint overhead` line already existed; this plan adds the `Rebuild overhead` line printed by `rebuild_all_comms()`.

## 6. Procedure

```bash
# 0. Once, on a login node: build DMTCP, the plugin, IS (classes B and C) and
#    nccl_bench into $SCRATCH/nccl-ckpt-report/bin.
tests/report/build.sh

# 1. Smoke test: 1 node, 2 GPUs, 1 repetition, short runs, debug QOS (30 min).
#    Confirms every benchmark x mode works before spending the budget.
SMOKE=1 tests/report/submit_all.sh
python3 tests/report/summarize.py $SCRATCH/nccl-ckpt-report/smoke-<stamp>

# 1b. Optional: also try the faster NCCL settings.
SMOKE=1 NCCL_PROFILE=default NET_PROFILE=ofi CAMPAIGN=smoke-fast tests/report/submit_all.sh

# 2. Full campaign: one job per node count, all rows in one results.csv.
CAMPAIGN=report-v1 tests/report/submit_all.sh

# 3. Tables and CSVs for the report.
python3 tests/report/summarize.py $SCRATCH/nccl-ckpt-report/report-v1
```

To run one measurement by hand inside `salloc -C gpu -N 1 --gpus-per-node=4 -q interactive -A m4128`:

```bash
tests/report/run_bench.sh is ckpt_restart 1 4 1
```

Each run directory, `<campaign>/runs/<bench>/n<N>x<G>/<mode>/rep<R>/`, contains:
- `meta.env` (configuration)
- `out.log`, and `restart.log` for restart runs
- `ckpt_cmd.log`
- `coord.log`
- `sizes.csv` (`nccl_bench` runs only)
- `result.env`

Images are deleted after their size is recorded. Set `KEEP_IMAGES=1` to keep them.

**Budget.** With the default targets, one scale takes about 85 minutes for 3 repetitions: 11 benchmarks × 3 timing modes plus 5 × 2 checkpoint modes. Each 1- and 2-node job covers two scales and gets 4 hours; the 4- and 8-node jobs get 2 hours. That's at most about 40 node-hours for `NODES_LIST="1 2 4 8"`. The smoke test costs 0.5 node-hours.

## 7. Threats to validity and known risks

- **`nccl_bench` instead of OSU.** OSU's `osu_xccl_*` benchmarks bootstrap and time through MPI, which the plugin does not support. `nccl_bench` reproduces OSU's methodology but is a different implementation. The report should say so, and should not compare its numbers directly with published OSU results.
- **NCCL transport.** The checkpoint-safe profile is slower than production NCCL. `native_ref` quantifies this so the report can state it honestly, rather than presenting overhead against a handicapped baseline.
- **IS repeats its NPB iterations** (`IS_REPEAT`, chosen by `IS_TARGET_SECONDS`) to reach the target duration. The report should give the pass count (printed by IS) alongside the time. Only IS's timed section is used as its application metric.
- **PyTorch:**
  - `import torch` under DMTCP requires the `pyshim` workaround, `USE_LIBUV=0` and `--no-gzip`; see `tests/multi/pytorch_dmtcp_env.sh`. These are applied in every mode, native included.
  - TCPStore watchdog warnings after a restart are expected (see `tests/multi/Makefile`).
  - Steps/s in the checkpoint modes includes the checkpoint pause, so it is not used as an overhead number.
- **Lustre variance.** Write bandwidth depends on load from other users of `$SCRATCH`. Report the spread across repetitions as well as the mean. Optionally sweep `CKPT_STRIPE_COUNT` (for example 1 vs 8) for the large PyTorch images.
- **Warm-up.** The first DMTCP launch on a node pays the page-cache cost for the `.so` files. Interleaving modes spreads that cost across all modes, and the per-repetition spread shows how large it is.

## 8. Mapping to report sections

| Report section | Source |
|---|---|
| Experimental setup | `sysinfo/*/nodes.txt`, `software.txt`; §3–§4 of this plan |
| Correctness | Success column of the checkpoint table, plus the failed-runs table |
| Runtime overhead | "Runtime overhead" table; `sizes.csv` (overhead vs message size) |
| Checkpoint cost breakdown | `agg_ckpt.csv`: drain, CUDA and write, image size |
| Restart cost | `agg_ckpt.csv`, `ckpt_restart` rows |
| Scaling | `agg_ckpt.csv` / `agg_overhead.csv` against `nranks` |
