# Campaign 20261113-cg (20261113-cg)

45 runs, 45 ok. Values are mean ± sample std over successful repetitions. `native`: no DMTCP, the same NCCL settings as every DMTCP run; `dmtcp`: under dmtcp_launch with the NCCL and CUDA plugins, no checkpoint; `native_ref`: NCCL module defaults (production reference). Overhead is the slowdown of `dmtcp` relative to `native`.

```
campaign=20261113-cg experiments=scaling submitted=2026-11-13T10:31:41-08:00 smoke=0
repo=2ecf8330b2ada482f8dea2a8ef0ead0999a44632 9 modified files
dmtcp=aeaa5461a0a1a4ff7b29004d2e906ae8b0590795
NCCL_PROFILE=default
NET_PROFILE=socket
REPS=3
```

## Runtime overhead, no checkpoint (E1, E2)

| Benchmark | Scale | Metric | native | dmtcp | overhead | native_ref | wall native (s) | wall dmtcp (s) |
|---|---|---|---|---|---|---|---|---|
| cg | 1x4 (4 GPUs) | cg_mops | 7497.187 ± 11.912 | 7441.350 ± 151.561 | +0.75% | 7462.350 ± 68.544 | 43.84 ± 0.05 | 43.93 ± 0.62 |
| cg | 2x4 (8 GPUs) | cg_mops | 12457.917 ± 509.015 | 12655.857 ± 107.821 | -1.56% | 13796.027 ± 277.018 | 28.61 ± 0.77 | 28.26 ± 0.43 |
| cg | 4x4 (16 GPUs) | cg_mops | 19583.887 ± 118.524 | 19132.623 ± 330.688 | +2.36% | 21819.953 ± 76.376 | 27.64 ± 0.19 | 27.92 ± 0.47 |

## Checkpoint and restart (E1, E2, E3)

`ckpt` = from `dmtcp_command -c` until the coordinator writes the restart script, i.e. every image is on disk (NCCL consensus, drain and teardown, CUDA checkpoint, image write). `nccl` = the NCCL plugin's share (max over ranks). `rounds` = most consensus rounds any checkpoint took. `resume rebuild` = communicator re-creation after a checkpoint the job survives. `restart latency` = `dmtcp_restart` until every rank has rebuilt its communicators; `restart total` = until the job completes.

| Benchmark | Scale | Mode | ok | ckpt (s) | nccl (s) | rounds | image total (GiB) | image/rank (MiB) | write (MiB/s) | resume rebuild (s) | restart latency (s) | restart rebuild (s) | restart total (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cg | 1x4 (4 GPUs) | ckpt_restart | 3/3 | 4.61 ± 0.06 | 0.871 ± 0.073 | 3 | 6.80 ± 0.00 | 1741 ± 1 | 1512 ± 20 | - | 4.19 ± 0.05 | 0.579 ± 0.040 | 32.67 ± 0.35 |
| cg | 1x4 (4 GPUs) | ckpt_resume | 3/3 | 4.78 ± 0.14 | 0.919 ± 0.208 | 3 | 6.81 ± 0.00 | 1742 ± 0 | 1459 ± 45 | 0.557 ± 0.081 | - | - | - |
| cg | 2x4 (8 GPUs) | ckpt_restart | 3/3 | 4.64 ± 0.26 | 0.875 ± 0.214 | 3 | 12.59 ± 0.00 | 1612 ± 0 | 2785 ± 161 | - | 4.51 ± 0.30 | 0.732 ± 0.138 | 18.02 ± 0.43 |
| cg | 2x4 (8 GPUs) | ckpt_resume | 3/3 | 4.72 ± 0.24 | 0.926 ± 0.177 | 2 | 12.60 ± 0.00 | 1612 ± 1 | 2736 ± 144 | 0.640 ± 0.021 | - | - | - |
| cg | 4x4 (16 GPUs) | ckpt_restart | 3/3 | 4.53 ± 0.10 | 0.873 ± 0.135 | 3 | 24.15 ± 0.00 | 1545 ± 0 | 5463 ± 114 | - | 4.13 ± 0.00 | 0.678 ± 0.064 | 45.58 ± 0.85 |
| cg | 4x4 (16 GPUs) | ckpt_resume | 3/3 | 4.70 ± 0.13 | 1.007 ± 0.132 | 3 | 24.15 ± 0.00 | 1546 ± 0 | 5267 ± 149 | 0.716 ± 0.044 | - | - | - |

