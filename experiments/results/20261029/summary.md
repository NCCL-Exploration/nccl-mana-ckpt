# Campaign 20261029 (20261029)

10 runs, 10 ok. Values are mean ± sample std over successful repetitions. `native`: no DMTCP, the same NCCL settings as every DMTCP run; `dmtcp`: under dmtcp_launch with the NCCL and CUDA plugins, no checkpoint; `native_ref`: NCCL module defaults (production reference). Overhead is the slowdown of `dmtcp` relative to `native`.

```
campaign=20261029 experiments=micro scaling torchsave p2p submitted=2026-10-29T12:26:45-07:00 smoke=0
repo=6be972a189b396b818e1ba79cd462f92d8402e9d 3 modified files
dmtcp=aeaa5461a0a1a4ff7b29004d2e906ae8b0590795
NCCL_PROFILE=default
NET_PROFILE=socket
REPS=3
```

## Runtime overhead, no checkpoint (E1, E2)

| Benchmark | Scale | Metric | native | dmtcp | overhead | native_ref | wall native (s) | wall dmtcp (s) |
|---|---|---|---|---|---|---|---|---|
| is | 1x4 (4 GPUs) | is_mops | 918.080 | - | - | - | 29.45 | - |
| is | 2x4 (8 GPUs) | is_mops | 1245.460 | - | - | - | 27.58 | - |
| is | 4x4 (16 GPUs) | is_mops | 2569.850 | - | - | - | 26.75 | - |
| nccl_latency | 1x2 (2 GPUs) | us_at_max | 884.130 | - | - | - | 15.27 | - |
| nccl_latency | 2x1 (2 GPUs) | us_at_max | 24431.850 | - | - | - | 16.00 | - |

## E1 nccl_latency, 1x2 (2 GPUs) (us)

| Size (B) | native | dmtcp | overhead | native_ref |
|---|---|---|---|---|
| 4 | 16.69 | - | - | - |
| 8 | 16.59 | - | - | - |
| 16 | 16.55 | - | - | - |
| 32 | 16.57 | - | - | - |
| 64 | 16.62 | - | - | - |
| 128 | 16.65 | - | - | - |
| 256 | 16.62 | - | - | - |
| 512 | 16.66 | - | - | - |
| 1024 | 16.70 | - | - | - |
| 2048 | 16.77 | - | - | - |
| 4096 | 17.14 | - | - | - |
| 8192 | 17.96 | - | - | - |
| 16384 | 19.21 | - | - | - |
| 32768 | 25.49 | - | - | - |
| 65536 | 27.36 | - | - | - |
| 131072 | 28.94 | - | - | - |
| 262144 | 30.32 | - | - | - |
| 524288 | 34.78 | - | - | - |
| 1048576 | 44.79 | - | - | - |
| 2097152 | 64.32 | - | - | - |
| 4194304 | 91.25 | - | - | - |
| 8388608 | 145.40 | - | - | - |
| 16777216 | 252.77 | - | - | - |
| 33554432 | 464.06 | - | - | - |
| 67108864 | 884.13 | - | - | - |

## E1 nccl_latency, 2x1 (2 GPUs) (us)

| Size (B) | native | dmtcp | overhead | native_ref |
|---|---|---|---|---|
| 4 | 48.12 | - | - | - |
| 8 | 47.86 | - | - | - |
| 16 | 47.41 | - | - | - |
| 32 | 47.31 | - | - | - |
| 64 | 47.39 | - | - | - |
| 128 | 47.45 | - | - | - |
| 256 | 51.36 | - | - | - |
| 512 | 51.60 | - | - | - |
| 1024 | 52.12 | - | - | - |
| 2048 | 52.68 | - | - | - |
| 4096 | 53.93 | - | - | - |
| 8192 | 57.76 | - | - | - |
| 16384 | 65.28 | - | - | - |
| 32768 | 80.32 | - | - | - |
| 65536 | 109.11 | - | - | - |
| 131072 | 156.06 | - | - | - |
| 262144 | 266.14 | - | - | - |
| 524288 | 447.94 | - | - | - |
| 1048576 | 747.11 | - | - | - |
| 2097152 | 1135.35 | - | - | - |
| 4194304 | 1895.41 | - | - | - |
| 8388608 | 3422.09 | - | - | - |
| 16777216 | 6377.23 | - | - | - |
| 33554432 | 12459.03 | - | - | - |
| 67108864 | 24431.85 | - | - | - |

## Checkpoint and restart (E1, E2, E3)

`ckpt` = from `dmtcp_command -c` until the coordinator writes the restart script, i.e. every image is on disk (NCCL consensus, drain and teardown, CUDA checkpoint, image write). `nccl` = the NCCL plugin's share (max over ranks). `rounds` = most consensus rounds any checkpoint took. `resume rebuild` = communicator re-creation after a checkpoint the job survives. `restart latency` = `dmtcp_restart` until every rank has rebuilt its communicators; `restart total` = until the job completes.

| Benchmark | Scale | Mode | ok | ckpt (s) | nccl (s) | rounds | image total (GiB) | image/rank (MiB) | write (MiB/s) | resume rebuild (s) | restart latency (s) | restart rebuild (s) | restart total (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|

## torch.save baseline (E3)

Per save: `d2h` = copy of the state dict (model + SGD momentum) to host memory, `write` = torch.save of that copy, `fsync` = until it is on the filesystem; `job save` = barrier to barrier, the time the job is stopped. `job load` = torch.load to the GPU + load_state_dict, page cache dropped first. Each phase is the max over ranks; mean ± std over every save of every run.

| Scale | Width | Params | Policy | n | payload/rank (MiB) | d2h (s) | write (s) | fsync (s) | job save (s) | job load (s) | MiB/s per rank |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1x4 (4 GPUs) | 4096 | 33,603,594 | all | 5 | 256 ± 0 | 0.045 ± 0.000 | 0.360 ± 0.003 | 0.006 ± 0.000 | 0.410 ± 0.003 | 0.300 ± 0.002 | 628 ± 4 |
| 1x4 (4 GPUs) | 4096 | 33,603,594 | rank0 | 5 | 256 ± 0 | 0.045 ± 0.001 | 0.337 ± 0.003 | 0.004 ± 0.000 | 0.387 ± 0.004 | 0.285 ± 0.002 | 665 ± 6 |
| 1x4 (4 GPUs) | 8192 | 134,316,042 | all | 5 | 1025 ± 0 | 0.175 ± 0.002 | 1.452 ± 0.011 | 0.005 ± 0.001 | 1.627 ± 0.007 | 1.350 ± 0.013 | 631 ± 3 |
| 1x4 (4 GPUs) | 8192 | 134,316,042 | rank0 | 5 | 1025 ± 0 | 0.161 ± 0.001 | 1.349 ± 0.011 | 0.018 ± 0.013 | 1.532 ± 0.020 | 1.120 ± 0.004 | 671 ± 9 |
| 1x4 (4 GPUs) | 16384 | 537,067,530 | all | 5 | 4098 ± 0 | 0.687 ± 0.012 | 5.823 ± 0.022 | 0.004 ± 0.000 | 6.507 ± 0.029 | 5.456 ± 0.092 | 631 ± 3 |
| 1x4 (4 GPUs) | 16384 | 537,067,530 | rank0 | 5 | 4098 ± 0 | 0.638 ± 0.002 | 5.349 ± 0.047 | 0.004 ± 0.002 | 6.006 ± 0.050 | 4.617 ± 0.017 | 684 ± 6 |
| 2x4 (8 GPUs) | 4096 | 33,603,594 | all | 5 | 256 ± 0 | 0.048 ± 0.002 | 0.363 ± 0.005 | 0.005 ± 0.001 | 0.414 ± 0.006 | 0.303 ± 0.002 | 622 ± 9 |
| 2x4 (8 GPUs) | 4096 | 33,603,594 | rank0 | 5 | 256 ± 0 | 0.045 ± 0.000 | 0.338 ± 0.001 | 0.006 ± 0.001 | 0.391 ± 0.002 | 0.290 ± 0.002 | 660 ± 3 |
| 2x4 (8 GPUs) | 8192 | 134,316,042 | all | 5 | 1025 ± 0 | 0.182 ± 0.001 | 1.448 ± 0.007 | 0.008 ± 0.002 | 1.633 ± 0.009 | 1.355 ± 0.009 | 629 ± 3 |
| 2x4 (8 GPUs) | 8192 | 134,316,042 | rank0 | 5 | 1025 ± 0 | 0.167 ± 0.004 | 1.343 ± 0.008 | 0.018 ± 0.009 | 1.531 ± 0.013 | 1.118 ± 0.009 | 671 ± 6 |
| 2x4 (8 GPUs) | 16384 | 537,067,530 | all | 5 | 4098 ± 0 | 0.703 ± 0.025 | 5.848 ± 0.032 | 0.005 ± 0.001 | 6.520 ± 0.023 | 5.545 ± 0.131 | 630 ± 2 |
| 2x4 (8 GPUs) | 16384 | 537,067,530 | rank0 | 5 | 4098 ± 0 | 0.644 ± 0.002 | 5.372 ± 0.043 | 0.003 ± 0.000 | 6.034 ± 0.045 | 4.608 ± 0.009 | 681 ± 5 |
| 4x4 (16 GPUs) | 4096 | 33,603,594 | all | 5 | 256 ± 0 | 0.049 ± 0.001 | 0.383 ± 0.055 | 0.012 ± 0.013 | 0.440 ± 0.054 | 0.308 ± 0.002 | 591 ± 63 |
| 4x4 (16 GPUs) | 4096 | 33,603,594 | rank0 | 5 | 256 ± 0 | 0.044 ± 0.001 | 0.336 ± 0.003 | 0.012 ± 0.014 | 0.394 ± 0.016 | 0.288 ± 0.002 | 655 ± 25 |
| 4x4 (16 GPUs) | 8192 | 134,316,042 | all | 5 | 1025 ± 0 | 0.184 ± 0.002 | 1.459 ± 0.007 | 0.012 ± 0.006 | 1.642 ± 0.012 | 1.380 ± 0.037 | 626 ± 5 |
| 4x4 (16 GPUs) | 8192 | 134,316,042 | rank0 | 5 | 1025 ± 0 | 0.164 ± 0.002 | 1.329 ± 0.008 | 0.009 ± 0.003 | 1.508 ± 0.009 | 1.116 ± 0.005 | 682 ± 4 |
| 4x4 (16 GPUs) | 16384 | 537,067,530 | all | 5 | 4098 ± 0 | 0.723 ± 0.014 | 5.823 ± 0.010 | 0.012 ± 0.004 | 6.497 ± 0.021 | 5.515 ± 0.009 | 632 ± 2 |
| 4x4 (16 GPUs) | 16384 | 537,067,530 | rank0 | 5 | 4098 ± 0 | 0.649 ± 0.005 | 5.319 ± 0.032 | 0.006 ± 0.001 | 5.989 ± 0.036 | 4.611 ± 0.006 | 686 ± 4 |

## Checkpoints with pending point-to-point operations (E4)

PASS = the run verified (every receiver got its sender's data, before and after the checkpoint) and no `[NCCL Drain]` line appeared, i.e. the checkpoint waited for the missing counterpart instead of completing the other side with a dummy message.

| Scenario | Scale | Mode | Verdict | Status | ckpt (s) | rounds | drained |
|---|---|---|---|---|---|---|---|
| recv_first | 1x2 (2 GPUs) | ckpt_resume | PASS | ok | 21.848 | 27 | 0 |
| recv_first | 2x1 (2 GPUs) | ckpt_resume | PASS | ok | 21.063 | 27 | 0 |

