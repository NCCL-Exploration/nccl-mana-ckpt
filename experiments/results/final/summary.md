# Campaign final (20261110-nocg 20261113-cg)

449 runs, 448 ok. Values are mean ± sample std over successful repetitions. `native`: no DMTCP, the same NCCL settings as every DMTCP run; `dmtcp`: under dmtcp_launch with the NCCL and CUDA plugins, no checkpoint; `native_ref`: NCCL module defaults (production reference). Overhead is the slowdown of `dmtcp` relative to `native`.

```
campaign=20261110 experiments=micro scaling torchsave p2p submitted=2026-11-10T18:38:18-08:00 smoke=0
repo=f7d5615e83a9e6347b33ba5a52bb58fd6f7f3eae 3 modified files
dmtcp=aeaa5461a0a1a4ff7b29004d2e906ae8b0590795
NCCL_PROFILE=default
NET_PROFILE=socket
REPS=3
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
| is | 1x4 (4 GPUs) | is_mops | 913.093 ± 11.534 | 886.213 ± 59.661 | +3.03% | 919.607 ± 6.086 | 30.05 ± 0.10 | 30.48 ± 0.65 |
| is | 2x4 (8 GPUs) | is_mops | 1272.317 ± 5.917 | 1272.310 ± 18.322 | +0.00% | 1395.010 ± 1.110 | 27.08 ± 0.12 | 27.48 ± 0.26 |
| is | 4x4 (16 GPUs) | is_mops | 2541.847 ± 28.939 | 2576.280 ± 50.726 | -1.34% | 3125.863 ± 11.362 | 29.49 ± 5.93 | 26.25 ± 0.68 |
| cg | 1x4 (4 GPUs) | cg_mops | 7497.187 ± 11.912 | 7441.350 ± 151.561 | +0.75% | 7462.350 ± 68.544 | 43.84 ± 0.05 | 43.93 ± 0.62 |
| cg | 2x4 (8 GPUs) | cg_mops | 12457.917 ± 509.015 | 12655.857 ± 107.821 | -1.56% | 13796.027 ± 277.018 | 28.61 ± 0.77 | 28.26 ± 0.43 |
| cg | 4x4 (16 GPUs) | cg_mops | 19583.887 ± 118.524 | 19132.623 ± 330.688 | +2.36% | 21819.953 ± 76.376 | 27.64 ± 0.19 | 27.92 ± 0.47 |
| mg | 1x4 (4 GPUs) | mg_mops | 29629.810 ± 59.388 | 29611.587 ± 55.225 | +0.06% | 29316.973 ± 252.294 | 28.25 ± 0.20 | 28.69 ± 0.07 |
| mg | 2x4 (8 GPUs) | mg_mops | 51201.960 ± 375.989 | 51173.103 ± 399.991 | +0.06% | 54566.837 ± 858.157 | 27.64 ± 0.09 | 27.85 ± 0.27 |
| mg | 4x4 (16 GPUs) | mg_mops | 91486.623 ± 627.417 | 91307.697 ± 638.460 | +0.20% | 99754.853 ± 470.721 | 26.42 ± 0.06 | 26.85 ± 0.13 |
| pytorch | 1x4 (4 GPUs) | pt_steps_per_s | 284.610 ± 0.763 | 279.553 ± 0.939 | +1.81% | 281.461 ± 3.402 | 45.40 ± 13.70 | 30.76 ± 3.53 |
| pytorch | 2x4 (8 GPUs) | pt_steps_per_s | 22.512 ± 0.533 | 23.526 ± 1.236 | -4.31% | 166.332 ± 0.029 | 65.64 ± 43.97 | 45.43 ± 2.27 |
| pytorch | 4x4 (16 GPUs) | pt_steps_per_s | 19.400 ± 1.049 | 19.996 ± 1.669 | -2.98% | 159.934 ± 0.072 | 60.37 ± 27.37 | 44.64 ± 0.97 |
| nccl_allgather | 1x2 (2 GPUs) | us_at_max | 1338.567 ± 0.265 | 1339.517 ± 0.892 | +0.07% | 1337.160 ± 3.282 | 23.34 ± 0.15 | 23.52 ± 0.21 |
| nccl_allgather | 2x1 (2 GPUs) | us_at_max | 50893.013 ± 430.901 | 49418.490 ± 747.829 | -2.90% | 4443.883 ± 28.042 | 18.00 ± 0.08 | 18.22 ± 0.14 |
| nccl_allreduce | 1x2 (2 GPUs) | us_at_max | 1047.640 ± 0.452 | 1048.187 ± 0.172 | +0.05% | 1047.197 ± 1.418 | 23.27 ± 0.16 | 23.53 ± 0.06 |
| nccl_allreduce | 2x1 (2 GPUs) | us_at_max | 48769.350 ± 748.697 | 49633.253 ± 333.096 | +1.77% | 3624.713 ± 13.633 | 21.36 ± 0.42 | 21.43 ± 0.16 |
| nccl_alltoall | 1x2 (2 GPUs) | us_at_max | 1194.173 ± 3.656 | 1193.060 ± 0.608 | -0.09% | 1194.857 ± 0.368 | 22.79 ± 0.04 | 23.18 ± 0.10 |
| nccl_alltoall | 2x1 (2 GPUs) | us_at_max | 48963.207 ± 726.536 | 53574.393 ± 5797.382 | +9.42% | 4328.547 ± 14.955 | 18.08 ± 0.08 | 18.45 ± 0.44 |
| nccl_bcast | 1x2 (2 GPUs) | us_at_max | 677.787 ± 0.812 | 678.647 ± 0.451 | +0.13% | 678.133 ± 0.371 | 23.79 ± 0.30 | 24.21 ± 0.40 |
| nccl_bcast | 2x1 (2 GPUs) | us_at_max | 17968.623 ± 71.879 | 18099.407 ± 83.534 | +0.73% | 2850.990 ± 3.283 | 34.76 ± 0.24 | 35.00 ± 0.63 |
| nccl_bibw | 1x2 (2 GPUs) | MBps_at_max | 117684.940 ± 30.624 | 117649.700 ± 86.785 | +0.03% | 117744.907 ± 108.044 | 18.22 ± 0.11 | 18.58 ± 0.17 |
| nccl_bibw | 2x1 (2 GPUs) | MBps_at_max | 2571.133 ± 54.921 | 2616.787 ± 187.995 | -1.74% | 37859.150 ± 318.489 | 156.20 ± 2.23 | 156.62 ± 5.65 |
| nccl_bw | 1x2 (2 GPUs) | MBps_at_max | 78971.697 ± 21.734 | 78960.533 ± 32.701 | +0.01% | 78931.040 ± 49.887 | 17.41 ± 0.06 | 17.62 ± 0.15 |
| nccl_bw | 2x1 (2 GPUs) | MBps_at_max | 2744.000 ± 31.625 | 2728.150 ± 91.063 | +0.58% | 22713.607 ± 17.869 | 79.91 ± 0.95 | 81.36 ± 2.37 |
| nccl_latency | 1x2 (2 GPUs) | us_at_max | 885.353 ± 0.864 | 885.273 ± 0.250 | -0.01% | 885.033 ± 1.715 | 15.31 ± 0.14 | 15.90 ± 0.28 |
| nccl_latency | 2x1 (2 GPUs) | us_at_max | 24454.470 ± 617.493 | 24333.127 ± 890.079 | -0.50% | 2997.953 ± 7.348 | 15.91 ± 0.07 | 16.07 ± 0.43 |
| nccl_reduce | 1x2 (2 GPUs) | us_at_max | 634.273 ± 0.664 | 635.273 ± 0.506 | +0.16% | 634.703 ± 0.320 | 23.77 ± 0.09 | 24.33 ± 0.18 |
| nccl_reduce | 2x1 (2 GPUs) | us_at_max | 17791.327 ± 70.809 | 17706.513 ± 21.508 | -0.48% | 2821.853 ± 0.381 | 17.24 ± 0.20 | 17.62 ± 0.32 |
| nccl_reduce_scatter | 1x2 (2 GPUs) | us_at_max | 601.133 ± 0.376 | 601.863 ± 0.092 | +0.12% | 600.653 ± 0.278 | 22.85 ± 0.42 | 23.55 ± 0.04 |
| nccl_reduce_scatter | 2x1 (2 GPUs) | us_at_max | 25160.093 ± 196.613 | 25143.360 ± 581.574 | -0.07% | 1976.907 ± 5.208 | 16.91 ± 0.11 | 17.34 ± 0.07 |

## Checkpoint and restart (E1, E2, E3)

`ckpt` = from `dmtcp_command -c` until the coordinator writes the restart script, i.e. every image is on disk (NCCL consensus, drain and teardown, CUDA checkpoint, image write). `nccl` = the NCCL plugin's share (max over ranks). `rounds` = most consensus rounds any checkpoint took. `resume rebuild` = communicator re-creation after a checkpoint the job survives. `restart latency` = `dmtcp_restart` until every rank has rebuilt its communicators; `restart total` = until the job completes.

| Benchmark | Scale | Mode | ok | ckpt (s) | nccl (s) | rounds | image total (GiB) | image/rank (MiB) | write (MiB/s) | resume rebuild (s) | restart latency (s) | restart rebuild (s) | restart total (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| is | 1x4 (4 GPUs) | ckpt_restart | 3/3 | 5.29 ± 0.15 | 0.980 ± 0.156 | 2 | 7.27 ± 0.00 | 1860 ± 0 | 1407 ± 39 | - | 4.76 ± 0.04 | 0.762 ± 0.177 | 39.68 ± 0.86 |
| is | 1x4 (4 GPUs) | ckpt_resume | 3/3 | 4.98 ± 0.15 | 0.833 ± 0.239 | 2 | 7.26 ± 0.00 | 1860 ± 0 | 1496 ± 46 | 0.755 ± 0.136 | - | - | - |
| is | 2x4 (8 GPUs) | ckpt_restart | 2/3 | 5.02 ± 0.30 | 1.004 ± 0.120 | 2 | 13.02 ± 0.00 | 1667 ± 0 | 2663 ± 157 | - | 4.82 ± 0.11 | 0.897 ± 0.104 | 39.16 ± 0.18 |
| is | 2x4 (8 GPUs) | ckpt_resume | 3/3 | 4.91 ± 0.08 | 0.894 ± 0.080 | 2 | 13.03 ± 0.00 | 1667 ± 0 | 2715 ± 45 | 0.843 ± 0.086 | - | - | - |
| is | 4x4 (16 GPUs) | ckpt_restart | 3/3 | 4.70 ± 0.08 | 0.856 ± 0.042 | 2 | 24.57 ± 0.00 | 1573 ± 0 | 5352 ± 86 | - | 4.73 ± 0.03 | 0.793 ± 0.039 | 38.02 ± 2.68 |
| is | 4x4 (16 GPUs) | ckpt_resume | 3/3 | 4.95 ± 0.25 | 0.881 ± 0.249 | 2 | 24.57 ± 0.00 | 1572 ± 0 | 5093 ± 255 | 0.942 ± 0.096 | - | - | - |
| cg | 1x4 (4 GPUs) | ckpt_restart | 3/3 | 4.61 ± 0.06 | 0.871 ± 0.073 | 3 | 6.80 ± 0.00 | 1741 ± 1 | 1512 ± 20 | - | 4.19 ± 0.05 | 0.579 ± 0.040 | 32.67 ± 0.35 |
| cg | 1x4 (4 GPUs) | ckpt_resume | 3/3 | 4.78 ± 0.14 | 0.919 ± 0.208 | 3 | 6.81 ± 0.00 | 1742 ± 0 | 1459 ± 45 | 0.557 ± 0.081 | - | - | - |
| cg | 2x4 (8 GPUs) | ckpt_restart | 3/3 | 4.64 ± 0.26 | 0.875 ± 0.214 | 3 | 12.59 ± 0.00 | 1612 ± 0 | 2785 ± 161 | - | 4.51 ± 0.30 | 0.732 ± 0.138 | 18.02 ± 0.43 |
| cg | 2x4 (8 GPUs) | ckpt_resume | 3/3 | 4.72 ± 0.24 | 0.926 ± 0.177 | 2 | 12.60 ± 0.00 | 1612 ± 1 | 2736 ± 144 | 0.640 ± 0.021 | - | - | - |
| cg | 4x4 (16 GPUs) | ckpt_restart | 3/3 | 4.53 ± 0.10 | 0.873 ± 0.135 | 3 | 24.15 ± 0.00 | 1545 ± 0 | 5463 ± 114 | - | 4.13 ± 0.00 | 0.678 ± 0.064 | 45.58 ± 0.85 |
| cg | 4x4 (16 GPUs) | ckpt_resume | 3/3 | 4.70 ± 0.13 | 1.007 ± 0.132 | 3 | 24.15 ± 0.00 | 1546 ± 0 | 5267 ± 149 | 0.716 ± 0.044 | - | - | - |
| mg | 1x4 (4 GPUs) | ckpt_restart | 3/3 | 5.53 ± 0.05 | 0.733 ± 0.107 | 2 | 9.15 ± 0.02 | 2342 ± 5 | 1693 ± 19 | - | 4.62 ± 0.19 | 0.575 ± 0.056 | 43.37 ± 0.44 |
| mg | 1x4 (4 GPUs) | ckpt_resume | 3/3 | 5.54 ± 0.11 | 0.737 ± 0.099 | 2 | 9.15 ± 0.02 | 2342 ± 4 | 1692 ± 35 | 0.608 ± 0.023 | - | - | - |
| mg | 2x4 (8 GPUs) | ckpt_restart | 3/3 | 5.16 ± 0.18 | 0.893 ± 0.159 | 3 | 14.91 ± 0.00 | 1909 ± 0 | 2963 ± 104 | - | 4.38 ± 0.31 | 0.711 ± 0.144 | 42.53 ± 1.54 |
| mg | 2x4 (8 GPUs) | ckpt_resume | 3/3 | 5.17 ± 0.14 | 0.892 ± 0.042 | 2 | 14.91 ± 0.00 | 1909 ± 0 | 2957 ± 83 | 0.689 ± 0.097 | - | - | - |
| mg | 4x4 (16 GPUs) | ckpt_restart | 3/3 | 5.07 ± 0.14 | 0.922 ± 0.023 | 4 | 26.47 ± 0.03 | 1694 ± 2 | 5354 ± 146 | - | 4.71 ± 0.01 | 0.719 ± 0.010 | 42.43 ± 0.37 |
| mg | 4x4 (16 GPUs) | ckpt_resume | 3/3 | 4.97 ± 0.16 | 0.892 ± 0.117 | 2 | 26.47 ± 0.04 | 1694 ± 2 | 5455 ± 181 | 0.738 ± 0.129 | - | - | - |
| pytorch | 1x4 (4 GPUs) | ckpt_restart | 3/3 | 10.93 ± 0.15 | 0.897 ± 0.144 | 2 | 21.71 ± 0.07 | 5557 ± 19 | 2035 ± 35 | - | 15.59 ± 0.35 | 0.660 ± 0.092 | 33.04 ± 0.16 |
| pytorch | 1x4 (4 GPUs) | ckpt_resume | 3/3 | 18.30 ± 12.81 | 1.003 ± 0.270 | 2 | 21.67 ± 0.13 | 5547 ± 32 | 1578 ± 783 | 0.591 ± 0.082 | - | - | - |
| pytorch | 2x4 (8 GPUs) | ckpt_restart | 3/3 | 10.78 ± 0.11 | 0.909 ± 0.146 | 2 | 42.78 ± 0.00 | 5476 ± 0 | 4064 ± 43 | - | 15.86 ± 0.09 | 0.818 ± 0.057 | 52.26 ± 4.35 |
| pytorch | 2x4 (8 GPUs) | ckpt_resume | 3/3 | 26.25 ± 26.66 | 1.045 ± 0.085 | 2 | 42.78 ± 0.01 | 5476 ± 1 | 2946 ± 1887 | 0.847 ± 0.067 | - | - | - |
| pytorch | 4x4 (16 GPUs) | ckpt_restart | 3/3 | 10.92 ± 0.22 | 0.939 ± 0.186 | 2 | 85.56 ± 0.00 | 5476 ± 0 | 8023 ± 158 | - | 15.93 ± 0.37 | 0.816 ± 0.029 | 47.79 ± 2.20 |
| pytorch | 4x4 (16 GPUs) | ckpt_resume | 3/3 | 24.87 ± 24.75 | 0.872 ± 0.225 | 2 | 85.56 ± 0.00 | 5476 ± 0 | 6066 ± 3833 | 0.740 ± 0.042 | - | - | - |
| pytorch_w8192 | 1x4 (4 GPUs) | ckpt_restart | 3/3 | 13.69 ± 0.43 | 0.933 ± 0.228 | 2 | 28.86 ± 0.77 | 7389 ± 197 | 2160 ± 27 | - | 15.69 ± 0.30 | 0.681 ± 0.096 | 48.11 ± 1.70 |
| pytorch_w8192 | 1x4 (4 GPUs) | ckpt_resume | 3/3 | 13.96 ± 0.31 | 0.998 ± 0.214 | 2 | 29.71 ± 0.50 | 7605 ± 129 | 2180 ± 42 | 0.679 ± 0.122 | - | - | - |
| pytorch_w8192 | 2x4 (8 GPUs) | ckpt_restart | 3/3 | 13.16 ± 0.12 | 0.821 ± 0.100 | 2 | 56.61 ± 0.01 | 7246 ± 1 | 4406 ± 40 | - | 15.93 ± 0.16 | 0.850 ± 0.028 | 53.85 ± 0.88 |
| pytorch_w8192 | 2x4 (8 GPUs) | ckpt_resume | 3/3 | 13.30 ± 0.15 | 0.955 ± 0.074 | 2 | 56.60 ± 0.00 | 7245 ± 0 | 4357 ± 49 | 0.690 ± 0.028 | - | - | - |
| pytorch_w8192 | 4x4 (16 GPUs) | ckpt_restart | 3/3 | 13.22 ± 0.11 | 0.881 ± 0.009 | 2 | 113.21 ± 0.01 | 7246 ± 1 | 8770 ± 74 | - | 16.50 ± 0.20 | 0.857 ± 0.098 | 49.73 ± 2.66 |
| pytorch_w8192 | 4x4 (16 GPUs) | ckpt_resume | 3/3 | 13.33 ± 0.16 | 0.960 ± 0.158 | 2 | 113.21 ± 0.01 | 7246 ± 0 | 8700 ± 103 | 0.862 ± 0.126 | - | - | - |
| pytorch_w16384 | 1x4 (4 GPUs) | ckpt_restart | 3/3 | 25.41 ± 0.59 | 1.054 ± 0.130 | 2 | 61.87 ± 1.16 | 15839 ± 296 | 2493 ± 14 | - | 17.52 ± 0.05 | 0.604 ± 0.136 | 54.03 ± 0.29 |
| pytorch_w16384 | 1x4 (4 GPUs) | ckpt_resume | 3/3 | 24.94 ± 0.21 | 0.687 ± 0.037 | 2 | 60.54 ± 2.32 | 15497 ± 593 | 2485 ± 80 | 0.559 ± 0.304 | - | - | - |
| pytorch_w16384 | 2x4 (8 GPUs) | ckpt_restart | 3/3 | 23.62 ± 0.30 | 1.770 ± 0.263 | 2 | 110.61 ± 0.00 | 14158 ± 0 | 4795 ± 60 | - | 17.45 ± 0.29 | 0.888 ± 0.093 | 61.28 ± 1.86 |
| pytorch_w16384 | 2x4 (8 GPUs) | ckpt_resume | 3/3 | 23.53 ± 0.29 | 1.705 ± 0.266 | 2 | 110.62 ± 0.00 | 14159 ± 0 | 4814 ± 61 | 0.725 ± 0.249 | - | - | - |
| pytorch_w16384 | 4x4 (16 GPUs) | ckpt_restart | 3/3 | 23.87 ± 0.45 | 2.186 ± 0.322 | 2 | 221.23 ± 0.00 | 14159 ± 0 | 9494 ± 182 | - | 17.20 ± 0.21 | 0.815 ± 0.034 | 57.98 ± 3.61 |
| pytorch_w16384 | 4x4 (16 GPUs) | ckpt_resume | 3/3 | 24.40 ± 1.91 | 1.788 ± 0.472 | 2 | 221.22 ± 0.01 | 14158 ± 1 | 9320 ± 699 | 0.855 ± 0.013 | - | - | - |
| nccl_allreduce | 1x2 (2 GPUs) | ckpt_restart | 3/3 | 3.35 ± 0.29 | 0.648 ± 0.193 | 2 | 2.37 ± 0.00 | 1215 ± 1 | 730 ± 66 | - | 2.62 ± 0.00 | 0.235 ± 0.021 | 45.14 ± 0.51 |
| nccl_allreduce | 1x2 (2 GPUs) | ckpt_resume | 3/3 | 3.41 ± 0.19 | 0.580 ± 0.060 | 2 | 2.37 ± 0.00 | 1215 ± 1 | 714 ± 40 | 0.259 ± 0.010 | - | - | - |
| nccl_allreduce | 2x1 (2 GPUs) | ckpt_restart | 3/3 | 2.27 ± 0.17 | 0.284 ± 0.104 | 2 | 1.94 ± 0.00 | 994 ± 0 | 880 ± 66 | - | 2.10 ± 0.01 | 0.075 ± 0.027 | 36.70 ± 1.03 |
| nccl_allreduce | 2x1 (2 GPUs) | ckpt_resume | 3/3 | 2.22 ± 0.10 | 0.250 ± 0.036 | 2 | 1.94 ± 0.00 | 994 ± 1 | 895 ± 40 | 0.143 ± 0.015 | - | - | - |
| nccl_alltoall | 1x2 (2 GPUs) | ckpt_restart | 3/3 | 3.32 ± 0.03 | 0.484 ± 0.113 | 3 | 2.46 ± 0.00 | 1261 ± 1 | 759 ± 5 | - | 2.62 ± 0.01 | 0.223 ± 0.036 | 45.36 ± 0.41 |
| nccl_alltoall | 1x2 (2 GPUs) | ckpt_resume | 3/3 | 3.39 ± 0.11 | 0.568 ± 0.123 | 2 | 2.46 ± 0.00 | 1260 ± 1 | 743 ± 23 | 0.281 ± 0.040 | - | - | - |
| nccl_alltoall | 2x1 (2 GPUs) | ckpt_restart | 3/3 | 2.42 ± 0.14 | 0.380 ± 0.086 | 2 | 2.03 ± 0.00 | 1039 ± 0 | 861 ± 50 | - | 1.93 ± 0.30 | 0.077 ± 0.049 | 37.80 ± 0.64 |
| nccl_alltoall | 2x1 (2 GPUs) | ckpt_resume | 3/3 | 2.38 ± 0.10 | 0.337 ± 0.024 | 2 | 2.03 ± 0.00 | 1040 ± 1 | 875 ± 38 | 0.141 ± 0.119 | - | - | - |
| nccl_bw | 1x2 (2 GPUs) | ckpt_restart | 3/3 | 3.33 ± 0.12 | 0.551 ± 0.193 | 2 | 2.44 ± 0.00 | 1251 ± 1 | 751 ± 27 | - | 2.81 ± 0.34 | 0.238 ± 0.034 | 38.65 ± 0.32 |
| nccl_bw | 1x2 (2 GPUs) | ckpt_resume | 3/3 | 3.26 ± 0.05 | 0.424 ± 0.151 | 2 | 2.45 ± 0.00 | 1253 ± 2 | 768 ± 11 | 0.208 ± 0.028 | - | - | - |
| nccl_bw | 2x1 (2 GPUs) | ckpt_restart | 3/3 | 2.34 ± 0.22 | 0.275 ± 0.214 | 2 | 2.02 ± 0.00 | 1032 ± 1 | 889 ± 81 | - | 2.10 ± 0.01 | 0.124 ± 0.039 | 38.67 ± 1.66 |
| nccl_bw | 2x1 (2 GPUs) | ckpt_resume | 3/3 | 2.35 ± 0.14 | 0.262 ± 0.043 | 2 | 2.02 ± 0.00 | 1033 ± 1 | 880 ± 54 | 0.107 ± 0.070 | - | - | - |

## torch.save baseline (E3)

Per save: `d2h` = copy of the state dict (model + SGD momentum) to host memory, `write` = torch.save of that copy, `fsync` = until it is on the filesystem; `job save` = barrier to barrier, the time the job is stopped. `job load` = torch.load to the GPU + load_state_dict, page cache dropped first. Each phase is the max over ranks; mean ± std over every save of every run.

| Scale | Width | Params | Policy | n | payload/rank (MiB) | d2h (s) | write (s) | fsync (s) | job save (s) | job load (s) | MiB/s per rank |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1x4 (4 GPUs) | 4096 | 33,603,594 | all | 15 | 256 ± 0 | 0.045 ± 0.001 | 0.365 ± 0.017 | 0.006 ± 0.001 | 0.416 ± 0.017 | 0.301 ± 0.002 | 620 ± 23 |
| 1x4 (4 GPUs) | 4096 | 33,603,594 | rank0 | 15 | 256 ± 0 | 0.044 ± 0.001 | 0.334 ± 0.002 | 0.008 ± 0.003 | 0.388 ± 0.003 | 0.285 ± 0.003 | 665 ± 6 |
| 1x4 (4 GPUs) | 8192 | 134,316,042 | all | 15 | 1025 ± 0 | 0.173 ± 0.004 | 1.440 ± 0.011 | 0.009 ± 0.001 | 1.617 ± 0.013 | 1.353 ± 0.008 | 635 ± 5 |
| 1x4 (4 GPUs) | 8192 | 134,316,042 | rank0 | 15 | 1025 ± 0 | 0.166 ± 0.004 | 1.345 ± 0.011 | 0.009 ± 0.004 | 1.523 ± 0.012 | 1.120 ± 0.012 | 675 ± 5 |
| 1x4 (4 GPUs) | 16384 | 537,067,530 | all | 15 | 4098 ± 0 | 0.674 ± 0.011 | 5.819 ± 0.038 | 0.009 ± 0.004 | 6.494 ± 0.040 | 5.481 ± 0.120 | 632 ± 4 |
| 1x4 (4 GPUs) | 16384 | 537,067,530 | rank0 | 15 | 4098 ± 0 | 0.652 ± 0.018 | 5.343 ± 0.030 | 0.008 ± 0.005 | 6.017 ± 0.034 | 4.627 ± 0.047 | 683 ± 4 |
| 2x4 (8 GPUs) | 4096 | 33,603,594 | all | 15 | 256 ± 0 | 0.048 ± 0.002 | 0.384 ± 0.053 | 0.007 ± 0.001 | 0.436 ± 0.054 | 0.309 ± 0.002 | 598 ± 60 |
| 2x4 (8 GPUs) | 4096 | 33,603,594 | rank0 | 15 | 256 ± 0 | 0.045 ± 0.001 | 0.339 ± 0.001 | 0.007 ± 0.002 | 0.393 ± 0.003 | 0.295 ± 0.003 | 655 ± 4 |
| 2x4 (8 GPUs) | 8192 | 134,316,042 | all | 15 | 1025 ± 0 | 0.172 ± 0.004 | 1.445 ± 0.014 | 0.010 ± 0.003 | 1.625 ± 0.014 | 1.373 ± 0.008 | 632 ± 5 |
| 2x4 (8 GPUs) | 8192 | 134,316,042 | rank0 | 15 | 1025 ± 0 | 0.165 ± 0.003 | 1.348 ± 0.010 | 0.007 ± 0.002 | 1.525 ± 0.011 | 1.141 ± 0.005 | 674 ± 5 |
| 2x4 (8 GPUs) | 16384 | 537,067,530 | all | 15 | 4098 ± 0 | 0.671 ± 0.011 | 5.809 ± 0.037 | 0.010 ± 0.005 | 6.479 ± 0.042 | 5.565 ± 0.112 | 634 ± 4 |
| 2x4 (8 GPUs) | 16384 | 537,067,530 | rank0 | 15 | 4098 ± 0 | 0.644 ± 0.007 | 5.374 ± 0.033 | 0.008 ± 0.005 | 6.040 ± 0.035 | 4.702 ± 0.028 | 680 ± 4 |
| 4x4 (16 GPUs) | 4096 | 33,603,594 | all | 15 | 256 ± 0 | 0.049 ± 0.002 | 0.390 ± 0.064 | 0.011 ± 0.005 | 0.444 ± 0.063 | 0.307 ± 0.002 | 589 ± 67 |
| 4x4 (16 GPUs) | 4096 | 33,603,594 | rank0 | 15 | 256 ± 0 | 0.045 ± 0.001 | 0.336 ± 0.003 | 0.010 ± 0.004 | 0.393 ± 0.006 | 0.289 ± 0.002 | 656 ± 9 |
| 4x4 (16 GPUs) | 8192 | 134,316,042 | all | 15 | 1025 ± 0 | 0.174 ± 0.004 | 1.446 ± 0.008 | 0.013 ± 0.006 | 1.624 ± 0.010 | 1.369 ± 0.008 | 633 ± 4 |
| 4x4 (16 GPUs) | 8192 | 134,316,042 | rank0 | 15 | 1025 ± 0 | 0.166 ± 0.004 | 1.343 ± 0.009 | 0.011 ± 0.007 | 1.524 ± 0.012 | 1.122 ± 0.006 | 674 ± 5 |
| 4x4 (16 GPUs) | 16384 | 537,067,530 | all | 15 | 4098 ± 0 | 0.685 ± 0.019 | 5.857 ± 0.032 | 0.014 ± 0.008 | 6.531 ± 0.033 | 5.586 ± 0.079 | 629 ± 3 |
| 4x4 (16 GPUs) | 16384 | 537,067,530 | rank0 | 15 | 4098 ± 0 | 0.655 ± 0.016 | 5.346 ± 0.033 | 0.007 ± 0.002 | 6.022 ± 0.036 | 4.627 ± 0.022 | 682 ± 4 |

## torch.save vs. transparent checkpoint, same model (E3)

torch.save: every rank saves its replica (policy `all`; `rank0` is the usual DDP practice). DMTCP: `ckpt` over the resume and restart runs of the same model and layout; restart latency from the restart runs. The DMTCP image holds the whole process (CUDA context, Python, NCCL, optimizer), torch.save only the tensors.

| Scale | Width | torch.save all (s) | torch.save rank0 (s) | DMTCP ckpt (s) | ckpt / save | torch.load (s) | DMTCP restart (s) | restart / load | payload/rank (MiB) | image/rank (MiB) | image / payload |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1x4 | 4096 | 0.416 | 0.388 | 14.612 | 35.2 | 0.301 | 15.585 | 51.7 | 256 | 5552 | 21.7 |
| 1x4 | 8192 | 1.617 | 1.523 | 13.821 | 8.5 | 1.353 | 15.692 | 11.6 | 1025 | 7497 | 7.3 |
| 1x4 | 16384 | 6.494 | 6.017 | 25.178 | 3.9 | 5.481 | 17.519 | 3.2 | 4098 | 15668 | 3.8 |
| 2x4 | 4096 | 0.436 | 0.393 | 18.517 | 42.5 | 0.309 | 15.863 | 51.3 | 256 | 5476 | 21.4 |
| 2x4 | 8192 | 1.625 | 1.525 | 13.230 | 8.1 | 1.373 | 15.925 | 11.6 | 1025 | 7245 | 7.1 |
| 2x4 | 16384 | 6.479 | 6.040 | 23.577 | 3.6 | 5.565 | 17.452 | 3.1 | 4098 | 14158 | 3.5 |
| 4x4 | 4096 | 0.444 | 0.393 | 17.898 | 40.3 | 0.307 | 15.934 | 51.9 | 256 | 5476 | 21.4 |
| 4x4 | 8192 | 1.624 | 1.524 | 13.273 | 8.2 | 1.369 | 16.504 | 12.1 | 1025 | 7246 | 7.1 |
| 4x4 | 16384 | 6.531 | 6.022 | 24.134 | 3.7 | 5.586 | 17.202 | 3.1 | 4098 | 14158 | 3.5 |

## Checkpoints with pending point-to-point operations (E4)

PASS = the run verified (every receiver got its sender's data, before and after the checkpoint) and no `[NCCL Drain]` line appeared, i.e. the checkpoint waited for the missing counterpart instead of completing the other side with a dummy message.

| Scenario | Scale | Mode | Verdict | Status | ckpt (s) | rounds | drained |
|---|---|---|---|---|---|---|---|
| chain | 1x3 (3 GPUs) | ckpt_restart | PASS | ok | 22.241 | 27 | 0 |
| chain | 1x3 (3 GPUs) | ckpt_resume | PASS | ok | 22.28 | 27 | 0 |
| chain | 2x2 (4 GPUs) | ckpt_restart | PASS | ok | 22.164 | 27 | 0 |
| chain | 2x2 (4 GPUs) | ckpt_resume | PASS | ok | 22.039 | 27 | 0 |
| create | 1x2 (2 GPUs) | ckpt_restart | PASS | ok | 22.08 | 25 | 0 |
| create | 1x2 (2 GPUs) | ckpt_resume | PASS | ok | 21.931 | 25 | 0 |
| destroy | 1x2 (2 GPUs) | ckpt_restart | PASS | ok | 22.106 | 26 | 0 |
| destroy | 1x2 (2 GPUs) | ckpt_resume | PASS | ok | 21.752 | 26 | 0 |
| multi | 1x4 (4 GPUs) | ckpt_restart | PASS | ok | 23.561 | 27 | 0 |
| multi | 1x4 (4 GPUs) | ckpt_resume | PASS | ok | 23.5 | 27 | 0 |
| multi | 2x2 (4 GPUs) | ckpt_restart | PASS | ok | 22.052 | 27 | 0 |
| multi | 2x2 (4 GPUs) | ckpt_resume | PASS | ok | 22.092 | 27 | 0 |
| recv_first | 1x2 (2 GPUs) | ckpt_restart | PASS | ok | 22.064 | 27 | 0 |
| recv_first | 1x2 (2 GPUs) | ckpt_resume | PASS | ok | 21.988 | 27 | 0 |
| recv_first | 2x1 (2 GPUs) | ckpt_restart | PASS | ok | 21.429 | 27 | 0 |
| recv_first | 2x1 (2 GPUs) | ckpt_resume | PASS | ok | 21.278 | 27 | 0 |
| recv_first_cold | 1x2 (2 GPUs) | ckpt_restart | PASS | ok | 22.116 | 27 | 0 |
| recv_first_cold | 1x2 (2 GPUs) | ckpt_resume | PASS | ok | 21.933 | 27 | 0 |
| relay | 1x3 (3 GPUs) | ckpt_restart | PASS | ok | 23.451 | 28 | 0 |
| relay | 1x3 (3 GPUs) | ckpt_resume | PASS | ok | 23.608 | 28 | 0 |
| send_first | 1x2 (2 GPUs) | ckpt_restart | PASS | ok | 22.104 | 27 | 0 |
| send_first | 1x2 (2 GPUs) | ckpt_resume | PASS | ok | 21.741 | 27 | 0 |
| send_first | 2x1 (2 GPUs) | ckpt_restart | PASS | ok | 21.252 | 27 | 0 |
| send_first | 2x1 (2 GPUs) | ckpt_resume | PASS | ok | 21.156 | 26 | 0 |
| split | 1x2 (2 GPUs) | ckpt_restart | PASS | ok | 21.789 | 26 | 0 |
| split | 1x2 (2 GPUs) | ckpt_resume | PASS | ok | 22.14 | 26 | 0 |

## Failed runs

| Benchmark | Scale | Mode | Rep | Status | First error | Run dir |
|---|---|---|---|---|---|---|
| is | 2x4 (8 GPUs) | ckpt_restart | 1 | fail_startup |  | /pscratch/sd/m/masonstu/nccl-ckpt-paper/20261110/runs/is/n2x4/ckpt_restart/rep1 |

