#!/usr/bin/env python3
"""Checkpoint and Restart Metrics Reporter for PyTorch AI Workloads.

Parses runtime timings, plugin overheads, storage footprints, and training
logs to compute and display comprehensive checkpointing cost metrics, and
exports a structured JSON report.
"""

import argparse
import json
import os
import re
import sys


def parse_avg_step_time(log_file):
    """Extract average step execution time from rank/restart logs."""
    if not os.path.exists(log_file):
        return None
    times = []
    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = re.search(r"step_time\s+([0-9.]+)ms", line)
            if m:
                try:
                    times.append(float(m.group(1)))
                except ValueError:
                    pass
    if times:
        return sum(times) / len(times)
    return None


def main():
    parser = argparse.ArgumentParser(description="Report AI Workload Checkpoint Metrics")
    parser.add_argument("--startup-ms", type=int, default=0, help="Startup duration to training readiness (ms)")
    parser.add_argument("--ckpt-trig-ms", type=int, default=0, help="Checkpoint trigger command duration (ms)")
    parser.add_argument("--ckpt-total-ms", type=int, default=0, help="Total checkpoint duration (drain + write) (ms)")
    parser.add_argument("--restart-total-ms", type=int, default=0, help="Restart execution duration (ms)")
    parser.add_argument("--first-step-ms", type=str, default="N/A", help="Time to first restarted step (ms)")
    parser.add_argument("--job-total-ms", type=int, default=0, help="Total job wall-clock time (ms)")
    parser.add_argument("--ckpt-bytes", type=int, default=0, help="Total bytes across checkpoint images")
    parser.add_argument("--nccl-overhead-sec", type=float, default=0.0, help="NCCL plugin checkpoint overhead (sec)")
    parser.add_argument("--cuda-overhead-sec", type=float, default=0.0, help="CUDA plugin checkpoint overhead (sec)")
    parser.add_argument("--ckpt-file-count", type=int, default=0, help="Number of checkpoint files generated")
    parser.add_argument("--rebuilt-comms", type=int, default=0, help="Number of communicators rebuilt")
    parser.add_argument("--pre-steps", type=int, default=0, help="Steps completed before checkpoint")
    parser.add_argument("--post-steps", type=int, default=0, help="Steps completed after restart")
    parser.add_argument("--ranks", type=int, default=1, help="Total distributed ranks")
    parser.add_argument("--gpus", type=int, default=1, help="Total GPUs used")
    parser.add_argument("--store", type=str, default="TCPStore (env://)", help="PyTorch distributed store method")
    parser.add_argument("--rank0-log", type=str, default="pt_rank0.log", help="Path to rank 0 training log")
    parser.add_argument("--restart-log", type=str, default="pt_restart.log", help="Path to restart log")
    parser.add_argument("--json-out", type=str, default="pt_ckpt_metrics.json", help="Path to JSON output report")

    args = parser.parse_args()

    # Conversions and calculations
    ckpt_mb = args.ckpt_bytes / (1024.0 * 1024.0)
    ckpt_gb = args.ckpt_bytes / (1024.0 * 1024.0 * 1024.0)
    ckpt_total_s = args.ckpt_total_ms / 1000.0
    disk_s = max(0.0, ckpt_total_s - args.nccl_overhead_sec - args.cuda_overhead_sec)
    disk_speed_mb_s = (ckpt_mb / disk_s) if disk_s > 0 else 0.0

    pre_avg_step_ms = parse_avg_step_time(args.rank0_log)
    post_avg_step_ms = parse_avg_step_time(args.restart_log)

    try:
        first_step_ms = int(args.first_step_ms)
    except (ValueError, TypeError):
        first_step_ms = None

    # Formatted Console Summary
    print("")
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║         AI WORKLOAD CHECKPOINT & RESTART METRICS             ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print("║  Configuration & Environment:                                ║")
    print(f"║    World Size / GPUs:         {args.ranks:>6} rank(s) / {args.gpus} GPU(s)        ║")
    print(f"║    Store Type:                {args.store[:30]:>30} ║")
    print(f"║    Comm Backend:              {'NCCL (MANA Plugin)':>30} ║")
    print("║──────────────────────────────────────────────────────────────║")
    print("║  Phase 1: Startup & Checkpoint Overhead:                     ║")
    print(f"║    Startup to training:       {args.startup_ms:>27} ms ║")
    print(f"║    Pre-ckpt steps (rank 0):   {args.pre_steps:>27}    ║")
    if pre_avg_step_ms is not None:
        print(f"║    Pre-ckpt avg step time:    {pre_avg_step_ms:>24.2f} ms ║")
    print(f"║    Checkpoint trigger (cmd):  {args.ckpt_trig_ms:>27} ms ║")
    print(f"║    Total Checkpoint Latency:  {args.ckpt_total_ms:>27} ms ║")
    print(f"║      ├─ NCCL Overhead (drain):{args.nccl_overhead_sec:>25.4f} s  ║")
    print(f"║      ├─ CUDA Overhead:        {args.cuda_overhead_sec:>25.4f} s  ║")
    print(f"║      └─ Disk Write (approx):  {disk_s:>25.4f} s  ║")
    print("║──────────────────────────────────────────────────────────────║")
    print("║  Checkpoint Storage Footprint:                               ║")
    print(f"║    Checkpoint images:         {args.ckpt_file_count:>22} file(s) ║")
    print(f"║    Total image size:          {ckpt_mb:>24.2f} MB ║")
    print(f"║                               {ckpt_gb:>24.3f} GB ║")
    print(f"║    Effective write speed:     {disk_speed_mb_s:>22.2f} MB/s ║")
    print("║──────────────────────────────────────────────────────────────║")
    print("║  Phase 2: Restart & Recovery:                                ║")
    print(f"║    NCCL comms rebuilt:        {args.rebuilt_comms:>27}    ║")
    if first_step_ms is not None:
        print(f"║    Time-to-first-step (TTFS): {first_step_ms:>27} ms ║")
    print(f"║    Post-restart duration:     {args.restart_total_ms:>27} ms ║")
    print(f"║    Post-restart steps:        {args.post_steps:>27}    ║")
    if post_avg_step_ms is not None:
        print(f"║    Post-restart avg step time:{post_avg_step_ms:>24.2f} ms ║")
    print("║──────────────────────────────────────────────────────────────║")
    print(f"║  Total Wall-Clock Time:       {args.job_total_ms:>27} ms ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print("")

    # Structured JSON Export
    metrics_data = {
        "config": {
            "ranks": args.ranks,
            "gpus": args.gpus,
            "store": args.store,
            "comm_backend": "NCCL (MANA Plugin)",
        },
        "phase1_startup_and_checkpoint": {
            "startup_to_training_ms": args.startup_ms,
            "pre_ckpt_steps": args.pre_steps,
            "pre_ckpt_avg_step_ms": round(pre_avg_step_ms, 2) if pre_avg_step_ms is not None else None,
            "ckpt_trigger_cmd_ms": args.ckpt_trig_ms,
            "ckpt_total_latency_ms": args.ckpt_total_ms,
            "nccl_overhead_sec": args.nccl_overhead_sec,
            "cuda_overhead_sec": args.cuda_overhead_sec,
            "disk_write_sec": round(disk_s, 4),
        },
        "checkpoint_storage": {
            "ckpt_file_count": args.ckpt_file_count,
            "ckpt_size_bytes": args.ckpt_bytes,
            "ckpt_size_mb": round(ckpt_mb, 2),
            "ckpt_size_gb": round(ckpt_gb, 3),
            "disk_write_speed_mb_s": round(disk_speed_mb_s, 2),
        },
        "phase2_restart_and_recovery": {
            "rebuilt_comms": args.rebuilt_comms,
            "time_to_first_step_ms": first_step_ms,
            "post_restart_duration_ms": args.restart_total_ms,
            "post_restart_steps": args.post_steps,
            "post_restart_avg_step_ms": round(post_avg_step_ms, 2) if post_avg_step_ms is not None else None,
        },
        "total_wall_clock_ms": args.job_total_ms,
    }

    try:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(metrics_data, f, indent=2)
        print(f"[metrics] Detailed JSON metrics report saved to: {args.json_out}")
    except Exception as e:
        print(f"[metrics] Warning: Could not write JSON metrics file: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
