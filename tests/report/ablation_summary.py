#!/usr/bin/env python3
"""Summarize an ablation.sh run into ablation_summary.md.

python3 tests/report/ablation_summary.py $SCRATCH/nccl-ckpt-report/<ablation-dir>
"""

import csv
import os
import re
import statistics
import sys
from collections import defaultdict

ERR_RE = re.compile(
    r"JASSERT|Segmentation fault|terminate called|CUDA_ERROR|cuCheckpoint|"
    r"NCCL WARN|Error\]|error:|FAIL|Killed",
)


def rows(d):
    p = os.path.join(d, "results.csv")
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return list(csv.DictReader(f))


def transports(run_dir):
    found = defaultdict(int)
    p = os.path.join(run_dir, "out.log")
    if os.path.exists(p):
        with open(p, errors="replace") as f:
            for line in f:
                m = re.search(r"via ((?:P2P|SHM|NET)\S*)", line)
                if m:
                    found[m.group(1)] += 1
    return " ".join(sorted(found)) or "-"


def first_error(run_dir):
    for name in ("out.log", "restart.log", "ckpt_cmd.log"):
        p = os.path.join(run_dir, name)
        if not os.path.exists(p):
            continue
        with open(p, errors="replace") as f:
            for line in f:
                if ERR_RE.search(line) and "NCCL INFO" not in line:
                    return line.strip()[:160]
    return ""


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main():
    top = sys.argv[1]
    out = [f"# Ablation {os.path.basename(top.rstrip('/'))}\n"]

    # --- Part 0: IS timing --------------------------------------------------
    r = rows(os.path.join(top, "is_timing"))
    if r:
        out.append("## IS timing (class B, 1x2)\n")
        out.append("| Mode | Status | Metric | Value | Wall (s) |")
        out.append("|---|---|---|---|---|")
        for x in r:
            cells = [x["mode"], x["status"], x["metric_name"], x["metric"], x["wall_s"]]
            out.append("| " + " | ".join(cells) + " |")
        out.append("")

    # --- Part 1: ablation ---------------------------------------------------
    abl = sorted(d for d in os.listdir(top) if d.startswith("abl_"))
    if abl:
        out.append("## NCCL feature ablation (1x2, nccl_bw / nccl_allreduce)\n")
        out.append(
            "Each profile starts from `conservative` and re-enables one feature. "
            "Transport is what NCCL logged (`via ...`) in the native run. "
            "Native metric: MB/s at 1 MiB for bw, µs at 1 MiB for allreduce.\n"
        )
        out.append(
            "| Profile | Bench | Transport | Native | ckpt_resume | ckpt_restart | First error |"
        )
        out.append("|---|---|---|---|---|---|---|")
        for d in abl:
            prof = d[len("abl_") :]
            by = {(x["bench"], x["mode"]): x for x in rows(os.path.join(top, d))}
            for b in ("nccl_bw", "nccl_allreduce"):
                nat = by.get((b, "native"))
                res = by.get((b, "ckpt_resume"))
                rst = by.get((b, "ckpt_restart"))
                if not (nat or res or rst):
                    continue
                tr = transports(nat["run_dir"]) if nat else "-"
                natv = (
                    f"{nat['metric']}"
                    if nat and nat["status"] == "ok"
                    else (nat["status"] if nat else "-")
                )

                def cell(x, restart=False):
                    if not x:
                        return "-"
                    if x["status"] != "ok":
                        return f"**{x['status']}**"
                    s = f"ok, ckpt {x['ckpt_s']} s"
                    if restart:
                        s += f", restart {x['restart_latency_s']} s"
                    return s

                err = ""
                for x in (res, rst, nat):
                    if x and x["status"] != "ok":
                        err = first_error(x["run_dir"])
                        if err:
                            break
                out.append(
                    f"| {prof} | {b} | {tr} | {natv} | {cell(res)} | {cell(rst, True)} | "
                    f"{err.replace('|', '/')} |"
                )
        out.append("")

    # --- Part 2: latency probe -------------------------------------------------
    probe = {}
    for d in sorted(os.listdir(top)):
        if not d.startswith("probe_"):
            continue
        per_size = defaultdict(list)
        for x in rows(os.path.join(top, d)):
            if x["status"] != "ok":
                continue
            p = os.path.join(x["run_dir"], "sizes.csv")
            if os.path.exists(p):
                with open(p) as f:
                    for line in f:
                        s, v = line.strip().split(",")
                        per_size[int(s)].append(float(v))
        probe[d[len("probe_") :]] = per_size
    if probe:
        pairs = [
            ("dmtcp_full", "native", "DMTCP + NCCL + CUDA plugins"),
            ("dmtcp_noplugins", "native", "DMTCP, no plugins"),
            ("dmtcp_cudaonly", "native", "DMTCP + CUDA plugin only"),
            ("p2p_dmtcp", "p2p_native", "P2P profile: DMTCP + plugins"),
            ("ll128_dmtcp", "ll128_native", "NCCL_PROTO=LL128: DMTCP + plugins"),
        ]
        sizes = sorted({s for v in probe.values() for s in v})
        out.append("## Latency probe: nccl_latency, 1x2, mean of repetitions (µs)\n")
        hdr = "| Size (B) | " + " | ".join(sorted(probe)) + " |"
        out.append(hdr)
        out.append("|" + "---|" * (len(probe) + 1))
        for s in sizes:
            cells = []
            for k in sorted(probe):
                v = probe[k].get(s, [])
                if v:
                    sd = statistics.stdev(v) if len(v) > 1 else 0
                    cells.append(f"{statistics.mean(v):.1f} ± {sd:.1f}")
                else:
                    cells.append("-")
            out.append(f"| {s} | " + " | ".join(cells) + " |")
        out.append("")
        out.append("### Overhead vs the matching native run\n")
        out.append("| Size (B) | " + " | ".join(p[2] for p in pairs) + " |")
        out.append("|" + "---|" * (len(pairs) + 1))
        for s in sizes:
            cells = []
            for a, b, _ in pairs:
                va, vb = probe.get(a, {}).get(s), probe.get(b, {}).get(s)
                if va and vb:
                    cells.append(f"{(statistics.mean(va) / statistics.mean(vb) - 1) * 100:+.1f}%")
                else:
                    cells.append("-")
            out.append(f"| {s} | " + " | ".join(cells) + " |")
        out.append("")

    text = "\n".join(out)
    with open(os.path.join(top, "ablation_summary.md"), "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
