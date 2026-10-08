#!/usr/bin/env python3
"""Aggregate a campaign's results.csv into report tables.

    python3 tests/report/summarize.py $SCRATCH/nccl-ckpt-report/<campaign>

Writes, next to results.csv:
    summary.md        Markdown tables (runtime overhead, checkpoint/restart, nccl_bench per size)
    agg_overhead.csv  one row per (bench, scale): native / dmtcp / native_ref statistics
    agg_ckpt.csv      one row per (bench, scale, mode): checkpoint/restart statistics
    sizes.csv         one row per (nccl_* bench, scale, message size): native vs dmtcp

Stdlib only, so it runs on a login node without any module.
"""

import csv
import os
import statistics
import sys
from collections import defaultdict

HIGHER_IS_BETTER = {"pt_steps_per_s", "MBps_at_max", "is_mops"}
BW_BENCHES = {"nccl_bw", "nccl_bibw"}


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def stats(values):
    v = [x for x in values if x is not None]
    if not v:
        return None
    return {
        "n": len(v),
        "mean": statistics.mean(v),
        "sd": statistics.stdev(v) if len(v) > 1 else 0.0,
        "min": min(v),
        "max": max(v),
    }


def fmt(s, digits=3):
    if s is None:
        return "-"
    if s["n"] == 1:
        return f"{s['mean']:.{digits}f}"
    return f"{s['mean']:.{digits}f} ± {s['sd']:.{digits}f}"


def overhead_pct(native, other, higher_is_better):
    """Slowdown of `other` relative to `native`, in percent (positive = slower)."""
    if native is None or other is None or native["mean"] == 0 or other["mean"] == 0:
        return None
    if higher_is_better:
        return (native["mean"] / other["mean"] - 1.0) * 100.0
    return (other["mean"] / native["mean"] - 1.0) * 100.0


def pct(x):
    return "-" if x is None else f"{x:+.2f}%"


def scale_key(r):
    return (int(r["nodes"]), int(r["gpus_per_node"]))


def scale_str(k):
    return f"{k[0]}x{k[1]} ({k[0] * k[1]} GPUs)"


def main():
    cdir = sys.argv[1] if len(sys.argv) > 1 else "."
    with open(os.path.join(cdir, "results.csv")) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit("no rows in results.csv")

    ok = [r for r in rows if r["status"] == "ok"]
    groups = defaultdict(list)
    for r in ok:
        groups[(r["bench"], scale_key(r), r["mode"])].append(r)
    benches = sorted({r["bench"] for r in rows})
    scales = sorted({scale_key(r) for r in rows})

    out = []
    out.append(f"# Campaign {os.path.basename(os.path.abspath(cdir))}\n")
    out.append(
        f"{len(rows)} runs, {len(ok)} ok. Values are mean ± sample std over successful "
        "repetitions. Overhead is the slowdown relative to `native` (same NCCL settings, "
        "no DMTCP); `native_ref` uses the NCCL module defaults (AWS-OFI on CXI, P2P/SHM on) "
        "and shows the cost of the checkpoint-safe NCCL settings themselves.\n"
    )

    # --- Runtime overhead -------------------------------------------------
    out.append("## Runtime overhead (no checkpoint)\n")
    out.append(
        "| Benchmark | Scale | Metric | native | dmtcp | overhead | native_ref | "
        "wall native (s) | wall dmtcp (s) | wall overhead |"
    )
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    agg_over = []
    for b in benches:
        for s in scales:
            nat = groups.get((b, s, "native"), [])
            dm = groups.get((b, s, "dmtcp"), [])
            ref = groups.get((b, s, "native_ref"), [])
            if not (nat or dm or ref):
                continue
            mname = (nat or dm or ref)[0]["metric_name"]
            hib = mname in HIGHER_IS_BETTER
            sn, sd, sr = (stats([fnum(r["metric"]) for r in g]) for g in (nat, dm, ref))
            wn, wd = (stats([fnum(r["wall_s"]) for r in g]) for g in (nat, dm))
            ov, wov = overhead_pct(sn, sd, hib), overhead_pct(wn, wd, False)
            out.append(
                f"| {b} | {scale_str(s)} | {mname} | {fmt(sn)} | {fmt(sd)} | {pct(ov)} | "
                f"{fmt(sr)} | {fmt(wn, 2)} | {fmt(wd, 2)} | {pct(wov)} |"
            )
            agg_over.append(
                {
                    "bench": b,
                    "nodes": s[0],
                    "gpus_per_node": s[1],
                    "nranks": s[0] * s[1],
                    "metric": mname,
                    "native_mean": sn and sn["mean"],
                    "native_sd": sn and sn["sd"],
                    "dmtcp_mean": sd and sd["mean"],
                    "dmtcp_sd": sd and sd["sd"],
                    "native_ref_mean": sr and sr["mean"],
                    "overhead_pct": ov,
                    "wall_native_mean": wn and wn["mean"],
                    "wall_dmtcp_mean": wd and wd["mean"],
                    "wall_overhead_pct": wov,
                }
            )
    out.append("")

    # --- Checkpoint / restart ---------------------------------------------
    out.append("## Checkpoint and restart\n")
    out.append(
        "`ckpt` = from `dmtcp_command -c` until the coordinator writes the restart script, "
        "i.e. every image is on disk (NCCL drain + teardown, CUDA checkpoint, image write; "
        "DMTCP 4.2's `-bc` does not block). `drain` = NCCL plugin share (max over ranks). "
        "`resume rebuild` = communicator re-creation after a checkpoint the job survives. "
        "`restart latency` = `dmtcp_restart` until every rank has rebuilt its communicators; "
        "`restart total` = until the job completes.\n"
    )
    out.append(
        "| Benchmark | Scale | Mode | success | ckpt (s) | drain (s) | image total (GiB) | "
        "image/rank (MiB) | write (MiB/s) | resume rebuild (s) | restart latency (s) | "
        "restart rebuild (s) | restart total (s) |"
    )
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    agg_ckpt = []
    for b in benches:
        for s in scales:
            for m in ("ckpt_resume", "ckpt_restart"):
                allruns = [
                    r for r in rows if r["bench"] == b and scale_key(r) == s and r["mode"] == m
                ]
                if not allruns:
                    continue
                g = groups.get((b, s, m), [])
                col = {
                    k: stats([fnum(r[k]) for r in g])
                    for k in (
                        "ckpt_s",
                        "nccl_drain_s",
                        "image_bytes",
                        "write_MBps",
                        "resume_rebuild_s",
                        "restart_latency_s",
                        "restart_rebuild_s",
                        "restart_total_s",
                    )
                }
                nr = s[0] * s[1]
                img = col["image_bytes"]
                img_gib = img and {k: (v / 2**30 if k != "n" else v) for k, v in img.items()}
                img_rank = img and {k: (v / 2**20 / nr if k != "n" else v) for k, v in img.items()}
                out.append(
                    f"| {b} | {scale_str(s)} | {m} | {len(g)}/{len(allruns)} | "
                    f"{fmt(col['ckpt_s'], 2)} | {fmt(col['nccl_drain_s'])} | {fmt(img_gib, 2)} | "
                    f"{fmt(img_rank, 0)} | {fmt(col['write_MBps'], 0)} | "
                    f"{fmt(col['resume_rebuild_s'])} | {fmt(col['restart_latency_s'], 2)} | "
                    f"{fmt(col['restart_rebuild_s'])} | {fmt(col['restart_total_s'], 2)} |"
                )
                row = {
                    "bench": b,
                    "nodes": s[0],
                    "gpus_per_node": s[1],
                    "nranks": nr,
                    "mode": m,
                    "runs": len(allruns),
                    "ok": len(g),
                }
                for k, v in col.items():
                    row[k + "_mean"] = v and v["mean"]
                    row[k + "_sd"] = v and v["sd"]
                agg_ckpt.append(row)
    out.append("")

    # --- nccl_bench per message size ---------------------------------------------
    sizes = defaultdict(lambda: defaultdict(list))  # (bench, scale, mode) -> size -> [values]
    for r in ok:
        if not r["bench"].startswith("nccl_") or r["mode"] not in ("native", "dmtcp", "native_ref"):
            continue
        path = os.path.join(r["run_dir"], "sizes.csv")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) == 2 and fnum(parts[1]) is not None:
                    sizes[(r["bench"], scale_key(r), r["mode"])][int(parts[0])].append(
                        float(parts[1])
                    )
    size_rows = []
    for b in [x for x in benches if x.startswith("nccl_")]:
        hib = b in BW_BENCHES
        unit = "MB/s" if hib else "us"
        for s in scales:
            nat = sizes.get((b, s, "native"))
            dm = sizes.get((b, s, "dmtcp"))
            ref = sizes.get((b, s, "native_ref"), {})
            if not nat or not dm:
                continue
            out.append(f"## {b}, {scale_str(s)} ({unit})\n")
            out.append("| Size (B) | native | dmtcp | overhead | native_ref |")
            out.append("|---|---|---|---|---|")
            for sz in sorted(set(nat) | set(dm)):
                sn, sd, sr = stats(nat.get(sz, [])), stats(dm.get(sz, [])), stats(ref.get(sz, []))
                ov = overhead_pct(sn, sd, hib)
                out.append(f"| {sz} | {fmt(sn, 2)} | {fmt(sd, 2)} | {pct(ov)} | {fmt(sr, 2)} |")
                size_rows.append(
                    {
                        "bench": b,
                        "nodes": s[0],
                        "gpus_per_node": s[1],
                        "nranks": s[0] * s[1],
                        "size": sz,
                        "unit": unit,
                        "native_mean": sn and sn["mean"],
                        "dmtcp_mean": sd and sd["mean"],
                        "native_ref_mean": sr and sr["mean"],
                        "overhead_pct": ov,
                    }
                )
            out.append("")

    # --- Failures ----------------------------------------------------------
    bad = [r for r in rows if r["status"] != "ok"]
    if bad:
        out.append("## Failed runs\n")
        out.append("| Benchmark | Scale | Mode | Rep | Status | Run dir |")
        out.append("|---|---|---|---|---|---|")
        for r in bad:
            out.append(
                f"| {r['bench']} | {scale_str(scale_key(r))} | {r['mode']} | {r['rep']} | "
                f"{r['status']} | {r['run_dir']} |"
            )
        out.append("")

    with open(os.path.join(cdir, "summary.md"), "w") as f:
        f.write("\n".join(out))
    for name, data in (
        ("agg_overhead.csv", agg_over),
        ("agg_ckpt.csv", agg_ckpt),
        ("sizes.csv", size_rows),
    ):
        if data:
            with open(os.path.join(cdir, name), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
                w.writeheader()
                w.writerows(data)
    print("\n".join(out))


if __name__ == "__main__":
    main()
