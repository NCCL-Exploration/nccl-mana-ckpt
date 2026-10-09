#!/usr/bin/env python3
"""Aggregate a campaign's results.csv into report tables.

    python3 tests/report/summarize.py $SCRATCH/nccl-ckpt-report/<campaign>

Writes, next to results.csv:
    summary.md        Markdown tables (success rate, runtime overhead, checkpoint/restart,
                      nccl_bench per size, failed and skipped runs)
    agg_overhead.csv  one row per (bench, scale, system): native / system / native_ref statistics
    agg_ckpt.csv      one row per (bench, scale, mode): checkpoint/restart statistics
    sizes.csv         one row per (nccl_* bench, scale, system, message size): native vs system

A system is what runs under checkpointing without taking a checkpoint: `dmtcp`
(the single-process plugin, tests/report/TEST_PLAN.md) or `ns` (netsplit,
netsplit/TEST_PLAN.md). Both compare against `native` from the same campaign.

Stdlib only, so it runs on a login node without any module.
"""

import csv
import os
import statistics
import sys
from collections import defaultdict

HIGHER_IS_BETTER = {"pt_steps_per_s", "MBps_at_max", "is_mops"}
BW_BENCHES = {"nccl_bw", "nccl_bibw"}
SYSTEMS = ("dmtcp", "ns")
CKPT_MODES = ("ckpt_resume", "ckpt_restart", "ns_resume", "ns_restart")
MODE_ORDER = ("native", "native_ref", "dmtcp", "ns") + CKPT_MODES


def outcome(status):
    """ok, hang (a phase limit passed; run_bench.sh killed the job) or fail."""
    if status == "ok":
        return "ok"
    if status.startswith("hang_") or status == "timeout":
        return "hang"
    return "fail"


def mode_key(m):
    return (MODE_ORDER.index(m) if m in MODE_ORDER else len(MODE_ORDER), m)


def first_error(run_dir):
    try:
        with open(os.path.join(run_dir, "first_error.txt")) as f:
            return f.readline().strip().replace("|", "/")
    except OSError:
        return ""


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

    modes_present = {r["mode"] for r in rows}
    systems = [m for m in SYSTEMS if m in modes_present]
    out = []
    out.append(f"# Campaign {os.path.basename(os.path.abspath(cdir))}\n")
    intro = (
        f"{len(rows)} runs, {len(ok)} ok. Values are mean ± sample std over successful "
        "repetitions. Overhead is the slowdown relative to `native`."
    )
    if "dmtcp" in modes_present:
        intro += (
            " `dmtcp` runs the single-process plugin under the same NCCL settings as "
            "`native`; `native_ref` uses the NCCL module defaults (AWS-OFI on CXI, P2P/SHM "
            "on) and shows the cost of the checkpoint-safe NCCL settings themselves."
        )
    if "ns" in modes_present:
        intro += (
            " `ns` runs netsplit (NCCL's network back end in a lower half that is never "
            "checkpointed, host-staged transfers) with no checkpoint; `native` is "
            "production NCCL with the same NCCL settings (the run's `nccl_profile` and "
            "`net_profile`)."
        )
    out.append(intro + "\n")

    # --- Success rate --------------------------------------------------------
    out.append("## Success rate\n")
    out.append(
        "`hang` = a phase limit passed (startup, checkpoint, progress after a resume, "
        "restart, completion, or the hard cap) and the job was killed; failed and hung "
        "runs are listed at the end.\n"
    )
    out.append("| Benchmark | Scale | Mode | ok | hang | fail |")
    out.append("|---|---|---|---|---|---|")
    tally = defaultdict(lambda: defaultdict(int))
    for r in rows:
        tally[(r["bench"], scale_key(r), r["mode"])][outcome(r["status"])] += 1
    for key in sorted(tally, key=lambda k: (k[0], k[1], mode_key(k[2]))):
        t = tally[key]
        n = sum(t.values())
        out.append(
            f"| {key[0]} | {scale_str(key[1])} | {key[2]} | {t['ok']}/{n} | {t['hang']} | "
            f"{t['fail']} |"
        )
    out.append("")

    # --- Runtime overhead -------------------------------------------------
    agg_over = []
    for system in systems:
        out.append(f"## Runtime overhead: {system} (no checkpoint)\n")
        out.append(
            f"| Benchmark | Scale | Metric | native | {system} | overhead | native_ref | "
            f"wall native (s) | wall {system} (s) | wall overhead |"
        )
        out.append("|---|---|---|---|---|---|---|---|---|---|")
        for b in benches:
            for s in scales:
                nat = groups.get((b, s, "native"), [])
                dm = groups.get((b, s, system), [])
                ref = groups.get((b, s, "native_ref"), [])
                if not dm:
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
                row = {
                    "bench": b,
                    "nodes": s[0],
                    "gpus_per_node": s[1],
                    "nranks": s[0] * s[1],
                    "metric": mname,
                    "native_mean": sn and sn["mean"],
                    "native_sd": sn and sn["sd"],
                    f"{system}_mean": sd and sd["mean"],
                    f"{system}_sd": sd and sd["sd"],
                    "native_ref_mean": sr and sr["mean"],
                    "overhead_pct": ov,
                    "wall_native_mean": wn and wn["mean"],
                    f"wall_{system}_mean": wd and wd["mean"],
                    "wall_overhead_pct": wov,
                }
                if len(systems) > 1:
                    row["system"] = system
                agg_over.append(row)
        out.append("")

    # --- Checkpoint / restart ---------------------------------------------
    out.append("## Checkpoint and restart\n")
    out.append(
        "`ckpt` = from `dmtcp_command -c` until the coordinator writes the restart script, "
        "i.e. every image is on disk (NCCL drain + teardown, CUDA checkpoint, image write; "
        "DMTCP 4.2's `-bc` does not block). `drain` = NCCL plugin share (max over ranks); "
        "`cuda+write` = ckpt − drain, per run. "
        "`resume rebuild` = communicator re-creation after a checkpoint the job survives. "
        "`restart latency` = `dmtcp_restart` (`nccl_restart` for ns_restart) until every rank "
        "has rebuilt its communicators; "
        "`restart total` = until the job completes.\n"
    )
    out.append(
        "| Benchmark | Scale | Mode | success | ckpt (s) | drain (s) | cuda+write (s) | "
        "image total (GiB) | image/rank (MiB) | write (MiB/s) | resume rebuild (s) | "
        "restart latency (s) | restart rebuild (s) | restart total (s) |"
    )
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    agg_ckpt = []
    for b in benches:
        for s in scales:
            for m in CKPT_MODES:
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
                col["cuda_write_s"] = stats(
                    [
                        fnum(r["ckpt_s"]) - fnum(r["nccl_drain_s"])
                        for r in g
                        if fnum(r["ckpt_s"]) is not None and fnum(r["nccl_drain_s"]) is not None
                    ]
                )
                nr = s[0] * s[1]
                img = col["image_bytes"]
                img_gib = img and {k: (v / 2**30 if k != "n" else v) for k, v in img.items()}
                img_rank = img and {k: (v / 2**20 / nr if k != "n" else v) for k, v in img.items()}
                out.append(
                    f"| {b} | {scale_str(s)} | {m} | {len(g)}/{len(allruns)} | "
                    f"{fmt(col['ckpt_s'], 2)} | {fmt(col['nccl_drain_s'])} | "
                    f"{fmt(col['cuda_write_s'], 2)} | {fmt(img_gib, 2)} | "
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
        if not r["bench"].startswith("nccl_") or r["mode"] not in ("native", "native_ref") + SYSTEMS:
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
            for system in systems:
                nat = sizes.get((b, s, "native"))
                dm = sizes.get((b, s, system))
                ref = sizes.get((b, s, "native_ref"), {})
                if not nat or not dm:
                    continue
                title = f"{b}, {scale_str(s)}" + (f", {system}" if len(systems) > 1 else "")
                out.append(f"## {title} ({unit})\n")
                out.append(f"| Size (B) | native | {system} | overhead | native_ref |")
                out.append("|---|---|---|---|---|")
                for sz in sorted(set(nat) | set(dm)):
                    sn, sd = stats(nat.get(sz, [])), stats(dm.get(sz, []))
                    sr = stats(ref.get(sz, []))
                    ov = overhead_pct(sn, sd, hib)
                    out.append(
                        f"| {sz} | {fmt(sn, 2)} | {fmt(sd, 2)} | {pct(ov)} | {fmt(sr, 2)} |"
                    )
                    row = {
                        "bench": b,
                        "nodes": s[0],
                        "gpus_per_node": s[1],
                        "nranks": s[0] * s[1],
                        "size": sz,
                        "unit": unit,
                        "native_mean": sn and sn["mean"],
                        f"{system}_mean": sd and sd["mean"],
                        "native_ref_mean": sr and sr["mean"],
                        "overhead_pct": ov,
                    }
                    if len(systems) > 1:
                        row["system"] = system
                    size_rows.append(row)
                out.append("")

    # --- Failures ----------------------------------------------------------
    bad = [r for r in rows if r["status"] != "ok"]
    if bad:
        out.append("## Failed and hung runs\n")
        out.append("| Benchmark | Scale | Mode | Rep | Status | First error | Run dir |")
        out.append("|---|---|---|---|---|---|---|")
        for r in bad:
            out.append(
                f"| {r['bench']} | {scale_str(scale_key(r))} | {r['mode']} | {r['rep']} | "
                f"{r['status']} | {first_error(r['run_dir']) or '-'} | {r['run_dir']} |"
            )
        out.append("")

    # Runs campaign.sbatch skipped because the job could have ended mid-run.
    skipped_path = os.path.join(cdir, "skipped.txt")
    if os.path.exists(skipped_path):
        with open(skipped_path) as f:
            skipped = [line.strip() for line in f if line.strip()]
        if skipped:
            out.append("## Skipped runs\n")
            out.extend(f"- {line}" for line in skipped)
            out.append("")

    with open(os.path.join(cdir, "summary.md"), "w") as f:
        f.write("\n".join(out))
    for name, data in (
        ("agg_overhead.csv", agg_over),
        ("agg_ckpt.csv", agg_ckpt),
        ("sizes.csv", size_rows),
    ):
        if data:
            # Rows of different systems carry different columns.
            fields = list(dict.fromkeys(k for row in data for k in row))
            with open(os.path.join(cdir, name), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields, restval="")
                w.writeheader()
                w.writerows(data)
    print("\n".join(out))


if __name__ == "__main__":
    main()
