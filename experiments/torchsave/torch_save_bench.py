#!/usr/bin/env python3
"""How long an application-level checkpoint with torch.save takes.

The baseline the transparent checkpoint is compared against: the same model
as tests/multi/pytorch_minimal_test.py (Linear(w, w), ReLU, Linear(w, w),
ReLU, Linear(w, 10), SGD with momentum), saved and loaded the way a training
script would, on the same nodes, GPUs and filesystem as the DMTCP images.

One process per GPU, launched by srun (experiments/lib/run_bench.sh, bench
"torchsave"); RANK, WORLD_SIZE and LOCAL_RANK come from launch_rank.sh and
PT_STORE_FILE names a FileStore on a shared filesystem. No DMTCP.

For every width and policy, TS_REPS times:
  save  barrier, then on each saving rank:
          d2h    copy the state dict (model, and optimizer with --optim) to
                 host memory, cuda synchronized
          write  torch.save of the host copy to <dir>/<width>/rank<r>.pt
          fsync  os.fsync, so the data has left the page cache
        then a barrier; job_save is the barrier-to-barrier time, i.e. how
        long the job is stopped for the checkpoint.
  load  the file's cached pages are dropped (posix_fadvise DONTNEED) so the
        read goes to the filesystem, barrier, torch.load(map_location=GPU),
        load_state_dict into the model (and optimizer), cuda synchronized,
        barrier; the restored parameters are checked against the saved ones.
Policies: "all" -- every rank saves its replica (one file per GPU, like
DMTCP's one image per rank); "rank0" -- only rank 0 saves, the usual DDP
practice since every replica is identical.

Each phase's time is the maximum over ranks (the job waits for the slowest).
Rank 0 writes one CSV row per (width, policy, rep) to --csv and prints
"PASS torch_save_bench" when every check passed.
"""

import argparse
import csv
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.nn as nn


def build(width, device):
    torch.manual_seed(1234)
    model = nn.Sequential(
        nn.Linear(width, width),
        nn.ReLU(),
        nn.Linear(width, width),
        nn.ReLU(),
        nn.Linear(width, 10),
    ).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # One step, so the optimizer has its momentum buffers (as mid-training).
    x = torch.randn(64, width, device=device)
    y = torch.randint(0, 10, (64,), device=device)
    opt.zero_grad()
    nn.CrossEntropyLoss()(model(x), y).backward()
    opt.step()
    torch.cuda.synchronize(device)
    return model, opt


def tensor_bytes(obj):
    if torch.is_tensor(obj):
        return obj.numel() * obj.element_size()
    if isinstance(obj, dict):
        return sum(tensor_bytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(tensor_bytes(v) for v in obj)
    return 0


def to_host(obj):
    if torch.is_tensor(obj):
        return obj.detach().to("cpu")
    if isinstance(obj, dict):
        return {k: to_host(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_host(v) for v in obj)
    return obj


def checksum(model):
    return float(sum(p.detach().double().sum().item() for p in model.parameters()))


def drop_cache(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="where the checkpoint files go")
    ap.add_argument("--csv", required=True, help="rank 0 writes the results here")
    ap.add_argument("--widths", type=int, nargs="+", default=[4096])
    ap.add_argument("--policies", nargs="+", default=["all", "rank0"])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--optim", type=int, default=1, help="1: save the optimizer state too")
    ap.add_argument("--keep", action="store_true", help="keep the files")
    args = ap.parse_args()

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    store = os.environ.get("PT_STORE_FILE")
    if store:
        dist.init_process_group(
            "nccl", init_method="file://" + os.path.abspath(store), rank=rank, world_size=world
        )
    else:
        dist.init_process_group("nccl", init_method="env://")
    if rank == 0:
        print(f"# torch_save_bench: {world} ranks", flush=True)

    def barrier():
        dist.barrier(device_ids=[local])

    def max_over_ranks(values):
        t = torch.tensor(values, dtype=torch.float64, device=dev)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return t.tolist()

    def sum_over_ranks(value):
        t = torch.tensor([value], dtype=torch.float64, device=dev)
        dist.all_reduce(t)
        return t.item()

    fields = [
        "width", "params", "policy", "rep", "optim", "savers", "payload_bytes_per_rank",
        "file_bytes_total", "d2h_s", "write_s", "fsync_s", "save_s", "job_save_s",
        "load_s", "job_load_s", "write_MBps_per_rank", "ok",
    ]
    out = None
    if rank == 0:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
        out = open(args.csv, "w", newline="")
        w = csv.DictWriter(out, fieldnames=fields, lineterminator="\n")
        w.writeheader()

    all_ok = True
    for width in args.widths:
        model, opt = build(width, dev)
        params = sum(p.numel() for p in model.parameters())
        ref = checksum(model)
        wdir = os.path.join(args.dir, f"w{width}")
        os.makedirs(wdir, exist_ok=True)
        path = os.path.join(wdir, f"rank{rank}.pt")
        for policy in args.policies:
            saver = policy == "all" or rank == 0
            for rep in range(1, args.reps + 1):
                # --- save ---------------------------------------------------
                barrier()
                j0 = time.perf_counter()
                d2h = write = fsync = 0.0
                nbytes = fbytes = 0
                if saver:
                    t0 = time.perf_counter()
                    state = {"model": model.state_dict()}
                    if args.optim:
                        state["optimizer"] = opt.state_dict()
                    nbytes = tensor_bytes(state)
                    host = to_host(state)
                    torch.cuda.synchronize(dev)
                    t1 = time.perf_counter()
                    with open(path, "wb") as f:
                        torch.save(host, f)
                        f.flush()
                        t2 = time.perf_counter()
                        os.fsync(f.fileno())
                    t3 = time.perf_counter()
                    d2h, write, fsync = t1 - t0, t2 - t1, t3 - t2
                    fbytes = os.path.getsize(path)
                    del host, state
                barrier()
                job_save = time.perf_counter() - j0

                # --- load ---------------------------------------------------
                if saver:
                    drop_cache(path)
                    # Clobber the parameters, so only a real restore passes the check.
                    with torch.no_grad():
                        for p in model.parameters():
                            p.zero_()
                barrier()
                j1 = time.perf_counter()
                load = 0.0
                good = 1.0
                if saver:
                    t4 = time.perf_counter()
                    state = torch.load(path, map_location=dev, weights_only=True)
                    model.load_state_dict(state["model"])
                    if args.optim:
                        opt.load_state_dict(state["optimizer"])
                    torch.cuda.synchronize(dev)
                    load = time.perf_counter() - t4
                    del state
                    good = 1.0 if abs(checksum(model) - ref) <= 1e-6 * max(1.0, abs(ref)) else 0.0
                barrier()
                job_load = time.perf_counter() - j1

                m = max_over_ranks([d2h, write, fsync, d2h + write + fsync, job_save, load,
                                    job_load, float(nbytes), 1.0 - good])
                ftot = sum_over_ranks(float(fbytes))
                ok = m[8] == 0.0
                all_ok = all_ok and ok
                if saver and not args.keep and os.path.exists(path):
                    os.remove(path)
                if rank == 0:
                    savers = world if policy == "all" else 1
                    row = dict(
                        width=width, params=params, policy=policy, rep=rep, optim=args.optim,
                        savers=savers, payload_bytes_per_rank=int(m[7]),
                        file_bytes_total=int(ftot), d2h_s=f"{m[0]:.6f}", write_s=f"{m[1]:.6f}",
                        fsync_s=f"{m[2]:.6f}", save_s=f"{m[3]:.6f}", job_save_s=f"{m[4]:.6f}",
                        load_s=f"{m[5]:.6f}", job_load_s=f"{m[6]:.6f}",
                        write_MBps_per_rank=f"{(m[7] / 2**20) / m[3]:.1f}" if m[3] > 0 else "",
                        ok=int(ok),
                    )
                    w.writerow(row)
                    out.flush()
                    print(
                        f"[ts] w={width} {policy} rep {rep}: job_save {m[4]:.3f}s "
                        f"(d2h {m[0]:.3f} write {m[1]:.3f} fsync {m[2]:.3f}) "
                        f"job_load {m[6]:.3f}s {int(m[7]) / 2**20:.0f} MiB/rank"
                        f"{'' if ok else ' CHECKSUM MISMATCH'}",
                        flush=True,
                    )
        del model, opt
        torch.cuda.empty_cache()

    if rank == 0:
        out.close()
        print("PASS torch_save_bench" if all_ok else "FAIL torch_save_bench", flush=True)
    dist.destroy_process_group()
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
