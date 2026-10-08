"""Minimal PyTorch DDP program for checkpoint/restart testing.

Deliberately smaller than pytorch_ddp_test.py, and conv-free: DMTCP cannot
dlopen cudnn (`Invalid handle. Cannot load symbol cudnnGetVersion`, reproducible
with no plugins loaded), so ResNet18 aborts before training starts. Linear layers
exercise the same NCCL path -- dist.init_process_group("nccl") -> DDP -> bucketed
ncclAllReduce in backward, all issued by ProcessGroupNCCL, which wraps
communicator creation and coalesced collectives in ncclGroupStart/ncclGroupEnd.

Run via `make pytorch_minimal` in this directory, which supplies the required
environment. With one GPU, both ranks share it through
NCCL_MULTI_RANK_GPU_ENABLE=1 (NCCL >= 2.30).
"""

import argparse
import math
import os
import sys
import time

# Must precede `import torch`: DMTCP deadlocks in the fork that
# ctypes.util.find_library() performs during the import.  pytorch_dmtcp_env.sh
# puts this on PYTHONPATH, in which case the import below is a no-op; doing it
# here as well keeps the script runnable on its own.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "pyshim"))
import dmtcp_torch_shim  # noqa: F401,E402

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    # -1 runs forever, so a checkpoint can be triggered mid-training.
    parser.add_argument("--steps", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--report-every", type=int, default=5)
    # Pause per step, so a checkpoint can land mid-training. 0 for throughput runs.
    parser.add_argument("--sleep", type=float, default=0.2)
    # Untimed steps before throughput is measured (first-step setup, NCCL warm-up).
    parser.add_argument("--warmup", type=int, default=5)
    # Instead of --steps: time the warm-up, agree on the slowest rank's step
    # time (all_reduce MAX), and run as many steps as fill DURATION seconds.
    # Every rank computes the same count, so the collectives stay matched.
    parser.add_argument("--duration", type=float, default=0)
    args = parser.parse_args()

    if "WORLD_SIZE" not in os.environ or "RANK" not in os.environ:
        print("Error: set WORLD_SIZE and RANK (see the Makefile target).", file=sys.stderr)
        sys.exit(1)

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # A TCPStore does not survive checkpoint/restart: DMTCP restores its sockets
    # by binding their original local addresses, which are still in TIME_WAIT
    # from the checkpointed process, so every rank comes back with a dead store
    # ("Bind failed: Address already in use", then TCPStore recv/send errors from
    # ProcessGroupNCCL's HeartbeatMonitor).  A FileStore has no sockets to
    # restore.  PT_STORE_FILE must be on a filesystem shared by all ranks.
    store_file = os.environ.get("PT_STORE_FILE")
    if store_file:
        dist.init_process_group(
            backend="nccl",
            init_method="file://" + os.path.abspath(store_file),
            rank=rank,
            world_size=world_size,
        )
    else:
        dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(local_rank)
    print(f"[pt] rank {rank}/{world_size} process group ready", flush=True)

    torch.manual_seed(1234)
    model = nn.Sequential(
        nn.Linear(args.width, args.width),
        nn.ReLU(),
        nn.Linear(args.width, args.width),
        nn.ReLU(),
        nn.Linear(args.width, 10),
    ).to(local_rank)
    model = DDP(model, device_ids=[local_rank])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    print(f"[pt] rank {rank} DDP constructed", flush=True)

    warmup = max(args.warmup, 2)
    step = 0
    t_start = time.time()
    t_measure = None
    while args.steps < 0 or step < args.steps or (args.duration > 0 and t_measure is None):
        if step == 1:
            t_first = time.time()  # step 0 includes one-time setup
        if step == warmup:
            t_measure = time.time()
            if args.duration > 0:
                per_step = torch.tensor(
                    [(t_measure - t_first) / (warmup - 1)], dtype=torch.float64, device=local_rank
                )
                dist.all_reduce(per_step, op=dist.ReduceOp.MAX)
                args.steps = warmup + max(1, math.ceil(args.duration / per_step.item()))
                if rank == 0:
                    print(
                        f"[pt] duration {args.duration:.1f} s: {per_step.item():.4f} s/step "
                        f"-> {args.steps} steps",
                        flush=True,
                    )
        inputs = torch.randn(args.batch_size, args.width, device=local_rank)
        targets = torch.randint(0, 10, (args.batch_size,), device=local_rank)
        optimizer.zero_grad()
        loss = criterion(model(inputs), targets)
        loss.backward()  # -> bucketed ncclAllReduce
        optimizer.step()
        torch.cuda.synchronize(local_rank)

        # An explicit collective with a checkable result, so correctness does
        # not rest on the loss value alone.
        probe = torch.full((4,), float(rank + 1), device=local_rank)
        dist.all_reduce(probe)
        expected = float(world_size * (world_size + 1) // 2)
        if abs(probe[0].item() - expected) > 1e-6:
            print(
                f"FAIL rank {rank} step {step}: all_reduce gave "
                f"{probe[0].item()}, want {expected}",
                flush=True,
            )
            sys.exit(1)

        if step % args.report_every == 0:
            print(
                f"[pt] rank {rank} step {step} loss {loss.item():.4f} all_reduce ok",
                flush=True,
            )
        step += 1
        if args.sleep > 0:
            time.sleep(args.sleep)

    # Throughput over the steps after warm-up (all steps if the run was shorter).
    t_end = time.time()
    if t_measure is not None and step > warmup:
        elapsed, measured = t_end - t_measure, step - warmup
    else:
        elapsed, measured = t_end - t_start, step
    print(
        f"[pt] rank {rank} elapsed {elapsed:.6f} s for {measured} steps "
        f"({measured / elapsed if elapsed > 0 else 0:.3f} steps/s)",
        flush=True,
    )

    print(f"PASS pytorch_minimal: rank {rank}, {step} steps", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
