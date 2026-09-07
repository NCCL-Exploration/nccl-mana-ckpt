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
    args = parser.parse_args()

    if "WORLD_SIZE" not in os.environ or "RANK" not in os.environ:
        print("Error: set WORLD_SIZE and RANK (see the Makefile target).", file=sys.stderr)
        sys.exit(1)

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

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

    step = 0
    while args.steps < 0 or step < args.steps:
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
        time.sleep(0.2)

    print(f"PASS pytorch_minimal: rank {rank}, {step} steps", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
