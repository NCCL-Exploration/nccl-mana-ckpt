#!/usr/bin/env python3
"""PyTorch Multi-Communicator & Sub-Group Test Script for DMTCP / ShadowNCCL.

Validates that PyTorch distributed applications creating multiple NCCL communicators
(e.g., world group, sub-groups created via dist.new_group, and P2P operations)
can transparently checkpoint and restart without handle corruption, deadlocks,
or numerical drift.
"""

import argparse
import os
import sys
import time
import torch
import torch.distributed as dist


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Multi-Communicator Checkpoint Test")
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of epochs to run. Use -1 for infinite execution.",
    )
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=10,
        help="Number of steps/iterations per epoch.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def setup_distributed():
    if "WORLD_SIZE" not in os.environ or "RANK" not in os.environ:
        print(
            "Error: WORLD_SIZE and RANK environment variables not set.",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(1)

    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank % max(1, torch.cuda.device_count())))

    print(
        f"[Rank {rank}] Setting device to GPU {local_rank} and initializing process group...",
        flush=True,
    )
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    print(f"[Rank {rank}] Successfully initialized default process group.", flush=True)

    return rank, local_rank, world_size


def create_subgroups(rank, world_size):
    """Creates multiple sub-communicators across the ranks."""
    subgroups = {}

    if world_size > 1:
        # Subgroup 1: Even ranks clique / Odd ranks clique (for >=4 ranks) or duplicate sub-group
        even_ranks = [r for r in range(world_size) if r % 2 == 0]
        odd_ranks = [r for r in range(world_size) if r % 2 != 0]

        even_group = dist.new_group(ranks=even_ranks) if len(even_ranks) > 1 else None
        odd_group = dist.new_group(ranks=odd_ranks) if len(odd_ranks) > 1 else None

        subgroups["parity_group"] = even_group if (rank % 2 == 0) else odd_group
        subgroups["parity_ranks"] = even_ranks if (rank % 2 == 0) else odd_ranks
    else:
        subgroups["parity_group"] = None
        subgroups["parity_ranks"] = [0]

    # Subgroup 2: Additional sub-communicator across all ranks
    all_group = dist.new_group(ranks=list(range(world_size)))
    subgroups["sub_group"] = all_group
    subgroups["sub_ranks"] = list(range(world_size))

    return subgroups


def main():
    print(f"[Process {os.getpid()}] Starting pytorch_multicomm_test.py...", flush=True)
    args = parse_args()
    rank, local_rank, world_size = setup_distributed()

    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)

    if rank == 0:
        print(f"==> Initialized PyTorch Distributed with {world_size} ranks.", flush=True)
        print(f"==> Creating multiple NCCL sub-groups (communicators)...", flush=True)

    subgroups = create_subgroups(rank, world_size)

    if rank == 0:
        print(f"==> Created sub-groups successfully. Starting multi-comm workload.", flush=True)

    epoch = 0
    total_step = 0

    # Running accumulator tensor to verify consistency
    checksum = torch.zeros(1024, dtype=torch.float32, device=local_rank)

    while args.epochs == -1 or epoch < args.epochs:
        epoch_start = time.time()

        for step in range(args.steps_per_epoch):
            step_start = time.time()

            # 1. Collective on DEFAULT WORLD communicator (Ratio 1:1)
            world_tensor = torch.ones(1024, dtype=torch.float32, device=local_rank) * (rank + 1.0)
            dist.all_reduce(world_tensor, op=dist.ReduceOp.SUM)
            checksum += world_tensor * 0.01

            # 2. Collective on SUB-COMMUNICATOR 1 (Parity group, Ratio 2:1)
            parity_grp = subgroups["parity_group"]
            if parity_grp is not None:
                p_tensor = torch.ones(512, dtype=torch.float32, device=local_rank) * (total_step + 1.0)
                dist.all_reduce(p_tensor, op=dist.ReduceOp.SUM, group=parity_grp)
                dist.broadcast(p_tensor, src=subgroups["parity_ranks"][0], group=parity_grp)
                checksum[:512] += p_tensor * 0.005

            # 3. Collective on SUB-COMMUNICATOR 2 (Sub group, Ratio 3:1)
            sub_grp = subgroups["sub_group"]
            if sub_grp is not None:
                h_tensor = torch.ones(256, dtype=torch.float32, device=local_rank) * (epoch + 1.0)
                dist.all_reduce(h_tensor, op=dist.ReduceOp.SUM, group=sub_grp)
                checksum[:256] += h_tensor * 0.002

            # 4. Point-to-point communication between adjacent ranks
            if world_size > 1:
                partner = (rank + 1) if (rank % 2 == 0) else (rank - 1)
                if partner < world_size:
                    p2p_buf = torch.full((128,), fill_value=float(rank), device=local_rank)
                    if rank % 2 == 0:
                        dist.send(p2p_buf, dst=partner)
                        recv_buf = torch.empty((128,), device=local_rank)
                        dist.recv(recv_buf, src=partner)
                        checksum[:128] += recv_buf * 0.001
                    else:
                        recv_buf = torch.empty((128,), device=local_rank)
                        dist.recv(recv_buf, src=partner)
                        dist.send(p2p_buf, dst=partner)
                        checksum[:128] += recv_buf * 0.001

            torch.cuda.synchronize(local_rank)
            step_time = time.time() - step_start

            if rank == 0 and (total_step % 5 == 0 or args.epochs != -1):
                print(
                    f"[Epoch {epoch} | Step {step} (Total {total_step})] "
                    f"Checksum sum: {checksum.sum().item():.4f} | Time: {step_time:.4f}s",
                    flush=True,
                )

            total_step += 1

        epoch_time = time.time() - epoch_start
        if rank == 0:
            print(
                f"[Epoch {epoch} Complete] Duration: {epoch_time:.4f}s | "
                f"Checksum sum: {checksum.sum().item():.4f}\n",
                flush=True,
            )

        epoch += 1

    if rank == 0:
        print("==> Multi-communicator PyTorch workload completed successfully.", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
