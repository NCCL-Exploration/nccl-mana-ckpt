#!/usr/bin/env python3
import os
import sys
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.models import resnet18 

def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch DDP Benchmark / Checkpoint Test Script")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of epochs to run. Use -1 for infinite execution.")
    parser.add_argument("--steps-per-epoch", type=int, default=10,
                        help="Number of steps/iterations per epoch.")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size per GPU.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Learning rate.")
    return parser.parse_args()

def setup_distributed():
    # PyTorch distributed initialization
    # Typically launched with torchrun or environment variables set manually
    if "WORLD_SIZE" not in os.environ or "RANK" not in os.environ:
        print("Error: World size and rank environment variables not set. Please launch using torchrun or MPI.", file=sys.stderr)
        sys.exit(1)
        
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    # We use the NCCL backend for multi-GPU training
    dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(local_rank)
    
    return rank, local_rank, world_size

def set_seed(seed, rank):
    # Ensure reproducibility, offsetting seed by rank so workers get different data
    worker_seed = seed + rank
    torch.manual_seed(worker_seed)
    torch.cuda.manual_seed_all(worker_seed)

def main():
    args = parse_args()
    
    rank, local_rank, world_size = setup_distributed()
    set_seed(args.seed, rank)
    
    if rank == 0:
        print(f"Initializing DDP run with {world_size} GPUs.", flush=True)
        print(f"Configuration: epochs={args.epochs}, steps-per-epoch={args.steps_per_epoch}, batch-size={args.batch_size}, seed={args.seed}", flush=True)

    # Initialize a standard ResNet18 model
    # We use random weights (weights=None) since this is a test benchmark
    model = resnet18(weights=None)
    model = model.to(local_rank)
    model = DDP(model, device_ids=[local_rank])
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    
    # Warmup / check device activity
    torch.cuda.synchronize(local_rank)
    
    epoch = 0
    total_step = 0
    
    # Keep running if epochs is -1, or if we have not reached the epoch limit
    while args.epochs == -1 or epoch < args.epochs:
        model.train()
        epoch_start_time = time.time()
        
        for step in range(args.steps_per_epoch):
            step_start_time = time.time()
            
            # Generate synthetic batch on-the-fly to avoid disk I/O bottlenecks
            # ResNet18 expects images of shape (B, 3, 224, 224) and targets in range [0, 1000)
            inputs = torch.randn(args.batch_size, 3, 224, 224, device=local_rank)
            targets = torch.randint(0, 1000, (args.batch_size,), device=local_rank)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            # Synchronize to ensure computation and communication (all_reduce gradients) are complete
            torch.cuda.synchronize(local_rank)
            step_duration = time.time() - step_start_time
            
            if rank == 0:
                print(f"[Epoch {epoch} | Step {step} (Total Step {total_step})] Loss: {loss.item():.4f} | Time: {step_duration:.4f}s", flush=True)
            
            total_step += 1
            
        epoch_duration = time.time() - epoch_start_time
        if rank == 0:
            print(f"[Epoch {epoch} Completed] Duration: {epoch_duration:.4f}s | Average Step Time: {epoch_duration / args.steps_per_epoch:.4f}s", flush=True)
            total_images = args.steps_per_epoch * args.batch_size * world_size
            mops_per_image = 3600.0 # ~3.6 GFLOPs for ResNet18 fwd+bwd
            total_mops = total_images * mops_per_image
            mops_per_sec = total_mops / epoch_duration
            print("\n PyTorch Benchmark Epoch Completed")
            print(f" Size            =             {total_images:12d}")
            print(f" Iterations      =             {args.steps_per_epoch:12d}")
            print(f" Time in seconds =             {epoch_duration:12.2f}")
            print(f" Total processes =             {world_size:12d}")
            print(f" Mop/s total     =             {mops_per_sec:12.2f}")
            print(f" Mop/s/process   =             {mops_per_sec/world_size:12.2f}\n", flush=True)
            
        epoch += 1

    if rank == 0:
        print("Training finished successfully.", flush=True)
        
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
