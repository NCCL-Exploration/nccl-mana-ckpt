#!/usr/bin/env python3
"""Gemma LLM Distributed Inference & Auto-Regressive Token Generation Benchmark.

Benchmarks multi-GPU PyTorch distributed inference on Gemma architecture under
DMTCP / ShadowNCCL transparent checkpointing.
"""

import argparse
import math
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization used in Gemma."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * (1.0 + self.weight)


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) used in Gemma."""

    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.base = base

    def forward(self, x: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        device = x.device
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, device=device).float() / self.dim)
        )
        t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class SwiGLUMLP(nn.Module):
    """SwiGLU Feed-Forward Network used in Gemma."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class GemmaAttention(nn.Module):
    """Multi-Head Attention with Rotary Embeddings for Gemma Inference."""

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.rotary_emb = RotaryEmbedding(self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b_sz, seq_len, _ = x.shape
        q = self.q_proj(x).view(b_sz, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(b_sz, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(b_sz, seq_len, self.num_heads, self.head_dim)

        cos, sin = self.rotary_emb(x, seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=x.device), diagonal=1
        )
        scores = scores + mask

        attn = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn, v)
        output = output.transpose(1, 2).contiguous().view(b_sz, seq_len, -1)
        return self.o_proj(output)


class GemmaDecoderLayer(nn.Module):
    """Gemma Transformer Decoder Block."""

    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_size)
        self.self_attn = GemmaAttention(hidden_size, num_heads)
        self.post_attention_layernorm = RMSNorm(hidden_size)
        self.mlp = SwiGLUMLP(hidden_size, intermediate_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = residual + self.self_attn(self.input_layernorm(x))
        residual = x
        x = residual + self.mlp(self.post_attention_layernorm(x))
        return x


class GemmaForCausalLM(nn.Module):
    """Gemma Language Model for Inference & Generation."""

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 2048,
        num_layers: int = 4,
        num_heads: int = 8,
        intermediate_size: int = 8192,
    ):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList(
            [
                GemmaDecoderLayer(hidden_size, num_heads, intermediate_size)
                for _ in range(num_layers)
            ]
        )
        self.norm = RMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 30) -> torch.Tensor:
        generated = input_ids.clone()
        for _ in range(max_new_tokens):
            logits = self.forward(generated)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            generated = torch.cat((generated, next_token), dim=1)
        return generated


def parse_args():
    parser = argparse.ArgumentParser(
        description="PyTorch Gemma LLM Distributed Inference Benchmark Script"
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=5,
        help="Number of prompt batches to run. Use -1 for infinite continuous execution.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=30,
        help="Number of tokens to generate per prompt.",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per GPU.")
    parser.add_argument("--prompt-len", type=int, default=64, help="Prompt sequence length.")
    parser.add_argument("--hidden-size", type=int, default=2048, help="Hidden dimension size.")
    parser.add_argument("--num-layers", type=int, default=4, help="Number of Transformer layers.")
    parser.add_argument("--num-heads", type=int, default=8, help="Number of attention heads.")
    parser.add_argument("--vocab-size", type=int, default=32000, help="Vocabulary size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def setup_distributed():
    if "WORLD_SIZE" not in os.environ or "RANK" not in os.environ:
        print(
            "Error: WORLD_SIZE and RANK environment variables not set. "
            "Please launch using torchrun or srun.",
            file=sys.stderr,
        )
        sys.exit(1)

    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    return rank, local_rank, world_size


def set_seed(seed: int, rank: int):
    worker_seed = seed + rank
    torch.manual_seed(worker_seed)
    torch.cuda.manual_seed_all(worker_seed)


def main():
    args = parse_args()
    rank, local_rank, world_size = setup_distributed()
    set_seed(args.seed, rank)

    if rank == 0:
        print(
            f"[Gemma Inference] Launching run across {world_size} GPUs.",
            flush=True,
        )
        print(
            f"[Gemma Config] layers={args.num_layers}, hidden={args.hidden_size}, "
            f"batch={args.batch_size}, prompt_len={args.prompt_len}, "
            f"max_new_tokens={args.max_new_tokens}",
            flush=True,
        )

    model = GemmaForCausalLM(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
    ).to(local_rank)
    model.eval()

    torch.cuda.synchronize(local_rank)
    prompt_idx = 0

    while args.num_prompts == -1 or prompt_idx < args.num_prompts:
        prompt_start_time = time.time()

        prompt_tokens = torch.randint(
            0, args.vocab_size, (args.batch_size, args.prompt_len), device=local_rank
        )

        with torch.no_grad():
            output_tokens = model.generate(prompt_tokens, max_new_tokens=args.max_new_tokens)

        torch.cuda.synchronize(local_rank)
        prompt_duration = time.time() - prompt_start_time

        if rank == 0:
            total_generated_tokens = args.batch_size * args.max_new_tokens * world_size
            tokens_per_sec = total_generated_tokens / prompt_duration
            print(
                f"[Prompt {prompt_idx}] Duration: {prompt_duration:.4f}s | "
                f"Output Shape: {output_tokens.shape} | "
                f"Generation Speed: {tokens_per_sec:.1f} tokens/s",
                flush=True,
            )

        prompt_idx += 1

    if rank == 0:
        print("[Gemma Inference] Completed successfully.", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
