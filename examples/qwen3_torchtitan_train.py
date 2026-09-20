#!/usr/bin/env python3
"""Train a tiny TorchTitan Qwen3 model on libtt through TorchAX."""

import argparse
import math
import time
from dataclasses import dataclass
from functools import partial

import jax
import optax
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchax
import torchax.train
from torchax import interop
from torchax.interop import JittableModule
from torchtitan.models.common import CosSinRoPE, Embedding, Linear, RMSNorm
from torchtitan.models.common.config_utils import make_ffn_config, make_gqa_config
from torchtitan.models.qwen3.model import Qwen3Model, Qwen3TransformerBlock
from torchtitan.protocols.module import Module


class TorchaxCausalAttention(Module):
    """TorchTitan attention backend expressed with TorchAX-supported SDPA.

    TorchTitan's Qwen3 projections use the flattened token layout `[T, N, H]`.
    PyTorch SDPA expects heads before tokens, so this adapter transposes to
    `[N, T, H]` and back. The tiny config uses the same number of query and
    key/value heads, which keeps this path deliberately simple.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        pass

    def __init__(self, config: Config):
        super().__init__()

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        attention_masks=None,
        scale: float | None = None,
        enable_gqa: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        if attention_masks is not None:
            raise ValueError("TorchaxCausalAttention expects its built-in causal mask")
        if enable_gqa:
            raise ValueError("the tiny example requires --heads to equal the KV head count")

        query = query.transpose(0, 1)
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=0.0,
            is_causal=True,
            scale=scale,
        )
        return output.transpose(0, 1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a tiny TorchTitan Qwen3 model through TorchAX and libtt."
    )
    parser.add_argument("--backend", default="tt", help="JAX backend (tt or cpu)")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--dtype", choices=("bf16", "f32"), default="bf16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--validate-loss",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="require the fixed synthetic batch's loss to decrease",
    )
    return parser.parse_args()


def validate_args(args):
    if args.steps < 1:
        raise SystemExit("--steps must be positive")
    if args.layers < 1:
        raise SystemExit("--layers must be positive")
    if args.heads < 1 or args.dim % args.heads:
        raise SystemExit("--dim must be divisible by --heads")
    if args.seq_len < 2 or args.vocab_size < 2:
        raise SystemExit("--seq-len and --vocab-size must both be at least 2")
    if args.backend == "tt":
        for name, value in (
            ("--seq-len", args.seq_len),
            ("--vocab-size", args.vocab_size),
            ("--dim", args.dim),
            ("--hidden-dim", args.hidden_dim),
        ):
            if value % 32:
                raise SystemExit(f"{name} must be a multiple of 32 on tt, got {value}")


def make_qwen3_config(args) -> Qwen3Model.Config:
    linear_init = {
        "weight": partial(nn.init.normal_, std=0.02),
        "bias": nn.init.zeros_,
    }
    norm_init = {"weight": nn.init.ones_}

    def norm(size):
        return RMSNorm.Config(
            normalized_shape=size,
            eps=1e-6,
            param_init=norm_init,
        )

    head_dim = args.dim // args.heads
    layers = []
    for _ in range(args.layers):
        layers.append(
            Qwen3TransformerBlock.Config(
                attention_norm=norm(args.dim),
                ffn_norm=norm(args.dim),
                attention=make_gqa_config(
                    dim=args.dim,
                    n_heads=args.heads,
                    n_kv_heads=args.heads,
                    head_dim=head_dim,
                    wqkv_param_init=linear_init,
                    wo_param_init=linear_init,
                    inner_attention=TorchaxCausalAttention.Config(),
                    fuse_qkv=False,
                    rope=CosSinRoPE.Config(
                        dim=head_dim,
                        max_context_length=args.seq_len,
                        theta=1_000_000.0,
                    ),
                    qk_norm=norm(head_dim),
                ),
                feed_forward=make_ffn_config(
                    dim=args.dim,
                    hidden_dim=args.hidden_dim,
                    w1_param_init=linear_init,
                    w2w3_param_init=linear_init,
                ),
            )
        )

    return Qwen3Model.Config(
        vocab_size=args.vocab_size,
        dim=args.dim,
        enable_weight_tying=True,
        tok_embeddings=Embedding.Config(
            num_embeddings=args.vocab_size,
            embedding_dim=args.dim,
            param_init={"weight": partial(nn.init.normal_, std=0.02)},
        ),
        norm=norm(args.dim),
        lm_head=Linear.Config(
            in_features=args.dim,
            out_features=args.vocab_size,
            param_init=linear_init,
        ),
        layers=layers,
    )


def make_batch(seq_len: int, vocab_size: int):
    # A fixed next-token task makes a successful parameter update visible in a
    # handful of steps without downloading a tokenizer or dataset.
    tokens = torch.arange(seq_len, dtype=torch.int64) % vocab_size
    labels = (tokens + 1) % vocab_size
    return tokens, labels


def main():
    args = parse_args()
    validate_args(args)

    jax.config.update("jax_platforms", args.backend)
    jax.config.update("jax_use_shardy_partitioner", False)
    devices = jax.devices(args.backend)
    if not devices:
        raise SystemExit(f"no JAX devices found for backend {args.backend!r}")
    if args.device_index >= len(devices):
        raise SystemExit(
            f"--device-index {args.device_index} is out of range for devices={devices}"
        )
    device = devices[args.device_index]

    torch.manual_seed(args.seed)
    torch_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    model = make_qwen3_config(args).build().to(dtype=torch_dtype)
    model.init_states(buffer_device=torch.device("cpu"))
    model.train()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    torchax.enable_globally()
    torchax.enable_performance_mode()
    env = torchax.default_env()

    with jax.default_device(device), env:
        model = model.to("jax")
        jittable_model = JittableModule(model)

        def model_fn(weights, buffers, tokens):
            return jittable_model.functional_call(
                "forward", weights, buffers, tokens
            )

        def loss_fn(logits, labels):
            logits = logits.reshape(-1, logits.shape[-1]).float()
            return F.cross_entropy(logits, labels.reshape(-1))

        optimizer = optax.sgd(args.learning_rate)
        opt_state = interop.call_jax(optimizer.init, jittable_model.params)
        train_step = torchax.train.make_train_step(model_fn, loss_fn, optimizer)
        train_step = interop.jax_jit(
            train_step,
            kwargs_for_jax_jit={"donate_argnums": (0, 2)},
        )

        tokens_cpu, labels_cpu = make_batch(args.seq_len, args.vocab_size)
        tokens = tokens_cpu.to("jax")
        labels = labels_cpu.to("jax")

        print(f"device: {device}")
        print(f"parameters: {parameter_count:,}")
        print(
            "model: "
            f"Qwen3 layers={args.layers} dim={args.dim} heads={args.heads} "
            f"hidden={args.hidden_dim} vocab={args.vocab_size} seq={args.seq_len}"
        )

        losses = []
        for step_index in range(args.steps):
            start = time.perf_counter()
            loss, jittable_model.params, opt_state = train_step(
                jittable_model.params,
                jittable_model.buffers,
                opt_state,
                tokens,
                labels,
            )
            loss_value = float(loss.item())
            elapsed = time.perf_counter() - start
            if not math.isfinite(loss_value):
                raise SystemExit(f"step {step_index + 1} produced non-finite loss")
            losses.append(loss_value)
            print(
                f"step {step_index + 1:02d}: loss={loss_value:.6f} "
                f"elapsed={elapsed:.3f}s"
            )

    if args.validate_loss and len(losses) > 1 and losses[-1] >= losses[0]:
        raise SystemExit(
            f"loss did not decrease: first={losses[0]:.6f}, last={losses[-1]:.6f}"
        )

    if len(losses) > 1:
        print(f"loss_delta: {losses[-1] - losses[0]:.6f}")
    print("training: ok")


if __name__ == "__main__":
    main()
