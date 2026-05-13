#!/usr/bin/env python3
import argparse
import time
from dataclasses import dataclass

import jax

jax.config.update("jax_use_shardy_partitioner", False)

import jax.numpy as jnp
import ml_dtypes
import numpy as np


@dataclass(frozen=True)
class Qwen3Config:
    vocab_size: int = 288
    hidden_size: int = 128
    intermediate_size: int = 384
    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    head_dim: int = 32
    max_position_embeddings: int = 512
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.02

    def __post_init__(self):
        if self.hidden_size != self.num_attention_heads * self.head_dim:
            raise ValueError("hidden_size must equal num_attention_heads * head_dim")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")


class ByteTokenizer:
    bos_token_id = 256
    eos_token_id = 257

    def encode(self, text: str, add_bos: bool = True) -> np.ndarray:
        tokens = list(text.encode("utf-8", errors="replace"))
        if add_bos:
            tokens.insert(0, self.bos_token_id)
        return np.asarray(tokens, dtype=np.int32)

    def decode(self, token_ids) -> str:
        data = bytes(int(token) for token in token_ids if 0 <= int(token) < 256)
        return data.decode("utf-8", errors="replace")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a tiny random-weight Qwen3-style decoder-only LLM with JAX."
    )
    parser.add_argument("--backend", default="tt")
    parser.add_argument("--prompt", default="Qwen3 tiny model:")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--intermediate-size", type=int, default=384)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--dtype", choices=("bf16", "f32"), default="bf16")
    return parser.parse_args()


def make_config(args) -> Qwen3Config:
    if args.hidden_size % args.attention_heads != 0:
        raise SystemExit("--hidden-size must be divisible by --attention-heads")
    return Qwen3Config(
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_hidden_layers=args.layers,
        num_attention_heads=args.attention_heads,
        num_key_value_heads=args.kv_heads,
        head_dim=args.hidden_size // args.attention_heads,
        max_position_embeddings=args.max_seq_len,
    )


def select_device(backend: str):
    try:
        devices = jax.devices(backend)
    except RuntimeError as err:
        raise SystemExit(f"could not initialize JAX backend {backend!r}: {err}") from err
    if not devices:
        raise SystemExit(f"expected at least one {backend!r} device, got devices={jax.devices()}")
    return devices[0]


def normal(rng: np.random.Generator, shape, std, dtype):
    return rng.normal(0.0, std, shape).astype(dtype)


def init_weights(config: Qwen3Config, seed: int, np_dtype):
    q_out = config.num_attention_heads * config.head_dim
    kv_out = config.num_key_value_heads * config.head_dim
    rng = np.random.default_rng(seed)

    layers = []
    for _ in range(config.num_hidden_layers):
        layers.append(
            {
                "input_norm": np.ones((config.hidden_size,), dtype=np_dtype),
                "post_attention_norm": np.ones((config.hidden_size,), dtype=np_dtype),
                "q_norm": np.ones((config.head_dim,), dtype=np_dtype),
                "k_norm": np.ones((config.head_dim,), dtype=np_dtype),
                "q_proj": normal(
                    rng,
                    (config.hidden_size, q_out),
                    config.initializer_range,
                    np_dtype,
                ),
                "k_proj": normal(
                    rng,
                    (config.hidden_size, kv_out),
                    config.initializer_range,
                    np_dtype,
                ),
                "v_proj": normal(
                    rng,
                    (config.hidden_size, kv_out),
                    config.initializer_range,
                    np_dtype,
                ),
                "o_proj": normal(
                    rng,
                    (q_out, config.hidden_size),
                    config.initializer_range,
                    np_dtype,
                ),
                "gate_proj": normal(
                    rng,
                    (config.hidden_size, config.intermediate_size),
                    config.initializer_range,
                    np_dtype,
                ),
                "up_proj": normal(
                    rng,
                    (config.hidden_size, config.intermediate_size),
                    config.initializer_range,
                    np_dtype,
                ),
                "down_proj": normal(
                    rng,
                    (config.intermediate_size, config.hidden_size),
                    config.initializer_range,
                    np_dtype,
                ),
            }
        )

    return {
        "embed_tokens": normal(
            rng,
            (config.vocab_size, config.hidden_size),
            config.initializer_range,
            np_dtype,
        ),
        "layers": layers,
        "norm": np.ones((config.hidden_size,), dtype=np_dtype),
        "lm_head": normal(
            rng,
            (config.hidden_size, config.vocab_size),
            config.initializer_range,
            np_dtype,
        ),
    }


def rms_norm(x, weight, eps):
    variance = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
    return (x * jax.lax.rsqrt(variance + eps) * weight).astype(x.dtype)


def silu(x):
    return x * jax.nn.sigmoid(x)


def rotate_half(x):
    half = x.shape[-1] // 2
    return jnp.concatenate((-x[..., half:], x[..., :half]), axis=-1)


def rope_cos_sin(config: Qwen3Config, seq_len: int):
    position_ids = jnp.arange(seq_len, dtype=jnp.float32)[:, None]
    inv_freq = 1.0 / (
        config.rope_theta
        ** (jnp.arange(0, config.head_dim, 2, dtype=jnp.float32) / config.head_dim)
    )
    freqs = position_ids * inv_freq[None, :]
    emb = jnp.concatenate((freqs, freqs), axis=-1)
    return jnp.cos(emb), jnp.sin(emb)


def apply_rope(x, cos, sin):
    return (x * cos[:, None, :] + rotate_half(x) * sin[:, None, :]).astype(x.dtype)


def self_attention(config: Qwen3Config, hidden_states, layer, cos, sin):
    seq_len = hidden_states.shape[0]
    query_states = hidden_states @ layer["q_proj"]
    key_states = hidden_states @ layer["k_proj"]
    value_states = hidden_states @ layer["v_proj"]

    query_states = query_states.reshape(seq_len, config.num_attention_heads, config.head_dim)
    key_states = key_states.reshape(seq_len, config.num_key_value_heads, config.head_dim)
    value_states = value_states.reshape(seq_len, config.num_key_value_heads, config.head_dim)

    query_states = rms_norm(query_states, layer["q_norm"], config.rms_norm_eps)
    key_states = rms_norm(key_states, layer["k_norm"], config.rms_norm_eps)
    query_states = apply_rope(query_states, cos, sin)
    key_states = apply_rope(key_states, cos, sin)

    repeats = config.num_attention_heads // config.num_key_value_heads
    key_states = jnp.repeat(key_states, repeats, axis=1)
    value_states = jnp.repeat(value_states, repeats, axis=1)

    scores = jnp.stack(
        [
            query_states[:, head, :] @ key_states[:, head, :].T
            for head in range(config.num_attention_heads)
        ],
        axis=1,
    )
    scores = scores.astype(jnp.float32) * (config.head_dim**-0.5)
    position_ids = jnp.arange(seq_len)
    causal_mask = position_ids[None, :] > position_ids[:, None]
    scores = jnp.where(causal_mask[:, None, :], -1.0e9, scores)
    probs = jax.nn.softmax(scores, axis=-1).astype(hidden_states.dtype)
    attn_output = jnp.stack(
        [
            probs[:, head, :] @ value_states[:, head, :]
            for head in range(config.num_attention_heads)
        ],
        axis=1,
    )
    attn_output = attn_output.reshape(seq_len, -1)
    return attn_output @ layer["o_proj"]


def mlp(hidden_states, layer):
    gate = silu(hidden_states @ layer["gate_proj"])
    up = hidden_states @ layer["up_proj"]
    return (gate * up) @ layer["down_proj"]


def qwen3_forward(config: Qwen3Config, weights, input_ids):
    hidden_states = weights["embed_tokens"][input_ids]
    cos, sin = rope_cos_sin(config, input_ids.shape[0])
    cos = cos.astype(hidden_states.dtype)
    sin = sin.astype(hidden_states.dtype)

    for layer in weights["layers"]:
        residual = hidden_states
        hidden_states = rms_norm(hidden_states, layer["input_norm"], config.rms_norm_eps)
        hidden_states = residual + self_attention(config, hidden_states, layer, cos, sin)

        residual = hidden_states
        hidden_states = rms_norm(
            hidden_states, layer["post_attention_norm"], config.rms_norm_eps
        )
        hidden_states = residual + mlp(hidden_states, layer)

    hidden_states = rms_norm(hidden_states, weights["norm"], config.rms_norm_eps)
    return hidden_states @ weights["lm_head"]


def sample_next_token(logits, rng: np.random.Generator, temperature: float, top_k: int) -> int:
    logits = np.asarray(logits, dtype=np.float32)
    if temperature <= 0:
        return int(np.argmax(logits))

    logits = logits / temperature
    if 0 < top_k < logits.size:
        candidates = np.lexsort((np.arange(logits.size), -logits))[:top_k]
        candidate_logits = logits[candidates]
        probs = np.exp(candidate_logits - np.max(candidate_logits))
        probs = probs / np.sum(probs)
        return int(rng.choice(candidates, p=probs))

    probs = np.exp(logits - np.max(logits))
    probs = probs / np.sum(probs)
    return int(rng.choice(np.arange(logits.size), p=probs))


def count_parameters(weights) -> int:
    leaves = jax.tree_util.tree_leaves(weights)
    return sum(value.size for value in leaves)


def make_forward(config, device):
    return jax.jit(
        lambda model_weights, ids: qwen3_forward(config, model_weights, ids)[-1],
        device=device,
    )


def generate(config, weights, device, input_ids, args, forward):
    rng = np.random.default_rng(args.seed + 1)
    tokens = input_ids.astype(np.int32).copy()

    for _ in range(args.max_new_tokens):
        ids = jax.device_put(jnp.asarray(tokens, dtype=jnp.int32), device)
        logits = np.asarray(forward(weights, ids))
        next_token = sample_next_token(logits, rng, args.temperature, args.top_k)
        tokens = np.append(tokens, np.int32(next_token))
        if next_token == ByteTokenizer.eos_token_id:
            break
        if tokens.size >= config.max_position_embeddings:
            break
    return tokens


def timed_generate(config, weights, device, input_ids, args, forward):
    start = time.perf_counter()
    output_ids = generate(config, weights, device, input_ids, args, forward)
    return output_ids, time.perf_counter() - start


def main():
    args = parse_args()
    config = make_config(args)
    if args.max_new_tokens < 0:
        raise SystemExit("--max-new-tokens must be non-negative")

    dtype = jnp.bfloat16 if args.dtype == "bf16" else jnp.float32
    np_dtype = ml_dtypes.bfloat16 if args.dtype == "bf16" else np.float32
    tokenizer = ByteTokenizer()
    input_ids = tokenizer.encode(args.prompt)
    if input_ids.size + args.max_new_tokens > config.max_position_embeddings:
        raise SystemExit(
            "prompt plus generation length exceeds --max-seq-len: "
            f"{input_ids.size + args.max_new_tokens} > {config.max_position_embeddings}"
        )

    device = select_device(args.backend)
    weights = jax.device_put(init_weights(config, args.seed, np_dtype), device)
    forward = make_forward(config, device)
    _, warmup_seconds = timed_generate(config, weights, device, input_ids, args, forward)
    output_ids, run_seconds = timed_generate(config, weights, device, input_ids, args, forward)

    print(f"device: {device}")
    print(f"parameters: {count_parameters(weights):,}")
    print(f"warmup_seconds: {warmup_seconds:.3f}")
    print(f"run_seconds: {run_seconds:.3f}")
    print(tokenizer.decode(output_ids))


if __name__ == "__main__":
    main()
