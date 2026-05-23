#!/usr/bin/env python3
import argparse
import json
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import jax
import ml_dtypes
import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoTokenizer

jax.config.update("jax_use_shardy_partitioner", False)

import jax.numpy as jnp

USE_TT_ROPE_DECODE = False
USE_TT_SWIGLU = False
USE_TT_LM_HEAD_TOP1 = False


@dataclass(frozen=True)
class Qwen3Config:
    vocab_size: int = 151936
    hidden_size: int = 2560
    intermediate_size: int = 9728
    num_hidden_layers: int = 36
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    max_position_embeddings: int = 40960
    rope_theta: float = 1000000.0
    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.02
    tie_word_embeddings: bool = True

    def __post_init__(self):
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")


class ByteTokenizer:
    bos_token_id = 256

    def encode(self, text: str, add_bos: bool = True) -> np.ndarray:
        tokens = list(text.encode("utf-8", errors="replace"))
        if add_bos:
            tokens.insert(0, self.bos_token_id)
        return np.asarray(tokens, dtype=np.int32)

    def decode(self, token_ids) -> str:
        data = bytes(int(token) for token in token_ids if 0 <= int(token) < 256)
        return data.decode("utf-8", errors="replace")


class HuggingFaceTokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def encode(self, text: str, raw_prompt: bool, thinking: bool) -> np.ndarray:
        if not raw_prompt and getattr(self.tokenizer, "chat_template", None):
            token_ids = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                add_generation_prompt=True,
                enable_thinking=thinking,
                tokenize=True,
            )
        else:
            token_ids = self.tokenizer.encode(text, add_special_tokens=True)
        if isinstance(token_ids, Mapping):
            token_ids = token_ids["input_ids"]
        if token_ids and isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        return np.asarray(token_ids, dtype=np.int32)

    def decode(self, token_ids) -> str:
        return self.tokenizer.decode(
            [int(token) for token in token_ids],
            skip_special_tokens=True,
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Qwen3-4B or a tiny random-weight Qwen3-style decoder-only LLM with JAX."
    )
    parser.add_argument("--backend", default="tt")
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--revision")
    parser.add_argument("--cache-dir")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--raw-prompt", action="store_true")
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--random-weights", action="store_true")
    parser.add_argument("--prompt", default="Write a short sentence about Tenstorrent hardware.")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--intermediate-size", type=int, default=384)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--max-seq-len", type=int)
    parser.add_argument("--dtype", choices=("bf16", "f32"), default="bf16")
    return parser.parse_args()


def make_random_config(args) -> Qwen3Config:
    if args.hidden_size % args.attention_heads != 0:
        raise SystemExit("--hidden-size must be divisible by --attention-heads")
    return Qwen3Config(
        vocab_size=288,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_hidden_layers=args.layers,
        num_attention_heads=args.attention_heads,
        num_key_value_heads=args.kv_heads,
        head_dim=args.hidden_size // args.attention_heads,
        max_position_embeddings=args.max_seq_len or 512,
        rope_theta=10000.0,
        tie_word_embeddings=False,
    )


def load_hf_config(model_dir: Path, max_seq_len: int | None) -> Qwen3Config:
    config_path = model_dir / "config.json"
    with config_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    config = Qwen3Config(
        vocab_size=int(raw["vocab_size"]),
        hidden_size=int(raw["hidden_size"]),
        intermediate_size=int(raw["intermediate_size"]),
        num_hidden_layers=int(raw["num_hidden_layers"]),
        num_attention_heads=int(raw["num_attention_heads"]),
        num_key_value_heads=int(raw["num_key_value_heads"]),
        head_dim=int(raw["head_dim"]),
        max_position_embeddings=int(raw["max_position_embeddings"]),
        rope_theta=float(raw.get("rope_theta", 1000000.0)),
        rms_norm_eps=float(raw.get("rms_norm_eps", 1e-6)),
        initializer_range=float(raw.get("initializer_range", 0.02)),
        tie_word_embeddings=bool(raw.get("tie_word_embeddings", False)),
    )

    if max_seq_len is not None:
        config = replace(config, max_position_embeddings=max_seq_len)
    return config


def resolve_model_dir(args) -> Path:
    model_path = Path(args.model).expanduser()
    if model_path.exists():
        return model_path

    allow_patterns = [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
        "*.model",
        "chat_template.json",
        "*.safetensors",
        "*.safetensors.index.json",
    ]
    return Path(
        snapshot_download(
            repo_id=args.model,
            revision=args.revision,
            cache_dir=args.cache_dir,
            local_files_only=args.local_files_only,
            allow_patterns=allow_patterns,
        )
    )


def load_hf_tokenizer(model_dir: Path) -> HuggingFaceTokenizer:
    return HuggingFaceTokenizer(
        AutoTokenizer.from_pretrained(
            model_dir,
            trust_remote_code=False,
            local_files_only=True,
        )
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


def precompute_rope_cos_sin(config: Qwen3Config, seq_len: int, dtype):
    position_ids = np.arange(seq_len, dtype=np.float32)[:, None]
    inv_freq = 1.0 / (config.rope_theta ** (np.arange(0, config.head_dim, 2, dtype=np.float32) / config.head_dim))
    freqs = position_ids * inv_freq[None, :]
    emb = np.concatenate((freqs, freqs), axis=-1)
    return np.ascontiguousarray(np.cos(emb).astype(dtype)), np.ascontiguousarray(np.sin(emb).astype(dtype))


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


def safetensor_files(model_dir: Path) -> list[Path]:
    index_paths = sorted(model_dir.glob("*.safetensors.index.json"))
    if index_paths:
        with index_paths[0].open("r", encoding="utf-8") as f:
            index = json.load(f)
        files = sorted({model_dir / name for name in index["weight_map"].values()})
    else:
        files = sorted(model_dir.glob("*.safetensors"))

    return files


def tensor_to_numpy(tensor, np_dtype):
    tensor = tensor.detach().cpu().contiguous()
    if np.dtype(np_dtype) == np.dtype(ml_dtypes.bfloat16):
        if tensor.dtype == torch.bfloat16:
            int_dtype = torch.uint16 if hasattr(torch, "uint16") else torch.int16
            return tensor.view(int_dtype).numpy().view(np.uint16).view(ml_dtypes.bfloat16)
        return tensor.numpy().astype(np_dtype, copy=False)

    if tensor.dtype == torch.bfloat16:
        return tensor.to(torch.float32).numpy().astype(np_dtype, copy=False)
    return tensor.numpy().astype(np_dtype, copy=False)


def load_checkpoint_arrays(model_dir: Path, np_dtype):
    arrays = {}

    for path in safetensor_files(model_dir):
        with safe_open(path, framework="pt", device="cpu") as f:
            for name in f.keys():
                arrays[name] = tensor_to_numpy(f.get_tensor(name), np_dtype)

    return arrays


def load_hf_weights(config: Qwen3Config, model_dir: Path, np_dtype):
    tensors = load_checkpoint_arrays(model_dir, np_dtype)

    weights = {
        "embed_tokens": np.ascontiguousarray(tensors["model.embed_tokens.weight"]),
        "layers": [],
        "norm": np.ascontiguousarray(tensors["model.norm.weight"]),
    }

    if config.tie_word_embeddings:
        weights["lm_head"] = np.ascontiguousarray(weights["embed_tokens"].T)
    else:
        weights["lm_head"] = np.ascontiguousarray(tensors["lm_head.weight"].T)

    for index in range(config.num_hidden_layers):
        prefix = f"model.layers.{index}"
        layer = {
            "input_norm": np.ascontiguousarray(tensors[f"{prefix}.input_layernorm.weight"]),
            "post_attention_norm": np.ascontiguousarray(
                tensors[f"{prefix}.post_attention_layernorm.weight"]
            ),
            "q_norm": np.ascontiguousarray(tensors[f"{prefix}.self_attn.q_norm.weight"]),
            "k_norm": np.ascontiguousarray(tensors[f"{prefix}.self_attn.k_norm.weight"]),
            "q_proj": np.ascontiguousarray(tensors[f"{prefix}.self_attn.q_proj.weight"].T),
            "k_proj": np.ascontiguousarray(tensors[f"{prefix}.self_attn.k_proj.weight"].T),
            "v_proj": np.ascontiguousarray(tensors[f"{prefix}.self_attn.v_proj.weight"].T),
            "o_proj": np.ascontiguousarray(tensors[f"{prefix}.self_attn.o_proj.weight"].T),
            "gate_proj": np.ascontiguousarray(tensors[f"{prefix}.mlp.gate_proj.weight"].T),
            "up_proj": np.ascontiguousarray(tensors[f"{prefix}.mlp.up_proj.weight"].T),
            "down_proj": np.ascontiguousarray(tensors[f"{prefix}.mlp.down_proj.weight"].T),
        }
        weights["layers"].append(layer)

    return weights


def rms_norm(x, weight, eps):
    variance = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
    scale = jax.lax.rsqrt(variance + eps)
    return (x * scale * weight).astype(x.dtype)


def silu(x):
    return x / (1.0 + jnp.exp(-x))


def swiglu(gate, up):
    if USE_TT_SWIGLU and gate.dtype == jnp.bfloat16 and up.dtype == jnp.bfloat16:
        return jax.ffi.ffi_call(
            "tt.swiglu",
            jax.ShapeDtypeStruct(gate.shape, gate.dtype),
        )(gate, up)
    return silu(gate) * up


def rotate_half(x):
    half = x.shape[-1] // 2
    return jnp.concatenate((-x[..., half:], x[..., :half]), axis=-1)


def apply_rope(x, cos, sin):
    return (x * cos[:, None, :] + rotate_half(x) * sin[:, None, :]).astype(x.dtype)


def apply_rope_decode(x, cos, sin):
    if USE_TT_ROPE_DECODE and x.dtype == jnp.bfloat16 and x.shape[-1] % 64 == 0:
        return jax.ffi.ffi_call(
            "tt.rope_decode",
            jax.ShapeDtypeStruct(x.shape, x.dtype),
        )(x, cos, sin)
    return apply_rope(x, cos, sin)


def causal_attention_bias(seq_len: int):
    position_ids = jnp.arange(seq_len)
    return jnp.where(position_ids[None, :] > position_ids[:, None], -1.0e9, 0.0)[None, None, :, :]


def decode_attention(config: Qwen3Config, query_states, key_cache, value_cache):
    repeats = config.num_attention_heads // config.num_key_value_heads
    grouped_query = query_states[0].reshape(config.num_key_value_heads, repeats, config.head_dim)
    scores = jnp.einsum("hrd,hdt->hrt", grouped_query, key_cache).astype(jnp.float32)
    scores = scores * (config.head_dim**-0.5)
    probs = jax.nn.softmax(scores, axis=-1).astype(query_states.dtype)
    return jnp.einsum("hrt,htd->hrd", probs, value_cache).reshape(
        1, config.num_attention_heads, config.head_dim
    )


def self_attention(config: Qwen3Config, hidden_states, layer, cos, sin, causal_bias, cache=None):
    seq_len = hidden_states.shape[0]
    query_states = hidden_states @ layer["q_proj"]
    key_states = hidden_states @ layer["k_proj"]
    value_states = hidden_states @ layer["v_proj"]

    query_states = query_states.reshape(seq_len, config.num_attention_heads, config.head_dim)
    key_states = key_states.reshape(seq_len, config.num_key_value_heads, config.head_dim)
    value_states = value_states.reshape(seq_len, config.num_key_value_heads, config.head_dim)

    query_states = rms_norm(query_states, layer["q_norm"], config.rms_norm_eps)
    key_states = rms_norm(key_states, layer["k_norm"], config.rms_norm_eps)

    if cache is None:
        query_states = apply_rope(query_states, cos, sin)
        key_states = apply_rope(key_states, cos, sin)
        key_cache = jnp.transpose(key_states, (1, 2, 0))
        value_cache = jnp.transpose(value_states, (1, 0, 2))
        attn_output = jax.nn.dot_product_attention(
            query_states, key_states, value_states, bias=causal_bias, implementation="xla"
        )
    else:
        query_states = apply_rope_decode(query_states, cos, sin)
        key_states = apply_rope_decode(key_states, cos, sin)
        key_cache = jnp.concatenate((cache[0], jnp.transpose(key_states, (1, 2, 0))), axis=2)
        value_cache = jnp.concatenate((cache[1], jnp.transpose(value_states, (1, 0, 2))), axis=1)
        attn_output = decode_attention(config, query_states, key_cache, value_cache)
    output = attn_output.reshape(seq_len, -1) @ layer["o_proj"]
    return output, (key_cache, value_cache)


def mlp(hidden_states, layer):
    gate = hidden_states @ layer["gate_proj"]
    up = hidden_states @ layer["up_proj"]
    return swiglu(gate, up) @ layer["down_proj"]


def qwen3_forward(config: Qwen3Config, weights, input_ids, caches=None):
    input_ids = input_ids.reshape((-1,))
    seq_len = input_ids.shape[0]
    if caches is not None and seq_len != 1:
        raise ValueError("cached decode expects exactly one new token")
    cache_len = 0 if caches is None else caches[0][0].shape[2]
    hidden_states = weights["embed_tokens"][input_ids]
    cos = weights["rope_cos"][cache_len : cache_len + seq_len]
    sin = weights["rope_sin"][cache_len : cache_len + seq_len]
    causal_bias = causal_attention_bias(seq_len) if caches is None else None
    new_caches = []
    layer_caches = caches if caches is not None else (None,) * len(weights["layers"])
    if len(layer_caches) != len(weights["layers"]):
        raise ValueError("KV cache must have one entry per decoder layer")
    for layer, cache in zip(weights["layers"], layer_caches):
        residual = hidden_states
        hidden_states = rms_norm(hidden_states, layer["input_norm"], config.rms_norm_eps)
        attn_output, new_cache = self_attention(config, hidden_states, layer, cos, sin, causal_bias, cache)
        new_caches.append(new_cache)
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = rms_norm(hidden_states, layer["post_attention_norm"], config.rms_norm_eps)
        hidden_states = residual + mlp(hidden_states, layer)
    return rms_norm(hidden_states, weights["norm"], config.rms_norm_eps), tuple(new_caches)


def qwen3_next_token_and_cache(config: Qwen3Config, weights, input_ids, caches=None):
    hidden_states, caches = qwen3_forward(config, weights, input_ids, caches=caches)
    head = weights["lm_head"] if "lm_head" in weights else weights["embed_tokens"].T
    hidden = hidden_states[-1:, :]
    if USE_TT_LM_HEAD_TOP1 and hidden.dtype == jnp.bfloat16 and head.dtype == jnp.bfloat16:
        token = jax.ffi.ffi_call(
            "tt.lm_head_top1",
            jax.ShapeDtypeStruct((1,), jnp.int32),
        )(hidden, head)[0]
    else:
        token = jax.lax.top_k((hidden @ head)[0], 1)[1][0].astype(jnp.int32)
    return token, caches


def count_parameters(config: Qwen3Config, weights) -> int:
    counted_weights = dict(weights)
    if config.tie_word_embeddings:
        counted_weights.pop("lm_head", None)
    leaves = jax.tree_util.tree_leaves(counted_weights)
    return sum(value.size for value in leaves)


def make_decode_step(config, device, args):
    def greedy_tokens(model_weights, ids):
        token, caches = qwen3_next_token_and_cache(config, model_weights, ids)
        generated = [token.reshape((1, 1))]
        for _ in range(1, args.max_new_tokens):
            token, caches = qwen3_next_token_and_cache(config, model_weights, token, caches)
            generated.append(token.reshape((1, 1)))
        return jnp.concatenate(generated, axis=0)

    return jax.jit(greedy_tokens, device=device)


def generate(config, weights, device, input_ids, args, decode_step):
    tokens = input_ids.astype(np.int32).copy()

    if args.max_new_tokens == 0:
        return tokens
    ids = jax.device_put(jnp.asarray(tokens, dtype=jnp.int32), device)
    generated = np.asarray(decode_step(weights, ids), dtype=np.int32).reshape(-1)
    return np.concatenate((tokens, generated))


def timed_generate(config, weights, device, input_ids, args, decode_step):
    start = time.perf_counter()
    output_ids = generate(config, weights, device, input_ids, args, decode_step)
    return output_ids, time.perf_counter() - start


def main():
    global USE_TT_ROPE_DECODE, USE_TT_SWIGLU, USE_TT_LM_HEAD_TOP1

    args = parse_args()
    if args.max_new_tokens < 0:
        raise SystemExit("--max-new-tokens must be non-negative")

    np_dtype = ml_dtypes.bfloat16 if args.dtype == "bf16" else np.float32
    USE_TT_ROPE_DECODE = args.backend == "tt" and args.dtype == "bf16"
    USE_TT_SWIGLU = args.backend == "tt" and args.dtype == "bf16"
    USE_TT_LM_HEAD_TOP1 = args.backend == "tt" and args.dtype == "bf16"

    if args.random_weights:
        config = make_random_config(args)
        tokenizer = ByteTokenizer()
        input_ids = tokenizer.encode(args.prompt)
        weights_host = init_weights(config, args.seed, np_dtype)
    else:
        model_dir = resolve_model_dir(args)
        config = load_hf_config(model_dir, args.max_seq_len)
        tokenizer = load_hf_tokenizer(model_dir)
        input_ids = tokenizer.encode(args.prompt, args.raw_prompt, args.thinking)
        weights_host = load_hf_weights(config, model_dir, np_dtype)

    if input_ids.size + args.max_new_tokens > config.max_position_embeddings:
        raise SystemExit(
            "prompt plus generation length exceeds --max-seq-len: "
            f"{input_ids.size + args.max_new_tokens} > {config.max_position_embeddings}"
        )
    weights_host["rope_cos"], weights_host["rope_sin"] = precompute_rope_cos_sin(config, input_ids.size + args.max_new_tokens, np_dtype)

    device = select_device(args.backend)
    weights = jax.device_put(weights_host, device)
    decode_step = make_decode_step(config, device, args)
    if args.warmup:
        _, warmup_seconds = timed_generate(config, weights, device, input_ids, args, decode_step)
    else:
        warmup_seconds = 0.0
    output_ids, run_seconds = timed_generate(config, weights, device, input_ids, args, decode_step)

    print(f"device: {device}")
    if not args.random_weights:
        print(f"model: {args.model}")
    print(f"parameters: {count_parameters(config, weights):,}")
    print(f"warmup_seconds: {warmup_seconds:.3f}")
    print(f"run_seconds: {run_seconds:.3f}")
    print(tokenizer.decode(output_ids))


if __name__ == "__main__":
    main()
