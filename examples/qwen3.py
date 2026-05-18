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


@dataclass(frozen=True)
class Qwen3Config:
    vocab_size: int = 151936
    hidden_size: int = 1024
    intermediate_size: int = 3072
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    max_position_embeddings: int = 40960
    rope_theta: float = 1000000.0
    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.02
    tie_word_embeddings: bool = True
    eos_token_id: int | tuple[int, ...] | None = 151645

    def __post_init__(self):
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


class HuggingFaceTokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.eos_token_id = tokenizer.eos_token_id

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
        description="Run Qwen3-0.6B or a tiny random-weight Qwen3-style decoder-only LLM with JAX."
    )
    parser.add_argument("--backend", default="tt")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
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
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--no-kv-cache", action="store_true")
    parser.add_argument("--stop-on-eos", action="store_true")
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
        eos_token_id=ByteTokenizer.eos_token_id,
    )


def load_hf_config(model_dir: Path, max_seq_len: int | None) -> Qwen3Config:
    config_path = model_dir / "config.json"
    with config_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    eos_token_id = raw.get("eos_token_id")
    if isinstance(eos_token_id, list):
        eos_token_id = tuple(int(token) for token in eos_token_id)
    elif eos_token_id is not None:
        eos_token_id = int(eos_token_id)

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
        eos_token_id=eos_token_id,
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
    inv_freq = 1.0 / (
        config.rope_theta
        ** (np.arange(0, config.head_dim, 2, dtype=np.float32) / config.head_dim)
    )
    freqs = position_ids * inv_freq[None, :]
    emb = np.concatenate((freqs, freqs), axis=-1)
    return (
        np.ascontiguousarray(np.cos(emb).astype(dtype)),
        np.ascontiguousarray(np.sin(emb).astype(dtype)),
    )


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

    if not config.tie_word_embeddings:
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
    return (x * jax.lax.rsqrt(variance + eps) * weight).astype(x.dtype)


def silu(x):
    return x * jax.nn.sigmoid(x)


def rotate_half(x):
    half = x.shape[-1] // 2
    return jnp.concatenate((-x[..., half:], x[..., :half]), axis=-1)


def apply_rope(x, cos, sin):
    return (x * cos[:, None, :] + rotate_half(x) * sin[:, None, :]).astype(x.dtype)


def causal_attention_bias(seq_len: int):
    position_ids = jnp.arange(seq_len)
    bias = jnp.where(
        position_ids[None, :] > position_ids[:, None],
        -1.0e9,
        0.0,
    )
    return bias[None, None, :, :]


def self_attention(
    config: Qwen3Config,
    hidden_states,
    layer,
    cos,
    sin,
    causal_bias,
    cache=None,
    use_cache: bool = False,
):
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

    new_cache = None
    if use_cache:
        key_cache = key_states.reshape((seq_len, -1))
        value_cache = value_states.reshape((seq_len, -1))
        if cache is not None:
            key_cache = jnp.concatenate((cache[0], key_cache), axis=0)
            value_cache = jnp.concatenate((cache[1], value_cache), axis=0)
        key_states = key_cache.reshape(
            (key_cache.shape[0], config.num_key_value_heads, config.head_dim)
        )
        value_states = value_cache.reshape(
            (value_cache.shape[0], config.num_key_value_heads, config.head_dim)
        )
        new_cache = (key_cache, value_cache)

    attn_output = jax.nn.dot_product_attention(
        query_states,
        key_states,
        value_states,
        bias=causal_bias,
        implementation="xla",
    )
    output = attn_output.reshape(seq_len, -1) @ layer["o_proj"]
    return output, new_cache


def mlp(hidden_states, layer):
    gate = silu(hidden_states @ layer["gate_proj"])
    up = hidden_states @ layer["up_proj"]
    return (gate * up) @ layer["down_proj"]


def qwen3_layers(
    config: Qwen3Config,
    weights,
    hidden_states,
    cos,
    sin,
    causal_bias,
    caches=None,
    use_cache: bool = False,
):
    new_caches = [] if use_cache else None
    layer_caches = caches if caches is not None else (None,) * len(weights["layers"])
    if len(layer_caches) != len(weights["layers"]):
        raise ValueError("KV cache must have one entry per decoder layer")
    for layer, cache in zip(weights["layers"], layer_caches):
        residual = hidden_states
        hidden_states = rms_norm(hidden_states, layer["input_norm"], config.rms_norm_eps)
        attn_output, new_cache = self_attention(
            config,
            hidden_states,
            layer,
            cos,
            sin,
            causal_bias,
            cache=cache,
            use_cache=use_cache,
        )
        if use_cache:
            new_caches.append(new_cache)
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = rms_norm(
            hidden_states, layer["post_attention_norm"], config.rms_norm_eps
        )
        hidden_states = residual + mlp(hidden_states, layer)

    return hidden_states, tuple(new_caches) if use_cache else None


def project_logits(weights, hidden_states):
    if "lm_head" in weights:
        return hidden_states @ weights["lm_head"]
    return hidden_states @ weights["embed_tokens"].T


def qwen3_forward(config: Qwen3Config, weights, input_ids, caches=None, use_cache=False):
    input_ids = input_ids.reshape((-1,))
    seq_len = input_ids.shape[0]
    if caches is not None:
        if seq_len != 1:
            raise ValueError("cached decode expects exactly one new token")
        use_cache = True
    cache_len = 0 if caches is None else caches[0][0].shape[0]
    hidden_states = weights["embed_tokens"][input_ids]
    cos = weights["rope_cos"][cache_len : cache_len + seq_len]
    sin = weights["rope_sin"][cache_len : cache_len + seq_len]
    causal_bias = causal_attention_bias(seq_len) if caches is None else None
    hidden_states, new_caches = qwen3_layers(
        config,
        weights,
        hidden_states,
        cos,
        sin,
        causal_bias,
        caches=caches,
        use_cache=use_cache,
    )
    return rms_norm(hidden_states, weights["norm"], config.rms_norm_eps), new_caches


def qwen3_hidden_states(config: Qwen3Config, weights, input_ids):
    hidden_states, _ = qwen3_forward(config, weights, input_ids)
    return hidden_states


def qwen3_next_logits(config: Qwen3Config, weights, input_ids):
    last_hidden = qwen3_hidden_states(config, weights, input_ids)[-1:, :]
    return project_logits(weights, last_hidden)[0]


def qwen3_next_logits_and_cache(config: Qwen3Config, weights, input_ids, caches=None):
    hidden_states, caches = qwen3_forward(
        config, weights, input_ids, caches=caches, use_cache=True
    )
    last_hidden = hidden_states[-1:, :]
    return project_logits(weights, last_hidden)[0], caches


def sample_next_token(
    decode_output, rng: np.random.Generator, temperature: float, top_k: int
) -> int:
    if isinstance(decode_output, (list, tuple)) and len(decode_output) == 2:
        logits, token_ids = decode_output
        top_k = 0
    else:
        logits, token_ids = decode_output, None

    logits = np.asarray(logits, dtype=np.float32)
    token_ids = (
        np.arange(logits.size, dtype=np.int32)
        if token_ids is None
        else np.asarray(token_ids, dtype=np.int32)
    )

    logits = logits.reshape(-1)
    token_ids = token_ids.reshape(-1)
    if 0 < top_k < logits.size:
        top = np.lexsort((token_ids, -logits))[:top_k]
        logits = logits[top]
        token_ids = token_ids[top]

    if temperature <= 0:
        return int(token_ids[np.argmax(logits)])

    logits = logits / temperature
    probs = np.exp(logits - np.max(logits))
    probs = probs / np.sum(probs)
    return int(rng.choice(token_ids, p=probs))


def count_parameters(weights) -> int:
    leaves = jax.tree_util.tree_leaves(weights)
    return sum(value.size for value in leaves)


def make_decode_step(config, device, args):
    device_top_k = 1 if args.temperature <= 0 else args.top_k

    def decode_output(logits):
        if 0 < device_top_k <= 32 and device_top_k < config.vocab_size:
            return jax.lax.top_k(logits, device_top_k)
        return logits

    if not args.no_kv_cache:
        if args.temperature <= 0 and not args.stop_on_eos:
            def greedy_tokens(model_weights, ids):
                logits, caches = qwen3_next_logits_and_cache(config, model_weights, ids)
                token = jax.lax.top_k(logits, 1)[1][0].astype(jnp.int32)
                generated = [token.reshape((1, 1))]
                for _ in range(1, args.max_new_tokens):
                    logits, caches = qwen3_next_logits_and_cache(
                        config, model_weights, token, caches
                    )
                    token = jax.lax.top_k(logits, 1)[1][0].astype(jnp.int32)
                    generated.append(token.reshape((1, 1)))
                return jnp.concatenate(generated, axis=0)

            return ("greedy_cached", jax.jit(greedy_tokens, device=device))

        def prefill(model_weights, ids):
            logits, caches = qwen3_next_logits_and_cache(config, model_weights, ids)
            return decode_output(logits), caches

        def decode_token(model_weights, token_id, caches):
            logits, new_caches = qwen3_next_logits_and_cache(
                config, model_weights, token_id, caches
            )
            return decode_output(logits), new_caches

        return (
            jax.jit(prefill, device=device),
            jax.jit(decode_token, device=device),
        )

    def logits(model_weights, ids):
        return qwen3_next_logits(config, model_weights, ids)

    if 0 < device_top_k <= 32 and device_top_k < config.vocab_size:
        return jax.jit(
            lambda model_weights, ids: decode_output(logits(model_weights, ids)),
            device=device,
        )

    return jax.jit(logits, device=device)


def generate(config, weights, device, input_ids, args, decode_step):
    rng = np.random.default_rng(args.seed + 1)
    tokens = input_ids.astype(np.int32).copy()
    eos_token_ids = config.eos_token_id
    if isinstance(eos_token_ids, int):
        eos_token_ids = (eos_token_ids,)

    if (
        isinstance(decode_step, tuple)
        and len(decode_step) == 2
        and decode_step[0] == "greedy_cached"
    ):
        if args.max_new_tokens == 0:
            return tokens
        ids = jax.device_put(jnp.asarray(tokens, dtype=jnp.int32), device)
        generated = np.asarray(decode_step[1](weights, ids), dtype=np.int32).reshape(-1)
        return np.concatenate((tokens, generated))

    if isinstance(decode_step, tuple):
        if args.max_new_tokens == 0:
            return tokens
        prefill_step, token_step = decode_step
        ids = jax.device_put(jnp.asarray(tokens, dtype=jnp.int32), device)
        decode_output, caches = prefill_step(weights, ids)
        next_token = sample_next_token(
            decode_output, rng, args.temperature, args.top_k
        )
        tokens = np.append(tokens, np.int32(next_token))
        if args.stop_on_eos and eos_token_ids is not None and next_token in eos_token_ids:
            return tokens

        for _ in range(1, args.max_new_tokens):
            token_id = jax.device_put(jnp.asarray([next_token], dtype=jnp.int32), device)
            decode_output, caches = token_step(weights, token_id, caches)
            next_token = sample_next_token(
                decode_output, rng, args.temperature, args.top_k
            )
            tokens = np.append(tokens, np.int32(next_token))
            if args.stop_on_eos and eos_token_ids is not None and next_token in eos_token_ids:
                break
            if tokens.size >= config.max_position_embeddings:
                break
        return tokens

    for _ in range(args.max_new_tokens):
        ids = jax.device_put(jnp.asarray(tokens, dtype=jnp.int32), device)
        next_token = sample_next_token(
            decode_step(weights, ids), rng, args.temperature, args.top_k
        )
        tokens = np.append(tokens, np.int32(next_token))
        if args.stop_on_eos and eos_token_ids is not None and next_token in eos_token_ids:
            break
        if tokens.size >= config.max_position_embeddings:
            break
    return tokens


def timed_generate(config, weights, device, input_ids, args, decode_step):
    start = time.perf_counter()
    output_ids = generate(config, weights, device, input_ids, args, decode_step)
    return output_ids, time.perf_counter() - start


def main():
    args = parse_args()
    if args.max_new_tokens < 0:
        raise SystemExit("--max-new-tokens must be non-negative")

    np_dtype = ml_dtypes.bfloat16 if args.dtype == "bf16" else np.float32

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
    weights_host["rope_cos"], weights_host["rope_sin"] = precompute_rope_cos_sin(
        config,
        input_ids.size + args.max_new_tokens,
        np_dtype,
    )

    device = select_device(args.backend)
    weights = jax.device_put(weights_host, device)
    decode_step = make_decode_step(config, device, args)
    if args.warmup:
        _, warmup_seconds = timed_generate(
            config, weights, device, input_ids, args, decode_step
        )
    else:
        warmup_seconds = 0.0
    output_ids, run_seconds = timed_generate(
        config, weights, device, input_ids, args, decode_step
    )

    print(f"device: {device}")
    if not args.random_weights:
        print(f"model: {args.model}")
    print(f"parameters: {count_parameters(weights):,}")
    print(f"warmup_seconds: {warmup_seconds:.3f}")
    print(f"run_seconds: {run_seconds:.3f}")
    print(tokenizer.decode(output_ids))


if __name__ == "__main__":
    main()
