#!/usr/bin/env python3
import argparse
import json
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

try:
    jax.config.update("jax_use_shardy_partitioner", False)
except Exception:
    pass


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
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.num_attention_heads <= 0:
            raise ValueError("num_attention_heads must be positive")
        if self.num_key_value_heads <= 0:
            raise ValueError("num_key_value_heads must be positive")
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

    if not files:
        raise SystemExit(f"no safetensors checkpoint files found in {model_dir}")
    return files


def tensor_to_numpy(tensor, np_dtype):
    import torch

    tensor = tensor.detach().cpu().contiguous()
    if np.dtype(np_dtype) == np.dtype(ml_dtypes.bfloat16):
        if tensor.dtype == torch.bfloat16:
            int_dtype = torch.uint16 if hasattr(torch, "uint16") else torch.int16
            return tensor.view(int_dtype).numpy().view(np.uint16).view(ml_dtypes.bfloat16)
        return tensor.numpy().astype(np_dtype, copy=False)

    if tensor.dtype == torch.bfloat16:
        return tensor.to(torch.float32).numpy().astype(np_dtype, copy=False)
    return tensor.numpy().astype(np_dtype, copy=False)


def checkpoint_tensor_names(config: Qwen3Config) -> tuple[set[str], set[str]]:
    required = {
        "model.embed_tokens.weight",
        "model.norm.weight",
    }
    optional = {"lm_head.weight"}

    for index in range(config.num_hidden_layers):
        prefix = f"model.layers.{index}"
        required.update(
            {
                f"{prefix}.input_layernorm.weight",
                f"{prefix}.post_attention_layernorm.weight",
                f"{prefix}.self_attn.q_norm.weight",
                f"{prefix}.self_attn.k_norm.weight",
                f"{prefix}.self_attn.q_proj.weight",
                f"{prefix}.self_attn.k_proj.weight",
                f"{prefix}.self_attn.v_proj.weight",
                f"{prefix}.self_attn.o_proj.weight",
                f"{prefix}.mlp.gate_proj.weight",
                f"{prefix}.mlp.up_proj.weight",
                f"{prefix}.mlp.down_proj.weight",
            }
        )

    if not config.tie_word_embeddings:
        required.add("lm_head.weight")
        optional.remove("lm_head.weight")
    return required, optional


def load_checkpoint_arrays(config: Qwen3Config, model_dir: Path, np_dtype):
    try:
        from safetensors import safe_open
    except ModuleNotFoundError as err:
        raise SystemExit(
            "missing Python dependency 'safetensors'. Install the example dependencies, "
            "for example: python3 -m pip install jax ml-dtypes transformers "
            "huggingface-hub safetensors torch"
        ) from err

    try:
        import torch  # noqa: F401
    except ModuleNotFoundError as err:
        raise SystemExit(
            "missing Python dependency 'torch'. Install the example dependencies, "
            "for example: python3 -m pip install jax ml-dtypes transformers "
            "huggingface-hub safetensors torch"
        ) from err

    required, optional = checkpoint_tensor_names(config)
    remaining = set(required) | set(optional)
    arrays = {}

    for path in safetensor_files(model_dir):
        with safe_open(path, framework="pt", device="cpu") as f:
            keys = set(f.keys())
            for name in sorted(remaining & keys):
                arrays[name] = tensor_to_numpy(f.get_tensor(name), np_dtype)
                remaining.remove(name)

    missing_required = sorted(required - arrays.keys())
    if missing_required:
        preview = ", ".join(missing_required[:8])
        if len(missing_required) > 8:
            preview += f", ... ({len(missing_required)} total)"
        raise SystemExit(f"checkpoint is missing required tensors: {preview}")
    return arrays


def expect_shape(name: str, value, expected):
    if value.shape != expected:
        raise SystemExit(f"{name} has shape {value.shape}, expected {expected}")
    return np.ascontiguousarray(value)


def load_hf_weights(config: Qwen3Config, model_dir: Path, np_dtype):
    tensors = load_checkpoint_arrays(config, model_dir, np_dtype)
    q_out = config.num_attention_heads * config.head_dim
    kv_out = config.num_key_value_heads * config.head_dim

    embed_tokens = expect_shape(
        "model.embed_tokens.weight",
        tensors["model.embed_tokens.weight"],
        (config.vocab_size, config.hidden_size),
    )
    weights = {
        "embed_tokens": embed_tokens,
        "layers": [],
        "norm": expect_shape(
            "model.norm.weight",
            tensors["model.norm.weight"],
            (config.hidden_size,),
        ),
    }

    if "lm_head.weight" in tensors:
        weights["lm_head"] = expect_shape(
            "lm_head.weight",
            tensors["lm_head.weight"].T,
            (config.hidden_size, config.vocab_size),
        )

    for index in range(config.num_hidden_layers):
        prefix = f"model.layers.{index}"
        layer = {
            "input_norm": expect_shape(
                f"{prefix}.input_layernorm.weight",
                tensors[f"{prefix}.input_layernorm.weight"],
                (config.hidden_size,),
            ),
            "post_attention_norm": expect_shape(
                f"{prefix}.post_attention_layernorm.weight",
                tensors[f"{prefix}.post_attention_layernorm.weight"],
                (config.hidden_size,),
            ),
            "q_norm": expect_shape(
                f"{prefix}.self_attn.q_norm.weight",
                tensors[f"{prefix}.self_attn.q_norm.weight"],
                (config.head_dim,),
            ),
            "k_norm": expect_shape(
                f"{prefix}.self_attn.k_norm.weight",
                tensors[f"{prefix}.self_attn.k_norm.weight"],
                (config.head_dim,),
            ),
            "q_proj": expect_shape(
                f"{prefix}.self_attn.q_proj.weight",
                tensors[f"{prefix}.self_attn.q_proj.weight"].T,
                (config.hidden_size, q_out),
            ),
            "k_proj": expect_shape(
                f"{prefix}.self_attn.k_proj.weight",
                tensors[f"{prefix}.self_attn.k_proj.weight"].T,
                (config.hidden_size, kv_out),
            ),
            "v_proj": expect_shape(
                f"{prefix}.self_attn.v_proj.weight",
                tensors[f"{prefix}.self_attn.v_proj.weight"].T,
                (config.hidden_size, kv_out),
            ),
            "o_proj": expect_shape(
                f"{prefix}.self_attn.o_proj.weight",
                tensors[f"{prefix}.self_attn.o_proj.weight"].T,
                (q_out, config.hidden_size),
            ),
            "gate_proj": expect_shape(
                f"{prefix}.mlp.gate_proj.weight",
                tensors[f"{prefix}.mlp.gate_proj.weight"].T,
                (config.hidden_size, config.intermediate_size),
            ),
            "up_proj": expect_shape(
                f"{prefix}.mlp.up_proj.weight",
                tensors[f"{prefix}.mlp.up_proj.weight"].T,
                (config.hidden_size, config.intermediate_size),
            ),
            "down_proj": expect_shape(
                f"{prefix}.mlp.down_proj.weight",
                tensors[f"{prefix}.mlp.down_proj.weight"].T,
                (config.intermediate_size, config.hidden_size),
            ),
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
    if "lm_head" in weights:
        return hidden_states @ weights["lm_head"]
    return hidden_states @ weights["embed_tokens"].T


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
    eos_token_ids = config.eos_token_id
    if isinstance(eos_token_ids, int):
        eos_token_ids = (eos_token_ids,)

    for _ in range(args.max_new_tokens):
        ids = jax.device_put(jnp.asarray(tokens, dtype=jnp.int32), device)
        logits = np.asarray(forward(weights, ids))
        next_token = sample_next_token(logits, rng, args.temperature, args.top_k)
        tokens = np.append(tokens, np.int32(next_token))
        if eos_token_ids is not None and next_token in eos_token_ids:
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
    if args.max_new_tokens < 0:
        raise SystemExit("--max-new-tokens must be non-negative")
    if args.max_seq_len is not None and args.max_seq_len <= 0:
        raise SystemExit("--max-seq-len must be positive")
    if args.temperature < 0:
        raise SystemExit("--temperature must be non-negative")

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

    device = select_device(args.backend)
    weights = jax.device_put(weights_host, device)
    forward = make_forward(config, device)
    if args.warmup:
        _, warmup_seconds = timed_generate(config, weights, device, input_ids, args, forward)
    else:
        warmup_seconds = 0.0
    output_ids, run_seconds = timed_generate(config, weights, device, input_ids, args, forward)

    print(f"device: {device}")
    if not args.random_weights:
        print(f"model: {args.model}")
    print(f"parameters: {count_parameters(weights):,}")
    print(f"warmup_seconds: {warmup_seconds:.3f}")
    print(f"run_seconds: {run_seconds:.3f}")
    print(tokenizer.decode(output_ids))


if __name__ == "__main__":
    main()
