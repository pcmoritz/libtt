"""PyTorch's decode attention prologue (SGLang-TorchTPU's Qwen3 and Qwen3.8): the
fused QKV projection split into heads, the q and k heads normalized and rotated,
and k and v side by side as the KV cache holds them. Decode attention takes q as
it is; prefill attention takes it padded to the cache's width.

Qwen3.8's attention gates its output: each q head is followed by its gate head
in the projection. Its norms are zero-centered (weight 1 + w, in FP32), and its
rotary embedding rotates the first 64 of 256 dims."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# head_dim, gated q, rotated dims, zero-centered norm weights.
QWEN3 = (128, False, 128, False)
QWEN38 = (256, True, 64, True)


def _rms_norm(x, weight, zero_centered):
    """SGLang's RMSNorm (GemmaRMSNorm if zero_centered) as SGLang-TorchTPU
    traces it: in FP32, back to BF16."""
    x = x.astype(jnp.float32)
    variance = jnp.mean(jnp.power(x, 2.0), axis=-1, keepdims=True)
    weight = weight.astype(jnp.float32)
    if zero_centered:
        weight = 1.0 + weight
    return (x * jax.lax.rsqrt(variance + 1e-6) * weight).astype(jnp.bfloat16)


def _rotate(x, cos_sin, tokens, heads, rotary):
    """rotate_half rotary embedding of the first `rotary` dims of the heads'
    rows [tokens * heads, head_dim], with one cos and sin row per token, as
    SGLang-TorchTPU traces it: the rotated halves, then any other dims."""
    half = rotary // 2
    x1 = x[:, :half].reshape(tokens, heads, half)
    x2 = x[:, half:rotary].reshape(tokens, heads, half)
    c, s = cos_sin[:, None, :half], cos_sin[:, None, half:]
    rest = [x[:, rotary:].reshape(tokens, heads, -1)] if rotary < x.shape[1] else []
    return [x1 * c - x2 * s, x2 * c + x1 * s, *rest]


def _norms(qkv, q_weight, k_weight, heads, kv_heads, model):
    head_dim, gated, _, zero_centered = model
    tokens = qkv.shape[0]
    q_width = (2 if gated else 1) * heads * head_dim
    q = qkv[:, :q_width].reshape(tokens, heads, -1)[..., :head_dim].reshape(tokens * heads, head_dim)
    k = qkv[:, q_width : q_width + kv_heads * head_dim].reshape(tokens * kv_heads, head_dim)
    return _rms_norm(q, q_weight, zero_centered), _rms_norm(k, k_weight, zero_centered)


def _rotations(q, k, v, cos_sin, tokens, heads, kv_heads, model, pad):
    """The rotated q, padded to the cache's width if `pad`, and all tokens' k
    and v side by side, as one paged cache update takes them."""
    head_dim, _, rotary, _ = model
    q = jnp.concatenate(_rotate(q, cos_sin, tokens, heads, rotary), axis=-1)
    if pad:
        q = jnp.pad(q, ((0, 0), (0, 0), (0, head_dim)))
    q = q.reshape(1, tokens, heads, -1)
    kv = jnp.concatenate([*_rotate(k, cos_sin, tokens, kv_heads, rotary), v], axis=-1)
    return q, kv.reshape(1, tokens, kv_heads, 2 * head_dim)


@pytest.mark.parametrize("pad", [False, True], ids=["q", "padded_q"])
@pytest.mark.parametrize(
    "model,tokens,heads,kv_heads,trace",
    [
        # Qwen3-8B on one chip and four, one and four tokens.
        (QWEN3, 1, 32, 8, False),
        (QWEN3, 1, 8, 2, False),
        (QWEN3, 4, 32, 8, False),
        (QWEN3, 4, 8, 2, True),
        # Qwen3.8-27B on four chips and one, one and four tokens.
        (QWEN38, 1, 6, 1, False),
        (QWEN38, 4, 6, 1, True),
        (QWEN38, 4, 24, 4, False),
    ],
)
def test_torch_qkv_norm_rope_prologue(model, tokens, heads, kv_heads, trace, pad, tmp_path):
    """The prologue runs as one split_qkv_norm_rope_decode and equals the
    separate ops (rms_norm, rotary_embedding, any pad and concat) bit for bit."""
    head_dim, gated, rotary, _ = model
    options = {"optimization_level": "O0", "enable_trace": str(trace).lower()}
    v_begin = ((2 if gated else 1) * heads + kv_heads) * head_dim
    width = v_begin + kv_heads * head_dim

    def prologue(qkv, q_weight, k_weight, cos_sin):
        q, k = _norms(qkv, q_weight, k_weight, heads, kv_heads, model)
        v = qkv[:, v_begin:].reshape(tokens, kv_heads, head_dim)
        return _rotations(q, k, v, cos_sin, tokens, heads, kv_heads, model, pad)

    fused = jax.jit(prologue, compiler_options={**options, "export_path": str(tmp_path)})
    # Apart: the norms in one program, the rotations in another.
    norms = jax.jit(lambda *a: _norms(*a, heads, kv_heads, model), compiler_options=options)
    rotations = jax.jit(lambda *a: _rotations(*a, tokens, heads, kv_heads, model, pad), compiler_options=options)
    device = jax.devices("tt")[0]
    rng = np.random.default_rng(tokens * 100 + heads)
    for step in range(3):  # With a trace: warmup, capture, and replay with new inputs.
        qkv = rng.normal(0, 2, (tokens, width)) * np.exp2(rng.integers(-4, 5, (tokens, width)))
        angles = rng.uniform(-3, 3, (tokens, rotary // 2))
        cos_sin = np.concatenate([np.cos(angles), np.sin(angles)], axis=-1)
        args = [
            jax.device_put(jnp.asarray(a, jnp.bfloat16), device)
            for a in (qkv, rng.normal(1, 0.3, head_dim), rng.normal(1, 0.3, head_dim), cos_sin)
        ]
        q, kv = fused(*args)
        q_norm, k_norm = norms(*args[:3])
        v = jax.device_put(jnp.asarray(qkv[:, v_begin:], jnp.bfloat16), device)
        want_q, want_kv = rotations(q_norm, k_norm, v.reshape(tokens, kv_heads, head_dim), args[3])
        np.testing.assert_array_equal(np.asarray(q, np.float32), np.asarray(want_q, np.float32))
        np.testing.assert_array_equal(np.asarray(kv, np.float32), np.asarray(want_kv, np.float32))
        if step == 0:
            irs = [path.read_text() for path in (tmp_path / "irs").glob("ttnn_[0-9]*.mlir")]
            assert any('"ttnn.split_qkv_norm_rope_decode"' in ir for ir in irs), "the prologue was not fused"
            assert not any('"ttnn.rotary_embedding"' in ir for ir in irs), "a rotary embedding was left"
        if not trace:
            break
