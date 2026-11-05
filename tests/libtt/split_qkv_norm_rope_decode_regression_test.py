"""PyTorch's decode attention prologue (SGLang-TorchTPU's Qwen3): the fused QKV
projection split into heads, the q and k heads normalized and rotated, q padded
to the width of the KV cache, which holds k and v side by side."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

HEAD_DIM = 128


def _rms_norm(x, weight):
    """SGLang's RMSNorm as SGLang-TorchTPU traces it: in FP32, back to BF16."""
    x = x.astype(jnp.float32)
    variance = jnp.mean(jnp.power(x, 2.0), axis=-1, keepdims=True)
    return (x * jax.lax.rsqrt(variance + 1e-6) * weight.astype(jnp.float32)).astype(jnp.bfloat16)


def _rotate(x, cos_sin, tokens, heads):
    """rotate_half rotary embedding of the heads' rows [tokens * heads, 128],
    with one cos and sin row per token, as SGLang-TorchTPU traces it."""
    half = HEAD_DIM // 2
    x1 = x[:, :half].reshape(tokens, heads, half)
    x2 = x[:, half:].reshape(tokens, heads, half)
    c, s = cos_sin[:, None, :half], cos_sin[:, None, half:]
    return x1 * c - x2 * s, x2 * c + x1 * s


def _norms(qkv, q_weight, k_weight, heads, kv_heads):
    tokens = qkv.shape[0]
    q = qkv[:, : heads * HEAD_DIM].reshape(tokens * heads, HEAD_DIM)
    k = qkv[:, heads * HEAD_DIM : (heads + kv_heads) * HEAD_DIM].reshape(tokens * kv_heads, HEAD_DIM)
    return _rms_norm(q, q_weight), _rms_norm(k, k_weight)


def _rotations(q, k, v, cos_sin, tokens, heads, kv_heads):
    """The rotated q, padded to the cache's width, and each token's k and v
    side by side, as the paged attention takes them."""
    q = jnp.concatenate(_rotate(q, cos_sin, tokens, heads), axis=-1)
    q = jnp.pad(q, ((0, 0), (0, 0), (0, HEAD_DIM))).reshape(1, tokens, heads, 2 * HEAD_DIM)
    kv = jnp.concatenate([*_rotate(k, cos_sin, tokens, kv_heads), v], axis=-1)
    return q, [kv[b : b + 1].reshape(1, 1, kv_heads, 2 * HEAD_DIM) for b in range(tokens)]


@pytest.mark.parametrize(
    "tokens,heads,kv_heads,trace",
    [
        # Qwen3-8B on one chip and four, one and four tokens.
        (1, 32, 8, False),
        (1, 8, 2, False),
        (4, 32, 8, False),
        (4, 8, 2, True),
    ],
)
def test_torch_qkv_norm_rope_prologue(tokens, heads, kv_heads, trace, tmp_path):
    """The prologue runs as one split_qkv_norm_rope_decode and equals the
    separate ops (rms_norm, rotary_embedding, pad and concat) bit for bit."""
    options = {"optimization_level": "O0", "enable_trace": str(trace).lower()}
    width = (heads + 2 * kv_heads) * HEAD_DIM

    def prologue(qkv, q_weight, k_weight, cos_sin):
        q, k = _norms(qkv, q_weight, k_weight, heads, kv_heads)
        v = qkv[:, (heads + kv_heads) * HEAD_DIM :].reshape(tokens, kv_heads, HEAD_DIM)
        return _rotations(q, k, v, cos_sin, tokens, heads, kv_heads)

    fused = jax.jit(prologue, compiler_options={**options, "export_path": str(tmp_path)})
    # Apart: the norms in one program, the rotations in another.
    norms = jax.jit(lambda *a: _norms(*a, heads, kv_heads), compiler_options=options)
    rotations = jax.jit(lambda *a: _rotations(*a, tokens, heads, kv_heads), compiler_options=options)
    device = jax.devices("tt")[0]
    rng = np.random.default_rng(tokens * 100 + heads)
    for step in range(3):  # With a trace: warmup, capture, and replay with new inputs.
        qkv = rng.normal(0, 2, (tokens, width)) * np.exp2(rng.integers(-4, 5, (tokens, width)))
        angles = rng.uniform(-3, 3, (tokens, HEAD_DIM // 2))
        cos_sin = np.concatenate([np.cos(angles), np.sin(angles)], axis=-1)
        args = [
            jax.device_put(jnp.asarray(a, jnp.bfloat16), device)
            for a in (qkv, rng.normal(1, 0.3, HEAD_DIM), rng.normal(1, 0.3, HEAD_DIM), cos_sin)
        ]
        q, kvs = fused(*args)
        q_norm, k_norm = norms(*args[:3])
        v = jax.device_put(jnp.asarray(qkv[:, (heads + kv_heads) * HEAD_DIM :], jnp.bfloat16), device)
        want_q, want_kvs = rotations(q_norm, k_norm, v.reshape(tokens, kv_heads, HEAD_DIM), args[3])
        np.testing.assert_array_equal(np.asarray(q, np.float32), np.asarray(want_q, np.float32))
        for b in range(tokens):
            np.testing.assert_array_equal(np.asarray(kvs[b], np.float32), np.asarray(want_kvs[b], np.float32))
        if step == 0:
            irs = [path.read_text() for path in (tmp_path / "irs").glob("ttnn_[0-9]*.mlir")]
            assert any('"ttnn.split_qkv_norm_rope_decode"' in ir for ir in irs), "the prologue was not fused"
            assert not any('"ttnn.rotary_embedding"' in ir for ir in irs), "a rotary embedding was left"
        if not trace:
            break
