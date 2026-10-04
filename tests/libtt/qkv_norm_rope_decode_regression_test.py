"""The decode attention prologue of Qwen3 runs as one program.

SGLang-JAX projects q, k and v, normalizes the q and k heads and applies the
rotary embedding; tt-mlir turns that into nlp_create_qkv_heads_decode, two
rms_norms and two rotary_embeddings, which now become a single
nlp_create_qkv_heads_decode_norm_rope. Its q, k and v must match NumPy.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

HIDDEN = 1024
HEAD_DIM = 128
EPS = 1e-6


def rms_norm(x, weight):
    # sglang-jax's RMSNorm: statistics in FP32, the output in the input dtype.
    xf = x.astype(jnp.float32)
    var = jnp.mean(lax.square(xf), axis=-1, keepdims=True)
    return (xf * lax.rsqrt(var + EPS) * weight.astype(jnp.float32)).astype(x.dtype)


def rotary(x, cos, sin):
    # sglang-jax's NeoX-style apply_rotary_emb on [tokens, heads, head_dim].
    cos, sin = cos[:, None, :].astype(x.dtype), sin[:, None, :].astype(x.dtype)
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate((x1 * cos - x2 * sin, x2 * cos + x1 * sin), axis=-1)


def make_prologue(heads, kv_heads):
    inv_freq = 1.0 / (1e6 ** (np.arange(0, HEAD_DIM, 2, dtype=np.float32) / HEAD_DIM))

    def prologue(x, wq, wk, wv, q_norm, k_norm, positions):
        with jax.named_scope("q_proj"):
            q = x @ wq
        with jax.named_scope("k_proj"):
            k = x @ wk
        with jax.named_scope("v_proj"):
            v = x @ wv
        q = rms_norm(q.reshape(-1, heads, HEAD_DIM), q_norm)
        k = rms_norm(k.reshape(-1, kv_heads, HEAD_DIM), k_norm)
        freqs = positions.astype(jnp.float32)[:, None] * inv_freq
        cos, sin = jnp.cos(freqs), jnp.sin(freqs)
        q, k = rotary(q, cos, sin), rotary(k, cos, sin)
        # The decode attention takes [1, batch, heads, head_dim].
        return (
            q.reshape(1, 1, heads, HEAD_DIM),
            k.reshape(1, 1, kv_heads, HEAD_DIM),
            v.reshape(1, 1, kv_heads, HEAD_DIM),
        )

    return prologue


def reference(x, wq, wk, wv, q_norm, k_norm, position, heads, kv_heads):
    x, wq, wk, wv, q_norm, k_norm = (a.astype(np.float32) for a in (x, wq, wk, wv, q_norm, k_norm))
    freqs = position * (1.0 / (1e6 ** (np.arange(0, HEAD_DIM, 2, dtype=np.float32) / HEAD_DIM)))
    cos, sin = np.cos(freqs), np.sin(freqs)

    def norm_rope(h, weight):
        h = h * (1.0 / np.sqrt(np.mean(h * h, axis=-1, keepdims=True) + EPS)) * weight
        h1, h2 = h[..., : HEAD_DIM // 2], h[..., HEAD_DIM // 2 :]
        return np.concatenate((h1 * cos - h2 * sin, h2 * cos + h1 * sin), axis=-1)

    q = norm_rope((x @ wq).reshape(heads, HEAD_DIM), q_norm)
    k = norm_rope((x @ wk).reshape(kv_heads, HEAD_DIM), k_norm)
    v = (x @ wv).reshape(kv_heads, HEAD_DIM)
    return q, k, v


@pytest.mark.parametrize("trace", [False, True])
# Qwen3-8B at TP4, TP2 and TP1, Qwen3-14B and Qwen3-32B at TP4.
@pytest.mark.parametrize("heads,kv_heads", [(8, 2), (16, 4), (32, 8), (10, 2), (16, 2)])
def test_qkv_norm_rope_decode(trace, heads, kv_heads, tmp_path):
    run = jax.jit(
        make_prologue(heads, kv_heads),
        compiler_options={
            "optimization_level": "O1",
            "enable_trace": str(trace).lower(),
            "export_path": str(tmp_path),
        },
    )
    device = jax.devices("tt")[0]
    rng = np.random.default_rng(heads)
    weights = [
        (rng.standard_normal((HIDDEN, n * HEAD_DIM)) / np.sqrt(HIDDEN)).astype(jnp.bfloat16)
        for n in (heads, kv_heads, kv_heads)
    ]
    norms = [rng.uniform(0.5, 1.5, HEAD_DIM).astype(jnp.bfloat16) for _ in range(2)]
    on_device = [jax.device_put(a, device) for a in weights + norms]
    for position in (5, 1000, 31):  # Warmup, capture, and replay with changed inputs.
        x = rng.standard_normal((1, HIDDEN)).astype(jnp.bfloat16)
        positions = np.array([position], dtype=np.int32)
        q, k, v = run(jax.device_put(x, device), *on_device, jax.device_put(positions, device))
        expected = reference(x, *weights, *norms, position, heads, kv_heads)
        for name, actual, want in zip("qkv", (q, k, v), expected):
            actual = np.asarray(actual).astype(np.float32).reshape(want.shape)
            np.testing.assert_allclose(actual, want, atol=0.06, rtol=0.03, err_msg=f"{name} at {position}")

    irs = [path.read_text() for path in (tmp_path / "irs").glob("ttnn_[0-9]*.mlir")]
    assert irs, "no IR was exported"
    assert any("ttnn.nlp_create_qkv_heads_decode_norm_rope" in ir for ir in irs), "the prologue was not fused"
    assert not any("ttnn.rotary_embedding" in ir for ir in irs), "a rotary embedding was left over"
