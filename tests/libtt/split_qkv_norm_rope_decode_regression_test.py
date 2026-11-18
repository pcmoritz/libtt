"""PyTorch's decode attention prologue (SGLang-TorchTPU's Qwen3 and Qwen3.8): the
fused QKV projection split into heads, the q and k heads normalized and rotated,
and k and v side by side as the KV cache holds them. Decode attention takes q as
it is; prefill attention takes it padded to the cache's width. No model crops
it or regroups its heads, but neither may reach the fused kernel, which writes
at least head_dim columns and each head's rotated dims next to its others.

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


def _rotate(x, cos_sin, tokens, heads, rotary, regroup=False):
    """rotate_half rotary embedding of the first `rotary` dims of the heads'
    rows [tokens * heads, head_dim], with one cos and sin row per token, as
    SGLang-TorchTPU traces it: the rotated halves, then any other dims. If
    `regroup`, all heads' rotated dims come first, then all heads' others."""
    half = rotary // 2
    x1 = x[:, :half].reshape(tokens, heads, half)
    x2 = x[:, half:rotary].reshape(tokens, heads, half)
    c, s = cos_sin[:, None, :half], cos_sin[:, None, half:]
    rest = [x[:, rotary:].reshape(tokens, heads, -1)] if rotary < x.shape[1] else []
    if regroup:
        rotated = jnp.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], axis=-1)
        return [part.reshape(tokens, 1, -1) for part in (rotated, *rest)]
    return [x1 * c - x2 * s, x2 * c + x1 * s, *rest]


def _norms(qkv, q_weight, k_weight, heads, kv_heads, model):
    head_dim, gated, _, zero_centered = model
    tokens = qkv.shape[0]
    q_width = (2 if gated else 1) * heads * head_dim
    q = qkv[:, :q_width].reshape(tokens, heads, -1)[..., :head_dim].reshape(tokens * heads, head_dim)
    k = qkv[:, q_width : q_width + kv_heads * head_dim].reshape(tokens * kv_heads, head_dim)
    return _rms_norm(q, q_weight, zero_centered), _rms_norm(k, k_weight, zero_centered)


def _rotations(q, k, v, cos_sin, tokens, heads, kv_heads, model, q_form):
    """The rotated q (as it is, padded to the cache's width, cropped by a tile
    with a negative pad, or with its heads regrouped) and all tokens' k and v
    side by side, as one paged cache update takes them."""
    head_dim, _, rotary, _ = model
    regroup = q_form == "regrouped_q"
    q = jnp.concatenate(_rotate(q, cos_sin, tokens, heads, rotary, regroup), axis=-1)
    high = {"q": 0, "padded_q": head_dim, "cropped_q": -32, "regrouped_q": 0}[q_form]
    if high:
        q = jax.lax.pad(q, jnp.zeros((), q.dtype), [(0, 0, 0), (0, 0, 0), (0, high, 0)])
    q = q.reshape(1, tokens, heads, -1)
    kv = jnp.concatenate([*_rotate(k, cos_sin, tokens, kv_heads, rotary), v], axis=-1)
    return q, kv.reshape(1, tokens, kv_heads, 2 * head_dim)


@pytest.mark.parametrize("q_form", ["q", "padded_q", "cropped_q", "regrouped_q"])
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
def test_torch_qkv_norm_rope_prologue(model, tokens, heads, kv_heads, trace, q_form, tmp_path):
    """The prologue runs as one split_qkv_norm_rope_decode (unless q is cropped
    or regrouped) and equals the separate ops (rms_norm, rotary_embedding, any
    pad and concat) bit for bit."""
    _check_prologue(model, tokens, heads, kv_heads, trace, q_form, tmp_path)


@pytest.mark.parametrize("tokens", [1, 4, 32])
def test_prologue_with_physical_layouts(tokens, tmp_path, monkeypatch):
    """With one KV head, k and v are [tokens, 1, 2 * head_dim] between the
    rotary embedding and the cache, which ttir-physical-layouts would store
    without the unit dim if it may (here: any cluster that saves anything). The
    prologue is one op by then (the TTIR fusion), so it still fuses."""
    monkeypatch.setenv("LIBTT_PHYSICAL_LAYOUTS_MIN_MB", "0")
    _check_prologue(QWEN38, tokens, 6, 1, False, "q", tmp_path)


def _check_prologue(model, tokens, heads, kv_heads, trace, q_form, tmp_path):
    head_dim, gated, rotary, _ = model
    if q_form == "regrouped_q" and (rotary == head_dim or heads == 1):
        pytest.skip("regrouping needs a partial rotary embedding and two q heads")
    options = {"optimization_level": "O0", "enable_trace": str(trace).lower()}
    v_begin = ((2 if gated else 1) * heads + kv_heads) * head_dim
    width = v_begin + kv_heads * head_dim

    def prologue(qkv, q_weight, k_weight, cos_sin):
        q, k = _norms(qkv, q_weight, k_weight, heads, kv_heads, model)
        v = qkv[:, v_begin:].reshape(tokens, kv_heads, head_dim)
        return _rotations(q, k, v, cos_sin, tokens, heads, kv_heads, model, q_form)

    fused = jax.jit(prologue, compiler_options={**options, "export_path": str(tmp_path)})
    # Apart: the norms in one program, the rotations in another.
    norms = jax.jit(lambda *a: _norms(*a, heads, kv_heads, model), compiler_options=options)
    rotations = jax.jit(lambda *a: _rotations(*a, tokens, heads, kv_heads, model, q_form), compiler_options=options)
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
            fused_op = any('"ttnn.split_qkv_norm_rope_decode"' in ir for ir in irs)
            if q_form in ("cropped_q", "regrouped_q"):
                assert not fused_op, f"a {q_form.split('_')[0]} q was fused"
            else:
                assert fused_op, "the prologue was not fused"
                assert not any('"ttnn.rotary_embedding"' in ir for ir in irs), "a rotary embedding was left"
        if not trace:
            break
