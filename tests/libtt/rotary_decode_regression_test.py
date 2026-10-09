"""Decode rotary embedding must preserve heads smaller than a tile."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize("trace", [False, True])
@pytest.mark.parametrize("heads", [4, 16, 32])
def test_rotary_decode_logical_heads(trace, heads):
    def decode(q, k, v, cos, sin, table, positions):
        rotated = jnp.concatenate((-q[..., 64:], q[..., :64]), axis=-1)
        q = (q * cos + rotated * sin).transpose(0, 2, 1, 3)
        out = jax.ffi.ffi_call(
            "tt.paged_scaled_dot_product_attention_decode",
            jax.ShapeDtypeStruct(q.shape, q.dtype),
            vmap_method="sequential",
        )(
            q,
            k,
            v,
            table,
            positions,
            is_causal=True,
            has_attention_mask=False,
            has_cur_pos_tensor=True,
            has_attention_sink=False,
        )
        # The fused rotary must not expose its head padding to this reshape.
        return out.reshape(1, heads * 128)

    run = jax.jit(
        decode,
        compiler_options={"optimization_level": "O1", "enable_trace": str(trace).lower()},
    )
    device = jax.devices("tt")[0]
    rng = np.random.default_rng(7)
    table = np.arange(4, dtype=np.int32).reshape(1, 4)
    positions = np.array([31], dtype=np.int32)
    for _ in range(3):  # Warmup, capture, and replay with changed inputs.
        q = rng.normal(0, 0.2, (1, heads, 1, 128)).astype(jnp.bfloat16)
        k, v = (rng.normal(0, 0.2, (4, 4, 32, 128)).astype(jnp.bfloat16) for _ in range(2))
        angles = np.tile(rng.uniform(-3, 3, (1, 1, 1, 64)), 2)
        cos, sin = (f(angles).astype(jnp.bfloat16) for f in (np.cos, np.sin))
        actual = run(*(jax.device_put(x, device) for x in (q, k, v, cos, sin, table, positions)))

        q, k, v, cos, sin = (x.astype(np.float32) for x in (q, k, v, cos, sin))
        q = q * cos + np.concatenate((-q[..., 64:], q[..., :64]), axis=-1) * sin
        q = q.astype(jnp.bfloat16).astype(np.float32).reshape(heads, 128)
        keys, values = (np.repeat(x[0], heads // 4, axis=0) for x in (k, v))
        scores = np.einsum("hd,hsd->hs", q, keys) / np.sqrt(128)
        scores = np.exp(scores - scores.max(axis=-1, keepdims=True))
        scores /= scores.sum(axis=-1, keepdims=True)
        expected = np.einsum("hs,hsd->hd", scores, values).reshape(1, -1)
        np.testing.assert_allclose(np.asarray(actual).astype(np.float32), expected, atol=0.004, rtol=0.02)


@pytest.mark.parametrize("trace", [False, True])
@pytest.mark.parametrize("heads", [1, 6, 32])
@pytest.mark.parametrize("tail", ["slice", "independent"])
def test_partial_rotary_decode(tmp_path, trace, heads, tail):
    """Qwen3.5 rotates the first 64 of 256 head dims: x[..., :64] by
    (x1 cos - x2 sin, x2 cos + x1 sin) with x[..., 64:] appended. The rotated
    part becomes one fused rotary; the tail is concatenated unchanged, also
    when it is not a slice of the rotated tensor."""

    def decode(x, cos, sin):
        rotary = x[..., :64]
        x1, x2 = rotary[..., :32], rotary[..., 32:]
        c, s = cos[:, None, :], sin[:, None, :]
        rest = x[..., 64:] if tail == "slice" else jnp.flip(x[..., 64:], axis=-1) * 2
        out = jnp.concatenate([x1 * c - x2 * s, x2 * c + x1 * s, rest], axis=-1)
        return out.reshape(1, 1, heads, 256)

    run = jax.jit(
        decode,
        compiler_options={
            "optimization_level": "O1",
            "enable_trace": str(trace).lower(),
            "export_path": str(tmp_path),
        },
    )
    device = jax.devices("tt")[0]
    rng = np.random.default_rng(11)
    for step in range(3):  # Warmup, capture, and replay with changed inputs.
        x = rng.normal(0, 1, (1, heads, 256)).astype(jnp.bfloat16)
        angles = rng.uniform(-3, 3, (1, 32))
        cos, sin = (f(angles).astype(jnp.bfloat16) for f in (np.cos, np.sin))
        actual = np.asarray(run(*(jax.device_put(a, device) for a in (x, cos, sin)))).astype(np.float32)
        if step == 0:
            irs = [path.read_text() for path in (tmp_path / "irs").glob("ttnn_[0-9]*.mlir")]
            assert any("ttnn.rotary_embedding" in ir for ir in irs)
        xf, c, s = (a.astype(np.float32) for a in (x, cos, sin))
        x1, x2 = xf[..., :32], xf[..., 32:64]
        rest = xf[..., 64:] if tail == "slice" else np.flip(xf[..., 64:], axis=-1) * 2
        expected = np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s, rest], axis=-1)
        np.testing.assert_allclose(actual.reshape(1, heads, 256), expected, atol=0.05, rtol=0.02)


@pytest.mark.parametrize("trace", [False, True])
@pytest.mark.parametrize("heads", [1, 6])
def test_partial_rotary_rank4(tmp_path, trace, heads):
    """The partial rotary on rank-4 input [1, heads, 1, 256], with no reshape
    after the concat: the fused rotary replaces the concat itself."""

    def rotate(x, cos, sin):
        rotary = x[..., :64]
        x1, x2 = rotary[..., :32], rotary[..., 32:]
        return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin, x[..., 64:]], axis=-1)

    run = jax.jit(
        rotate,
        compiler_options={
            "optimization_level": "O1",
            "enable_trace": str(trace).lower(),
            "export_path": str(tmp_path),
        },
    )
    device = jax.devices("tt")[0]
    rng = np.random.default_rng(13)
    for step in range(3):
        x = rng.normal(0, 1, (1, heads, 1, 256)).astype(jnp.bfloat16)
        angles = rng.uniform(-3, 3, (1, 1, 1, 32))
        cos, sin = (f(angles).astype(jnp.bfloat16) for f in (np.cos, np.sin))
        actual = np.asarray(run(*(jax.device_put(a, device) for a in (x, cos, sin)))).astype(np.float32)
        if step == 0:
            irs = [path.read_text() for path in (tmp_path / "irs").glob("ttnn_[0-9]*.mlir")]
            assert any("ttnn.rotary_embedding" in ir for ir in irs)
        xf, c, s = (a.astype(np.float32) for a in (x, cos, sin))
        x1, x2 = xf[..., :32], xf[..., 32:64]
        expected = np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s, xf[..., 64:]], axis=-1)
        np.testing.assert_allclose(actual, expected, atol=0.05, rtol=0.02)


@pytest.mark.parametrize("trace", [False, True])
@pytest.mark.parametrize("tokens,heads", [(3, 1), (8, 8), (32, 6), (33, 6), (129, 65)])
@pytest.mark.parametrize("rotary", [128, 64])
def test_rotary_batched_decode(tmp_path, trace, tokens, heads, rotary):
    """A decode batch, each token at its own position: x [tokens, heads, 128]
    rotated by its token's cos and sin row, then reshaped to
    [1, tokens, heads, 128] for attention. It becomes one fused rotary over the
    tokens; with a partial rotary (Qwen3.5) the tail is appended unchanged.
    33 tokens span two cache tiles, the second partly filled; 65 heads take three
    tile rows per token, and 129 tokens several rows per core, which read a
    token's cache tiles again and cross from one token to the next."""

    half = rotary // 2

    def decode(x, cos, sin):
        x1, x2 = x[..., :rotary][..., :half], x[..., :rotary][..., half:]
        c, s = cos[:, None, :], sin[:, None, :]
        parts = [x1 * c - x2 * s, x2 * c + x1 * s] + ([x[..., rotary:]] if rotary < 128 else [])
        return jnp.concatenate(parts, axis=-1).reshape(1, tokens, heads, 128)

    run = jax.jit(
        decode,
        compiler_options={
            "optimization_level": "O1",
            "enable_trace": str(trace).lower(),
            "export_path": str(tmp_path),
        },
    )
    device = jax.devices("tt")[0]
    rng = np.random.default_rng(17)
    for step in range(3):  # Warmup, capture, and replay with changed inputs.
        x = rng.normal(0, 1, (tokens, heads, 128)).astype(jnp.bfloat16)
        angles = rng.uniform(-3, 3, (tokens, half))
        cos, sin = (f(angles).astype(jnp.bfloat16) for f in (np.cos, np.sin))
        actual = np.asarray(run(*(jax.device_put(a, device) for a in (x, cos, sin)))).astype(np.float32)
        if step == 0:
            irs = [path.read_text() for path in (tmp_path / "irs").glob("ttnn_[0-9]*.mlir")]
            assert any("ttnn.rotary_embedding" in ir for ir in irs)
        xf, c, s = (a.astype(np.float32) for a in (x, cos, sin))
        x1, x2 = xf[..., :half], xf[..., half:rotary]
        c, s = c[:, None, :], s[:, None, :]
        expected = np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s, xf[..., rotary:]], axis=-1)
        np.testing.assert_allclose(actual.reshape(tokens, heads, 128), expected, atol=0.05, rtol=0.02)


@pytest.mark.parametrize("trace", [False, True])
def test_rotary_decode_mixed_cache_broadcasts(tmp_path, trace):
    """With tokens == heads, a cache broadcast per token in one rotation half
    and per head in the other has the same shapes as the batched decode
    rotary but is not a rotary: it must not be fused, and is computed as
    written."""

    n = 8

    def decode(x, cos, sin):
        x1, x2 = x[..., :64], x[..., 64:]
        lo = x1 * cos[:, None, :] - x2 * sin[:, None, :]
        hi = x2 * cos[None, :, :] + x1 * sin[:, None, :]
        return jnp.concatenate([lo, hi], axis=-1).reshape(1, n, n, 128)

    run = jax.jit(
        decode,
        compiler_options={
            "optimization_level": "O1",
            "enable_trace": str(trace).lower(),
            "export_path": str(tmp_path),
        },
    )
    device = jax.devices("tt")[0]
    rng = np.random.default_rng(19)
    for step in range(3):  # Warmup, capture, and replay with changed inputs.
        x = rng.normal(0, 1, (n, n, 128)).astype(jnp.bfloat16)
        angles = rng.uniform(-3, 3, (n, 64))
        cos, sin = (f(angles).astype(jnp.bfloat16) for f in (np.cos, np.sin))
        actual = np.asarray(run(*(jax.device_put(a, device) for a in (x, cos, sin)))).astype(np.float32)
        if step == 0:
            irs = [path.read_text() for path in (tmp_path / "irs").glob("ttnn_[0-9]*.mlir")]
            assert not any("ttnn.rotary_embedding" in ir for ir in irs)
        xf, c, s = (a.astype(np.float32) for a in (x, cos, sin))
        x1, x2 = xf[..., :64], xf[..., 64:]
        lo = x1 * c[:, None, :] - x2 * s[:, None, :]
        hi = x2 * c[None, :, :] + x1 * s[:, None, :]
        expected = np.concatenate([lo, hi], axis=-1)
        np.testing.assert_allclose(actual.reshape(n, n, 128), expected, atol=0.05, rtol=0.02)


@pytest.mark.parametrize("trace", [False, True])
@pytest.mark.parametrize("rotary", [128, 64])
def test_rotary_prefill(tmp_path, trace, rotary):
    """A prefill, x [1, heads, seq, 128] rotated position by position by caches
    [1, 1, seq, half]: rotary_embedding's sequence mode, which streams one cache
    tile per input tile. 65 positions take three tile rows, the last partly
    filled, and six heads repeat them; with a partial rotary (Qwen3.5) the tail
    is appended unchanged."""

    half = rotary // 2

    def prefill(x, cos, sin):
        x1, x2 = x[..., :rotary][..., :half], x[..., :rotary][..., half:]
        parts = [x1 * cos - x2 * sin, x2 * cos + x1 * sin] + ([x[..., rotary:]] if rotary < 128 else [])
        return jnp.concatenate(parts, axis=-1)

    run = jax.jit(
        prefill,
        compiler_options={
            "optimization_level": "O1",
            "enable_trace": str(trace).lower(),
            "export_path": str(tmp_path),
        },
    )
    device = jax.devices("tt")[0]
    rng = np.random.default_rng(29)
    for step in range(3):  # Warmup, capture, and replay with changed inputs.
        x = rng.normal(0, 1, (1, 6, 65, 128)).astype(jnp.bfloat16)
        angles = rng.uniform(-3, 3, (1, 1, 65, half))
        cos, sin = (f(angles).astype(jnp.bfloat16) for f in (np.cos, np.sin))
        actual = np.asarray(run(*(jax.device_put(a, device) for a in (x, cos, sin)))).astype(np.float32)
        if step == 0:
            irs = [path.read_text() for path in (tmp_path / "irs").glob("ttnn_[0-9]*.mlir")]
            assert any("ttnn.rotary_embedding" in ir for ir in irs)
        xf, c, s = (a.astype(np.float32) for a in (x, cos, sin))
        x1, x2 = xf[..., :half], xf[..., half:rotary]
        expected = np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s, xf[..., rotary:]], axis=-1)
        np.testing.assert_allclose(actual, expected, atol=0.05, rtol=0.02)
