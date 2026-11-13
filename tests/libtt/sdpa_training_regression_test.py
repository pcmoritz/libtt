"""tt-train's fused attention for training, as JAX custom calls and as TorchTitan's splash attention.

`tt.sdpa_fw` and `tt.sdpa_bw` run ttml::metal::sdpa_fw and sdpa_bw: grouped-query
attention whose forward also returns each row's log-sum-exp (the first column of
[B, H, S, 32] FP32 intermediates) and whose backward takes it back instead of
recomputing the forward. In TPU compatibility mode they replace TorchTitan's
Pallas splash attention (`jax_plugins.libtt.torchtitan_tpu_kernels`), which saves
the same log-sum-exp; packed documents (segment IDs) and sliding windows pass the
kernels a mask, one sequence at a time.
"""

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest

from jax_plugins.libtt import torchtitan_tpu_kernels

HEADS, SEQ, DIM = 8, 256, 128


def _inputs(seed, batch=1, kv_heads=2):
    rng = np.random.default_rng(seed)

    def tensor(heads):
        return rng.standard_normal((batch, heads, SEQ, DIM)).astype(ml_dtypes.bfloat16)

    return tensor(HEADS), tensor(kv_heads), tensor(kv_heads), tensor(HEADS)


def _reference(q, k, v, g, segment_ids=None, window=None):
    """Causal attention in float64: out, logsumexp, dq, dk, dv."""
    q, k, v, g = (x.astype(np.float64) for x in (q, k, v, g))
    batch, kv_heads = k.shape[:2]
    group = q.shape[1] // kv_heads
    kr, vr = np.repeat(k, group, axis=1), np.repeat(v, group, axis=1)
    scale = 1 / np.sqrt(q.shape[-1])
    s = np.einsum("bhqd,bhkd->bhqk", q, kr) * scale
    rows, cols = np.arange(SEQ)[:, None], np.arange(SEQ)[None, :]
    mask = np.broadcast_to(cols <= rows, (batch, 1, SEQ, SEQ))
    if window is not None:
        mask = mask & (cols >= rows - window)
    if segment_ids is not None:
        mask = mask & (segment_ids[:, None, :, None] == segment_ids[:, None, None, :])
    s = np.where(mask, s, -np.inf)
    lse = np.log(np.sum(np.exp(s - s.max(-1, keepdims=True)), -1)) + s.max(-1)
    p = np.exp(s - lse[..., None])
    out = np.einsum("bhqk,bhkd->bhqd", p, vr)
    dp = np.einsum("bhqd,bhkd->bhqk", g, vr)
    ds = p * (dp - np.sum(g * out, -1, keepdims=True)) * scale
    dq = np.einsum("bhqk,bhkd->bhqd", ds, kr)
    dk = np.einsum("bhqk,bhqd->bhkd", ds, q).reshape(batch, kv_heads, group, SEQ, DIM).sum(2)
    dv = np.einsum("bhqk,bhqd->bhkd", p, g).reshape(batch, kv_heads, group, SEQ, DIM).sum(2)
    return out, lse, dq, dk, dv


def _relative_error(got, want):
    got = np.asarray(got).astype(np.float64)
    return np.max(np.abs(got - want)) / np.max(np.abs(want))


def _device(*arrays):
    device = jax.devices("tt")[0]
    return [None if a is None else jax.device_put(a, device) for a in arrays]


def test_custom_calls_match_reference():
    q, k, v, g = _inputs(0)
    out_want, lse_want, dq_want, dk_want, dv_want = _reference(q, k, v, g)

    def forward(q, k, v):
        return jax.ffi.ffi_call(
            "tt.sdpa_fw",
            (jax.ShapeDtypeStruct(q.shape, q.dtype), jax.ShapeDtypeStruct((*q.shape[:3], 32), jnp.float32)),
        )(q, k, v, mask_type=1)

    def backward(g, out, q, k, v, intermediates):
        shapes = tuple(jax.ShapeDtypeStruct(x.shape, x.dtype) for x in (q, k, v))
        return jax.ffi.ffi_call("tt.sdpa_bw", shapes)(g, out, q, k, v, intermediates, mask_type=1)

    tq, tk, tv, tg = _device(q, k, v, g)
    out, intermediates = jax.jit(forward)(tq, tk, tv)
    assert intermediates.shape == (1, HEADS, SEQ, 32) and intermediates.dtype == jnp.float32
    assert _relative_error(out, out_want) < 0.02
    assert _relative_error(np.asarray(intermediates)[..., 0], lse_want) < 2e-3
    dq, dk, dv = jax.jit(backward)(tg, out, tq, tk, tv, intermediates)
    for got, want in ((dq, dq_want), (dk, dk_want), (dv, dv_want)):
        assert _relative_error(got, want) < 0.03


def _segment_ids():
    # Two sequences holding documents of different lengths.
    starts = np.zeros((2, SEQ), np.int32)
    starts[0, [70, 128, 200]] = 1
    starts[1, [33]] = 1
    return np.cumsum(starts, axis=1).astype(np.int32)


@pytest.mark.parametrize(
    "batch,kv_heads,segment_ids,window",
    [(1, 2, None, None), (2, 2, _segment_ids(), None), (1, 1, None, None), (1, 2, None, 100)],
    ids=["causal", "segment_ids", "multi_query", "sliding_window"],
)
def test_splash_attention_functions(batch, kv_heads, segment_ids, window):
    # The functions TorchTitan's splash attention builds its torch ops from:
    # the forward saves (out, logsumexp) and the backward takes them back.
    q, k, v, g = _inputs(1, batch, kv_heads)
    out_want, lse_want, dq_want, dk_want, dv_want = _reference(q, k, v, g, segment_ids, window)
    splash_fn, splash_bwd_fn = torchtitan_tpu_kernels._make_splash_attention_fn(
        seq_len=SEQ, n_heads=HEADS, n_kv_heads=kv_heads, is_causal=True, local_window_size=window, block_q=512
    )
    if kv_heads == 1:
        # Multi-query attention: TorchTitan squeezes the KV head dimension.
        k, v = k[:, 0], v[:, 0]
        dk_want, dv_want = dk_want[:, 0], dv_want[:, 0]
    tq, tk, tv, tg, tsegments = _device(q, k, v, g, segment_ids)
    out, lse = splash_fn(tq, tk, tv, tsegments)
    assert lse.shape == (batch, HEADS, SEQ) and lse.dtype == jnp.float32
    assert _relative_error(out, out_want) < 0.02
    assert _relative_error(lse, lse_want) < 2e-3
    dq, dk, dv = splash_bwd_fn(tq, tk, tv, out, lse, tg, tsegments)
    assert dk.shape == k.shape
    for got, want in ((dq, dq_want), (dk, dk_want), (dv, dv_want)):
        assert _relative_error(got, want) < 0.03
