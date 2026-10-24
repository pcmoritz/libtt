"""Wide, short tensors are computed in chunked, dense tiles.

A [2, W] tensor fills 2 of every 32 rows of its tiles, so elementwise ops over
it, like the sampler's over [batch, vocab] logits, move 16 times its data. The
ttir-dense-rows pass (tt_mlir_dense_rows.patch) computes such regions as
[2, W / 32, 32]: same row-major data, full tiles. These tests compare each
program with the pass on and off (LIBTT_DENSE_ROWS=0) and check that the
chunked shape reaches the runtime IR.
"""

import os
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

# 268 chunks per row: the chunked tiles end in a partial tile.
W = 8576
RNG = np.random.default_rng(0)
X = RNG.standard_normal((2, W), dtype=np.float32) * 4
T = np.array([0.7, 1.3], np.float32)
DENSE = re.compile(rf"tensor<2x{W // 32}x32x")


def _put(value):
    return jax.device_put(np.asarray(value), jax.devices("tt")[0])


def _run(fn, args, trace, export_path, dense):
    options = {"optimization_level": "O1", "enable_trace": str(trace).lower()}
    if export_path is not None:
        options["export_path"] = str(export_path)
    previous = os.environ.get("LIBTT_DENSE_ROWS")
    os.environ["LIBTT_DENSE_ROWS"] = "1" if dense else "0"
    try:
        # A fresh function per compilation: the setting is read while compiling.
        out = jax.jit(lambda *a: fn(*a), compiler_options=options)(*[_put(a) for a in args])
        return jax.tree.map(np.asarray, out)
    finally:
        if previous is None:
            del os.environ["LIBTT_DENSE_ROWS"]
        else:
            os.environ["LIBTT_DENSE_ROWS"] = previous


def _runtime_ir(export_path):
    (ir,) = (export_path / "irs").glob("ttnn_runtime_*.mlir")
    return ir.read_text()


def _compare(fn, args, trace, tmp_path, exact=False, chunked=True):
    plain = _run(fn, args, trace, None, dense=False)
    dense = _run(fn, args, trace, tmp_path, dense=True)
    for p, d in zip(jax.tree.leaves(plain), jax.tree.leaves(dense)):
        assert p.shape == d.shape and p.dtype == d.dtype
        if exact or not np.issubdtype(p.dtype, np.floating):
            np.testing.assert_array_equal(d, p)
        else:
            np.testing.assert_allclose(d, p, rtol=2e-6, atol=1e-30)
    assert bool(DENSE.search(_runtime_ir(tmp_path))) == chunked
    return dense


def _softmax(x, t):
    z = x / t[:, None]
    e = jnp.exp(z - jnp.max(z, axis=-1, keepdims=True))
    return e / jnp.sum(e, axis=-1, keepdims=True)


@pytest.mark.parametrize("trace", [False, True])
def test_elementwise_chain(trace, tmp_path):
    def fn(x, t):
        z = x / t[:, None]
        return jnp.where(z > 1.0, jnp.exp(-jnp.abs(z)), z * z - 3.0)

    out = _compare(fn, (X, T), trace, tmp_path, exact=True)
    z = X / T[:, None]
    np.testing.assert_allclose(out, np.where(z > 1.0, np.exp(-np.abs(z)), z * z - 3.0), rtol=2e-2, atol=1e-3)


@pytest.mark.parametrize("trace", [False, True])
def test_softmax_and_log_probs(trace, tmp_path):
    def fn(x, t):
        p = _softmax(x, t)
        return p, jnp.log(p).clip(min=jnp.finfo(jnp.float32).min), jnp.sum(p, axis=-1)

    p, logp, total = _compare(fn, (X, T), trace, tmp_path)
    # Its max and sums run as masked reductions of the chunked rows.
    assert "ttnn.masked_reduce" in _runtime_ir(tmp_path)
    z = X / T[:, None]
    ref = np.exp(z - z.max(-1, keepdims=True))
    ref /= ref.sum(-1, keepdims=True)
    np.testing.assert_allclose(p, ref, rtol=5e-2, atol=1e-7)
    np.testing.assert_allclose(total, 1.0, rtol=2e-2)
    assert np.all(np.isfinite(logp))


@pytest.mark.parametrize("trace", [False, True])
def test_reductions(trace, tmp_path):
    def fn(x):
        return (
            jnp.max(x, axis=-1),
            jnp.min(x * 2.0, axis=-1, keepdims=True),
            jnp.mean(x - 1.0, axis=-1),
            jnp.sum((x > 0).astype(jnp.int32), axis=-1),
            x - jnp.max(x, axis=-1, keepdims=True),
            # Tile padding must not count: every value is below 0, or above.
            jnp.max(-jnp.abs(x) - 1.0, axis=-1),
            jnp.min(jnp.abs(x) + 1.0, axis=-1),
        )

    mx, mn, mean, count, centered, neg_max, pos_min = _compare(fn, (X,), trace, tmp_path)
    np.testing.assert_array_equal(neg_max, (-np.abs(X) - 1).max(-1))
    np.testing.assert_array_equal(pos_min, (np.abs(X) + 1).min(-1))
    np.testing.assert_array_equal(mx, X.max(-1))
    np.testing.assert_array_equal(mn, (X * 2).min(-1, keepdims=True))
    np.testing.assert_allclose(mean, (X - 1).mean(-1), rtol=1e-4)
    np.testing.assert_array_equal(count, (X > 0).sum(-1))
    np.testing.assert_array_equal(centered, X - X.max(-1, keepdims=True))


@pytest.mark.parametrize("trace", [False, True])
def test_row_and_column_operands(trace, tmp_path):
    def fn(x, t):
        column = lax.broadcasted_iota(jnp.int32, (1, W), 1).astype(jnp.float32)
        masked = jnp.where(x >= t[:, None], x, jnp.float32(-1e12))
        return masked + column * 1e-3

    out = _compare(fn, (X, T), trace, tmp_path, exact=True)
    ref = np.where(X >= T[:, None], X, np.float32(-1e12)) + np.arange(W, dtype=np.float32)[None] * np.float32(1e-3)
    np.testing.assert_allclose(out, ref, rtol=1e-6)


@pytest.mark.parametrize("trace", [False, True])
def test_cond_branches(trace, tmp_path):
    # Chunked before the cond for its own sake, then in both branches and
    # after: the chunked value flows through the case.
    def fn(x, t, b):
        z = jnp.exp(x / t[:, None]) * 2.0 - 1.0
        y = lax.cond(b, lambda y: y * 2.0 + 1.0, lambda y: y - jnp.max(y, axis=-1, keepdims=True), z)
        return jnp.abs(y) * 0.5

    for b in (True, False):
        out = _compare(fn, (X, T, np.bool_(b)), trace, tmp_path / str(b))
        z = np.exp(X / T[:, None]) * 2 - 1
        ref = np.abs(z * 2 + 1 if b else z - z.max(-1, keepdims=True)) * 0.5
        np.testing.assert_allclose(out, ref, rtol=3e-2, atol=1e-3)


def test_unlikely_branch_stays_wide(tmp_path):
    # A penalty-like branch alone is not worth reshaping its input and output
    # on the path around it.
    def fn(x, penalty, b):
        return lax.cond(b, lambda y: y - penalty * 2.0, lambda y: y, x)

    penalty = np.abs(X)
    _compare(fn, (X, penalty, np.bool_(False)), False, tmp_path, exact=True, chunked=False)


@pytest.mark.parametrize("trace", [False, True])
def test_loop_invariant_operand(trace, tmp_path):
    def fn(x, t):
        p = _softmax(x, t)

        def body(i, c):
            return c + jnp.sum(jnp.where(p * 2.0 >= c[:, None], p, 0.0), axis=-1)

        return lax.fori_loop(0, 5, body, jnp.zeros((2,), jnp.float32)), jnp.log(p)

    _compare(fn, (X, T), trace, tmp_path)


@pytest.mark.parametrize("trace", [False, True])
def test_gumbel_argmax(trace, tmp_path):
    bits = RNG.integers(0, 2**32, size=(2, W), dtype=np.uint32)

    def fn(x, t, bits):
        u = lax.bitcast_convert_type((bits >> 9) | np.uint32(0x3F800000), jnp.float32) - 1.0
        u = jnp.maximum(u, jnp.finfo(jnp.float32).tiny)
        return jnp.argmax(x / t[:, None] - jnp.log(-jnp.log(u)), axis=-1)

    out = _compare(fn, (X, T, bits), trace, tmp_path)
    assert "ttnn.argmax" not in _runtime_ir(tmp_path)
    u = ((bits >> 9) | np.uint32(0x3F800000)).view(np.float32) - 1.0
    u = np.maximum(u, np.finfo(np.float32).tiny)
    noisy = X / T[:, None] - np.log(-np.log(u))
    # Approximate logs on device: the winner's noisy value is within a hair of the max.
    assert np.all(noisy[np.arange(2), out] >= noisy.max(-1) - 1e-2)


def test_narrow_rows_stay(tmp_path):
    x = RNG.standard_normal((2, 1024), dtype=np.float32)
    plain = _run(lambda x: jnp.exp(x) * 2.0, (x,), False, None, dense=False)
    dense = _run(lambda x: jnp.exp(x) * 2.0, (x,), False, tmp_path, dense=True)
    np.testing.assert_array_equal(dense, plain)
    assert "x32x32x" not in _runtime_ir(tmp_path)


@pytest.mark.parametrize("trace", [False, True])
def test_argmax_ties_take_the_first(trace, tmp_path):
    x = np.zeros((2, W), np.float32)
    x[0, [17, 4000, 8000]] = 3.0  # three maxima: the first wins
    # row 1: all equal, index 0

    out = _compare(lambda x: jnp.argmax(x * 2.0, axis=-1), (x,), trace, tmp_path)
    np.testing.assert_array_equal(out, [17, 0])

