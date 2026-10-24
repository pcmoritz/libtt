"""A sum of a compare against a per-row threshold, or of a select on one,
runs as one fused masked reduction.

The binary searches behind SGLang-JAX's top-k and top-p masks reduce such a
compare over the vocabulary 31 times per step. As three ops (compare, select,
sum) each step passes over the [batch, vocab] input and two intermediates of
its size, every pass latency-bound on its tiles. ttnn.masked_reduce reads the
input once and accumulates the mask, or the masked values, on the compute
engine; the compiler forms it from the sum (tt_mlir_masked_reduce.patch).
"""

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

SOURCE = np.random.default_rng(0).standard_normal((2, 2048), dtype=np.float32) * 4


def _jit(fn, trace=False, export_path=None):
    options = {"optimization_level": "O1", "enable_trace": str(trace).lower()}
    if export_path is not None:
        options["export_path"] = str(export_path)
    return jax.jit(fn, compiler_options=options)


def _put(value):
    return jax.device_put(np.asarray(value), jax.devices("tt")[0])


def _runtime_ir(export_path):
    (ir,) = (export_path / "irs").glob("ttnn_runtime_*.mlir")
    return ir.read_text()


def _count_threshold(x, k):
    """SGLang-JAX's top-k search: the largest threshold with at least k values
    above it in each row (positive here, so float bits order like integers)."""
    xt = jnp.swapaxes(x, 0, 1)

    def body(i, bits):
        bit = jnp.int32(1 << (30 - i))
        threshold = lax.bitcast_convert_type(bits | bit, jnp.float32)
        count = jnp.sum(xt > threshold[None, :], axis=0)
        return bits | jnp.where(count >= k, bit, jnp.int32(0))

    bits = lax.fori_loop(0, 31, body, jnp.zeros((x.shape[0],), jnp.int32))
    return lax.bitcast_convert_type(bits, jnp.float32)


def _mass_threshold(probs, p):
    """SGLang-JAX's top-p search: the largest threshold keeping mass >= p."""
    pt = jnp.swapaxes(probs, 0, 1)

    def body(i, bits):
        bit = jnp.int32(1 << (30 - i))
        threshold = lax.bitcast_convert_type(bits | bit, jnp.float32)
        mass = jnp.sum(jnp.where(pt >= threshold[None, :], pt, 0.0), axis=0)
        return bits | jnp.where(mass >= p, bit, jnp.int32(0))

    bits = lax.fori_loop(0, 31, body, jnp.zeros((probs.shape[0],), jnp.int32))
    return lax.bitcast_convert_type(bits, jnp.float32)


@pytest.mark.parametrize("trace", [False, True])
def test_count_search_runs_fused(trace, tmp_path):
    k = 20
    run = _jit(lambda x: _count_threshold(x, k), trace, tmp_path)
    threshold = np.asarray(run(_put(SOURCE)))
    assert np.all((SOURCE > threshold[:, None]).sum(axis=1) >= k)
    assert np.all((SOURCE > np.nextafter(threshold, np.inf)[:, None]).sum(axis=1) < k)
    ir = _runtime_ir(tmp_path)
    # The step's [2] thresholds go in, as the candidates' int32 bits (no
    # bitcast), and its [2] int32 counts come out as they are: no reshape to
    # and from [2, 1], no typecast of the counts.
    assert re.search(r'"ttnn\.masked_reduce".*tensor<2xsi32.*-> tensor<2xsi32', ir)
    assert re.search(r'"ttnn\.reshape".*-> tensor<2x1x', ir) is None
    # No compare or select over the vocabulary survives.
    assert re.search(r'"ttnn\.(gt|ge|lt|le|where)".*tensor<2x2048x', ir) is None


@pytest.mark.parametrize("trace", [False, True])
def test_mass_search_runs_fused(trace, tmp_path):
    p = 0.9
    probs = np.exp(SOURCE) / np.exp(SOURCE).sum(axis=1, keepdims=True)
    run = _jit(lambda v: _mass_threshold(v, p), trace, tmp_path)
    threshold = np.asarray(run(_put(probs)))
    kept = np.where(probs >= threshold[:, None], probs, 0.0).sum(axis=1)
    assert np.all(kept >= p - 1e-3)
    dropped = np.where(probs >= np.nextafter(threshold, np.inf)[:, None], probs, 0.0).sum(axis=1)
    assert np.all(dropped < p + 1e-3)
    ir = _runtime_ir(tmp_path)
    assert re.search(r'"ttnn\.masked_reduce".*tensor<2xsi32.*-> tensor<2xf32', ir)
    assert re.search(r'"ttnn\.reshape".*-> tensor<2x1x', ir) is None


@pytest.mark.parametrize(
    "dtype,shape",
    [(np.float32, (40, 1000)), (jnp.bfloat16, (4, 4096)), (np.float32, (3, 2, 1000))],
)
def test_counts_and_sums_match(dtype, shape, tmp_path):
    # Thresholds taken from the rows, so ties tell > from >=; a width that is
    # not whole tiles; rows across two tile rows; a leading batch dimension.
    rng = np.random.default_rng(1)
    x = (rng.standard_normal(shape, dtype=np.float32) * 4).astype(dtype)
    t = np.take_along_axis(x, rng.integers(0, shape[-1], size=shape[:-1] + (1,)), axis=-1)
    xf, tf = x.astype(np.float32), t.astype(np.float32)

    def reduce_all(x, t):
        return (
            jnp.sum(x > t, axis=-1),
            jnp.sum(x >= t, axis=-1, keepdims=True),
            jnp.sum(t < x, axis=-1),
            jnp.sum(jnp.where(x <= t, x, 0), axis=-1),
            jnp.sum(jnp.where(x >= t, x, 0), axis=-1, keepdims=True),
        )

    got = [np.asarray(v) for v in _jit(reduce_all, export_path=tmp_path)(_put(x), _put(t))]
    # Four fused reductions: x > t and t < x are one computation.
    assert _runtime_ir(tmp_path).count("ttnn.masked_reduce") == 4
    np.testing.assert_array_equal(got[0], (xf > tf).sum(axis=-1))
    np.testing.assert_array_equal(got[1], (xf >= tf).sum(axis=-1, keepdims=True))
    np.testing.assert_array_equal(got[2], (tf < xf).sum(axis=-1))
    tolerance = dict(rtol=1e-5, atol=1e-2) if dtype == np.float32 else dict(rtol=2e-2, atol=1e-1)
    np.testing.assert_allclose(got[3], np.where(xf <= tf, xf, 0).sum(axis=-1), **tolerance)
    np.testing.assert_allclose(got[4], np.where(xf >= tf, xf, 0).sum(axis=-1, keepdims=True), **tolerance)
