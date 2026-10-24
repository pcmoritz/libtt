"""Reductions over the rows of a tall, thin tensor run as wide-row reductions.

The binary searches behind SGLang-JAX's top-k and top-p masks swap the vocab
axis to the front and reduce over it 32 times per mask. On TT a [vocab, batch]
tensor has one tile column, so that reduction, and even the elementwise ops
feeding it, run on one core: 2.9 ms per search step against 0.5 ms on
[batch, vocab]. The compiler rebuilds such a reduction chain on the transposed
shape (tt_mlir_wide_reductions.patch), consuming the transpose that produced
the operand instead of adding one.

Wide softmax rows are the other sampler cost: TTNN's softmax parallelizes over
rows, so two rows of a vocabulary take 12 ms where the decomposition takes 1 ms.
The softmax fusion skips that shape.
"""

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

VOCAB = 2048
SOURCE = np.random.default_rng(0).standard_normal((2, VOCAB), dtype=np.float32) * 4


def _jit(fn, trace, export_path=None):
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
    """The largest threshold with at least k values above it in each row, found
    the way SGLang-JAX's masks search: 31 reductions over the vocab axis, which
    the search puts first. Thresholds stay positive here (the k-th largest of
    these rows is), so float bit patterns order like integers."""
    xt = jnp.swapaxes(x, 0, 1)

    def body(i, bits):
        bit = jnp.int32(1 << (30 - i))
        candidate = bits | bit
        threshold = lax.bitcast_convert_type(candidate, jnp.float32)
        count = jnp.sum(xt > threshold[None, :], axis=0)
        return bits | jnp.where(count >= k, bit, jnp.int32(0))

    bits = lax.fori_loop(0, 31, body, jnp.zeros((x.shape[0],), jnp.int32))
    return lax.bitcast_convert_type(bits, jnp.float32)


@pytest.mark.parametrize("trace", [False, True])
def test_search_reduces_over_wide_rows(trace, tmp_path):
    k = 20
    run = _jit(lambda x: _count_threshold(x, k), trace, tmp_path)
    threshold = np.asarray(run(_put(SOURCE)))
    # At least k values lie above the threshold, and fewer above the next
    # larger float.
    assert np.all((SOURCE > threshold[:, None]).sum(axis=1) >= k)
    assert np.all((SOURCE > np.nextafter(threshold, np.inf)[:, None]).sum(axis=1) < k)
    ir = _runtime_ir(tmp_path)
    # The vocab axis stays last: no transpose survives, and every reduction
    # over it is a last-dim reduction.
    assert "ttnn.permute" not in ir and "ttnn.transpose" not in ir
    assert re.search(r'"ttnn\.sum".*dim_arg = \[0 : i32\]', ir) is None


@pytest.mark.parametrize("trace", [False, True])
def test_mass_search_reduces_over_wide_rows(trace):
    # The top-p variant: a masked sum of probabilities over the vocab axis.
    # The device's softmax is the reference, since the search ran on it.
    p = 0.9

    def mass_threshold(x):
        probs = jax.nn.softmax(x, axis=-1)
        pt = jnp.swapaxes(probs, 0, 1)

        def body(i, bits):
            bit = jnp.int32(1 << (30 - i))
            candidate = bits | bit
            threshold = lax.bitcast_convert_type(candidate, jnp.float32)
            mass = jnp.sum(jnp.where(pt >= threshold[None, :], pt, 0.0), axis=0)
            return bits | jnp.where(mass >= p, bit, jnp.int32(0))

        bits = lax.fori_loop(0, 31, body, jnp.zeros((x.shape[0],), jnp.int32))
        return lax.bitcast_convert_type(bits, jnp.float32), probs

    run = _jit(mass_threshold, trace)
    threshold, probs = (np.asarray(v) for v in run(_put(SOURCE)))
    kept = np.where(probs >= threshold[:, None], probs, 0.0).sum(axis=1)
    assert np.all(kept >= p - 1e-3)
    dropped = np.where(probs >= np.nextafter(threshold, np.inf)[:, None], probs, 0.0).sum(axis=1)
    assert np.all(dropped < p + 1e-3)


@pytest.mark.parametrize("trace", [False, True])
def test_wide_softmax_stays_decomposed(trace, tmp_path):
    x = np.random.default_rng(1).standard_normal((2, 16384), dtype=np.float32) * 3
    run = _jit(lambda v: jax.nn.softmax(v, axis=-1), trace, tmp_path)
    result = np.asarray(run(_put(x)))
    expected = np.asarray(jax.nn.softmax(jnp.asarray(x), axis=-1))
    np.testing.assert_allclose(result, expected, rtol=2e-3, atol=1e-6)
    assert "ttnn.softmax" not in _runtime_ir(tmp_path)
