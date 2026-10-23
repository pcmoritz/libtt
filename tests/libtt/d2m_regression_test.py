"""Exact small-tensor arithmetic and search loops through D2M fusion."""

import operator

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax


def _run(fn, args, path, d2m=True):
    options = {
        "optimization_level": "O1",
        "enable_trace": "true",
        "enable_create_d2m_subgraphs": str(d2m).lower(),
        "export_path": str(path),
    }
    device = jax.devices("tt")[0]
    result = jax.jit(fn, compiler_options=options)(
        *(jax.device_put(x, device) for x in args)
    )
    result = jax.tree.map(np.asarray, result)
    (ir,) = (path / "irs").glob("ttnn_runtime_*.mlir")
    return result, ir.read_text()


@pytest.mark.parametrize(
    "compare",
    [operator.eq, operator.ne, operator.lt, operator.le, operator.gt, operator.ge],
)
@pytest.mark.parametrize("dtype", [np.int32, np.float32])
def test_comparisons_preserve_operand_precision(compare, dtype, tmp_path):
    if dtype == np.int32:
        # These include float NaN/-0 bit patterns and integers that round to
        # the same bf16 value. Comparisons must use signed integer ordering.
        x = np.array(
            [-2147483648, -2147483647, -1, 0, 1, 16777217, 2139095041, 2147483647],
            dtype=dtype,
        )[None, :]
        y = np.array([0, -2147483648, 0, 0, 1, 16777216, 2139095040, -1], dtype=dtype)[
            None, :
        ]
    else:
        x = np.array(
            [0.94996, 0.95091, -0.94996, -0.95091, 1.00001, -1.00001, 0, -0.0],
            dtype=dtype,
        )[None, :]
        y = np.array([0.95, 0.95, -0.95, -0.95, 1, -1, 0, 0], dtype=dtype)[None, :]

    def select(a, b):
        a = a ^ jnp.int32(1) if dtype == np.int32 else a + a
        return jnp.where(compare(a, b), jnp.ones_like(a), -jnp.ones_like(a))

    encoded = x ^ np.int32(1) if dtype == np.int32 else x / 2
    got, ir = _run(select, (encoded, y), tmp_path)
    np.testing.assert_array_equal(got, np.where(compare(x, y), 1, -1))
    assert '"ttnn.generic"' in ir


def test_integer_broadcast_arithmetic_is_exact(tmp_path):
    rng = np.random.default_rng(3)
    args = tuple(
        rng.integers(0, 1024, shape, dtype=np.int32)
        for shape in [(2, 32), (1, 32), (2, 1), (1, 1)]
    )

    def calculate(a, b, c, d):
        return (a + b) * c - d

    got, ir = _run(calculate, args, tmp_path)
    np.testing.assert_array_equal(got, calculate(*args))
    assert '"ttnn.generic"' in ir


def _float_search(shape, predicate):
    """Search the signed monotonic encoding of all finite float32 values."""

    def float_bits(bits):
        return bits ^ jnp.where(bits < 0, jnp.int32(0x7FFFFFFF), jnp.int32(0))

    def test(bits):
        bits = float_bits(bits)
        finite = (bits & jnp.int32(0x7F800000)) != jnp.int32(0x7F800000)
        return jnp.where(
            finite, predicate(lax.bitcast_convert_type(bits, jnp.float32)), bits >= 0
        )

    zero = jnp.zeros(shape, jnp.int32)
    initial = jnp.where(test(zero), jnp.int32(-2147483648), zero)

    def step(i, bits):
        bit = jnp.int32(1) << (30 - i)
        return bits | jnp.where(test(bits | bit), jnp.int32(0), bit)

    bits = lax.fori_loop(0, 31, step, initial)
    return lax.bitcast_convert_type(float_bits(bits), jnp.float32)


@pytest.mark.parametrize("mode", ["topp", "topk", "topk_negative"])
def test_search_matches_unfused_and_numpy(mode, tmp_path):
    x = np.random.default_rng(0).standard_normal((2, 2048), dtype=np.float32) * 4
    if mode == "topp":
        x = np.exp(x - x.max(-1, keepdims=True))
        x /= x.sum(-1, keepdims=True)
        limit = np.array([0.95, 0.9], np.float32)

        def search(x, p):
            xt = jnp.swapaxes(x, -1, -2)
            return _float_search(
                x.shape[:-1],
                lambda t: jnp.sum(jnp.where(xt >= t[None, :], xt, 0), axis=0) < p,
            )

    else:
        if mode == "topk_negative":
            x -= 20
        limit = np.array([20, 50], np.int32)

        def search(x, k):
            xt = jnp.swapaxes(x, -1, -2)
            return -_float_search(
                x.shape[:-1], lambda t: jnp.sum(xt > (-t)[None, :], axis=0) >= k
            )

    reference, _ = _run(search, (x, limit), tmp_path / "plain", d2m=False)
    got, ir = _run(search, (x, limit), tmp_path / "d2m")
    np.testing.assert_array_equal(got.view(np.uint32), reference.view(np.uint32))
    if mode == "topp":
        mass = np.where(x >= got[:, None], x, 0).sum(-1)
        next_mass = np.where(x >= np.nextafter(got, np.inf)[:, None], x, 0).sum(-1)
        assert np.all(mass >= limit)
        assert np.all(next_mass < limit)
    else:
        np.testing.assert_array_equal((x >= got[:, None]).sum(-1), limit)
    assert '"ttnn.generic"' in ir
    assert '"ttnn.masked_reduce"' in ir
    assert '"ttnn.execute_trace"' in ir
