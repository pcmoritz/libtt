"""Integer division, shifts, unsigned min/max and narrow indices must be exact."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax


def _run(fn, *args):
    device = jax.devices("tt")[0]
    return np.asarray(jax.jit(fn)(*(jax.device_put(a, device) for a in args)))


@pytest.mark.parametrize("dtype", [np.int32, np.int16, np.uint16, np.uint8])
def test_integer_division_is_exact(dtype):
    info = np.iinfo(dtype)
    rng = np.random.default_rng(0)
    x = rng.integers(max(info.min, -(2**30)), min(info.max, 2**30), (4, 64)).astype(dtype)
    y = rng.integers(1, min(info.max, 1000), (4, 64)).astype(dtype)
    if info.min < 0:
        y = np.where(rng.integers(0, 2, y.shape) == 1, -y, y).astype(dtype)
    np.testing.assert_array_equal(_run(lax.div, x, y), np.trunc(x / y.astype(np.float64)).astype(dtype))
    np.testing.assert_array_equal(_run(jnp.floor_divide, x, y), np.floor_divide(x, y))


@pytest.mark.parametrize("op", ["min", "max"])
@pytest.mark.parametrize("axis", [0, 1, None])
def test_uint32_min_max(op, axis):
    values = np.array(
        [[0, 0xFFFFFFFF, 3], [0x80000000, 0x7FFFFFFF, 4_000_000_000], [5, 2, 3_000_000_000]],
        dtype=np.uint32,
    )
    expected = getattr(np, op)(values, axis=axis)
    np.testing.assert_array_equal(_run(lambda x: getattr(jnp, op)(x, axis=axis), values), expected)


@pytest.mark.parametrize("dtype", [np.int8, np.uint8, np.int16, np.uint16])
@pytest.mark.parametrize("op", ["shift_left", "shift_right_logical", "shift_right_arithmetic"])
def test_narrow_integer_shifts(dtype, op):
    info = np.iinfo(dtype)
    rng = np.random.default_rng(1)
    x = rng.integers(info.min, info.max, (4, 32), endpoint=True).astype(dtype)
    y = rng.integers(0, info.bits + 3, (4, 32)).astype(dtype)
    expected = np.asarray(jax.jit(getattr(lax, op), device=jax.devices("cpu")[0])(x, y))
    np.testing.assert_array_equal(_run(getattr(lax, op), x, y), expected)


def test_uint8_dynamic_slice_index():
    values = np.arange(200, dtype=np.int32)
    actual = _run(lambda x, i: lax.dynamic_slice(x, (i,), (1,)), values, np.uint8(128))
    np.testing.assert_array_equal(actual, [128])


def test_float16_where_with_scalar_predicate():
    values = np.array([1.5, -2.0, 3.0, 0.25], np.float16)
    actual = _run(lambda x, d: jnp.where(d > 0, x, jnp.nan), values, np.float32(3.0))
    np.testing.assert_array_equal(actual, values)
    np.testing.assert_allclose(
        _run(lambda x: jnp.var(x, axis=0), values.reshape(2, 2)),
        np.var(values.reshape(2, 2).astype(np.float32), axis=0).astype(np.float16),
        rtol=1e-3,
    )
