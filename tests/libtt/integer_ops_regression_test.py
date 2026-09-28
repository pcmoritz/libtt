"""Integer division, shifts, unsigned min/max, narrow indices and Float16 casts must be exact."""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax


def _run(fn, *args):
    device = jax.devices("tt")[0]
    return np.asarray(jax.jit(fn)(*(jax.device_put(a, device) for a in args)))


def _cpu(fn, *args):
    return np.asarray(jax.jit(fn, device=jax.devices("cpu")[0])(*args))


@pytest.mark.parametrize("dtype", [np.int32, np.int16, np.uint16, np.uint8])
def test_integer_division_is_exact(dtype):
    info = np.iinfo(dtype)
    rng = np.random.default_rng(0)
    x = rng.integers(max(info.min, -(2**30)), min(info.max, 2**30), (4, 64)).astype(dtype)
    y = rng.integers(1, min(info.max, 1000), (4, 64)).astype(dtype)
    if info.min < 0:
        y = np.where(rng.integers(0, 2, y.shape) == 1, -y, y).astype(dtype)
    # Boundaries: extreme dividends, divisors of +-1 (except INT_MIN / -1, which overflows), and for
    # Int32 quotients far above 2^24, where a float32 division would round.
    dividends = [info.max, info.min + 1, 0, 1] + ([2**30 + 7, -(2**31) + 5] if dtype == np.int32 else [])
    divisors = [1, 3, info.max] + ([-1, -3, info.min] if info.min < 0 else [])
    pairs = np.array(list(itertools.product(dividends, divisors)), dtype=np.int64)
    x = np.concatenate([x.ravel(), pairs[:, 0].astype(dtype)])
    y = np.concatenate([y.ravel(), pairs[:, 1].astype(dtype)])
    np.testing.assert_array_equal(_run(lax.div, x, y), _cpu(lax.div, x, y))
    np.testing.assert_array_equal(_run(jnp.floor_divide, x, y), np.floor_divide(x, y))


_UINT32_VALUES = np.array(
    [
        [[0, 0xFFFFFFFF, 3, 0x80000000], [0x7FFFFFFF, 4_000_000_000, 0xFFFF0000, 0x0000FFFF]],
        [[5, 0x80000001, 3_000_000_000, 0xFFFF0001], [0xFFFFFFFE, 2, 0x00010000, 0xFFFF]],
    ],
    dtype=np.uint32,
)


@pytest.mark.parametrize("op", ["min", "max"])
@pytest.mark.parametrize("axis", [0, 2, -1, (0, 2), (1, 2), None])
@pytest.mark.parametrize("keepdims", [False, True])
def test_uint32_min_max(op, axis, keepdims):
    expected = getattr(np, op)(_UINT32_VALUES, axis=axis, keepdims=keepdims)
    actual = _run(lambda x: getattr(jnp, op)(x, axis=axis, keepdims=keepdims), _UINT32_VALUES)
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("dtype", [np.int8, np.uint8, np.int16, np.uint16])
@pytest.mark.parametrize("op", ["shift_left", "shift_right_logical", "shift_right_arithmetic"])
def test_narrow_integer_shifts(dtype, op):
    info = np.iinfo(dtype)
    top_bit = np.array(1 << (info.bits - 1), np.uint64).astype(dtype)
    values = [info.min, info.max, top_bit, top_bit | 1, 0, 1, 0x55, -1 if info.min < 0 else info.max - 1]
    # Amounts 0, width - 1, width and width + 1, plus the largest unsigned amount (-1 when signed).
    amounts = [0, 1, info.bits - 1, info.bits, info.bits + 1, -1 if info.min < 0 else info.max]
    pairs = list(itertools.product(values, amounts))
    x = np.array([v for v, _ in pairs], np.int64).astype(dtype)
    y = np.array([a for _, a in pairs], np.int64).astype(dtype)
    fn = getattr(lax, op)
    np.testing.assert_array_equal(_run(fn, x, y), _cpu(fn, x, y))


def test_uint8_dynamic_slice_index():
    values = np.arange(200, dtype=np.int32)
    actual = _run(lambda x, i: lax.dynamic_slice(x, (i,), (1,)), values, np.uint8(128))
    np.testing.assert_array_equal(actual, [128])


def test_uint8_arange_widened():
    # A UInt8 arange cast to a wider integer is created directly in the wider type.
    offset = np.array([1000], np.int32)
    actual = _run(lambda o: jnp.arange(256, dtype=jnp.uint8).astype(jnp.int32) + o, offset)
    np.testing.assert_array_equal(actual, np.arange(256, dtype=np.int32) + 1000)


def test_int8_arange_to_unsigned():
    # Negative values must still wrap when a signed arange is cast to an unsigned type.
    offset = np.array([1], np.uint32)
    actual = _run(lambda o: jnp.arange(-4, 4, dtype=jnp.int8).astype(jnp.uint32) + o, offset)
    np.testing.assert_array_equal(actual, np.arange(-4, 4, dtype=np.int8).astype(np.uint32) + 1)


def test_float16_where_with_scalar_predicate():
    values = np.array([1.5, -2.0, 3.0, 0.25], np.float16)
    actual = _run(lambda x, d: jnp.where(d > 0, x, jnp.nan), values, np.float32(3.0))
    np.testing.assert_array_equal(actual, values)
    np.testing.assert_allclose(
        _run(lambda x: jnp.var(x, axis=0), values.reshape(2, 2)),
        np.var(values.reshape(2, 2).astype(np.float32), axis=0).astype(np.float16),
        rtol=1e-3,
    )


def _float_boundaries(dtype):
    info = jnp.finfo(dtype)
    values = [3.0, -1.5, 0.0, -0.0, 0.5, 1 + float(info.eps), np.inf, -np.inf, np.nan, -np.nan]
    values += [float(info.max), -float(info.max), float(info.smallest_subnormal), float(info.tiny)]
    return np.array(values, np.float32).astype(dtype)


@pytest.mark.parametrize("src, dst", [(jnp.bfloat16, np.float16), (np.float16, jnp.bfloat16)])
def test_float16_bfloat16_typecast(src, dst):
    values = _float_boundaries(src)
    if src == np.float16:
        # A NaN with a small payload must stay NaN rather than round to infinity.
        values = np.concatenate([values, np.array([0x7C01, 0xFC01], np.uint16).view(np.float16)])
    # Pass the raw bits and bitcast on device: Float16 program inputs flush -0 and subnormals.
    actual = _run(lambda bits: lax.bitcast_convert_type(bits, src).astype(dst), values.view(np.uint16))
    expected = values.astype(np.float32).astype(dst)
    # Compare as float32 so NaNs in the same positions count as equal for bfloat16 too.
    np.testing.assert_array_equal(actual.astype(np.float32), expected.astype(np.float32))
    np.testing.assert_array_equal(np.signbit(actual[~np.isnan(expected)]), np.signbit(expected[~np.isnan(expected)]))
