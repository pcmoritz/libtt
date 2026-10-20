"""float8 tensors: storage, conversions and native matmuls.

float8 tensors are stored as their bits. Conversions to and from float8 are
exact integer arithmetic on the device, so they match ml_dtypes (and
PyTorch): rounding to nearest even, subnormals, NaN on e4m3fn overflow and
infinity on e5m2 overflow. A matmul of two float8 matrices reads them
natively on the matrix unit, which flushes subnormal operands to zero.
"""

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest

FORMATS = [jnp.float8_e4m3fn, jnp.float8_e5m2]


def all_codes(dtype):
    """Every bit pattern of the format, as a 32 x 32 array (each one 4 times)."""
    return np.tile(np.arange(256, dtype=np.uint8), 4).reshape(32, 32).view(dtype)


def assert_same_bits(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.dtype == expected.dtype
    width = {1: np.uint8, 2: np.uint16, 4: np.uint32}[actual.dtype.itemsize]
    actual_bits, expected_bits = actual.view(width), expected.view(width)
    nan = np.isnan(expected.astype(np.float32))
    np.testing.assert_array_equal(np.isnan(actual.astype(np.float32)), nan)
    np.testing.assert_array_equal(actual_bits[~nan], expected_bits[~nan])


@pytest.mark.parametrize("dtype", FORMATS)
def test_float8_buffers_round_trip(dtype):
    codes = all_codes(dtype)
    assert_same_bits(jax.device_put(codes), codes)
    moved = jax.jit(lambda x: jnp.concatenate([x.T, x[::-1]], axis=0))(codes)
    assert_same_bits(moved, np.concatenate([codes.T, codes[::-1]], axis=0))


@pytest.mark.parametrize("dtype", FORMATS)
def test_decode_every_code(dtype):
    codes = all_codes(dtype)
    decoded = jax.jit(lambda x: x.astype(jnp.float32))(codes)
    assert_same_bits(decoded, codes.astype(np.float32))
    to_bf16 = jax.jit(lambda x: x.astype(jnp.bfloat16))(codes)
    assert_same_bits(to_bf16, codes.astype(jnp.bfloat16))


def encode_inputs():
    """Every bfloat16, random float32 values over the formats' ranges, and edges."""
    bf16 = (np.arange(1 << 16, dtype=np.uint32) << 16).view(np.float32)
    rng = np.random.default_rng(0)
    scales = np.ldexp(1.0, rng.integers(-20, 18, size=1 << 15)).astype(np.float32)
    random = (rng.uniform(1.0, 2.0, size=1 << 15).astype(np.float32) * scales) * rng.choice(
        [-1, 1], size=1 << 15
    ).astype(np.float32)
    edges = np.array(
        [440, 447.9, 448, 455.9, 456, 456.1, 463.9, 464, 464.1, 479.9, 480, 1e6,
         57344, 59391, 59392, 61439, 61440, 65535, 65536, np.inf, -np.inf, np.nan,
         2.0**-10, 1.5 * 2.0**-10, 2.0**-9, 2.5 * 2.0**-9, 2.0**-17, 3 * 2.0**-17, 0.0, -0.0],
        np.float32,
    )
    values = np.concatenate([bf16, random, edges, -edges])
    return np.pad(values, (0, -len(values) % 1024)).reshape(-1, 32)


@pytest.mark.parametrize("dtype", FORMATS)
def test_encode_rounds_like_ml_dtypes(dtype):
    values = encode_inputs()
    encoded = jax.jit(lambda x: x.astype(dtype))(values)
    assert_same_bits(encoded, values.astype(dtype))


@pytest.mark.parametrize("dtype", FORMATS)
def test_arithmetic_on_float8(dtype):
    rng = np.random.default_rng(1)
    x = rng.normal(size=(64, 32)).astype(np.float32).astype(dtype)
    y = rng.normal(size=(64, 32)).astype(np.float32).astype(dtype)
    total, larger = jax.jit(lambda x, y: (x + y, x > y))(x, y)
    assert_same_bits(total, (x.astype(np.float32) + y.astype(np.float32)).astype(dtype))
    np.testing.assert_array_equal(larger, x.astype(np.float32) > y.astype(np.float32))


def random_float8(shape, dtype, rng):
    """Random finite float8 values, subnormals included."""
    codes = rng.integers(0, 256, size=shape).astype(np.uint8)
    values = codes.view(dtype)
    finite = np.isfinite(values.astype(np.float32))
    values[~finite] = np.array(1.0, dtype)
    return values


def flush_subnormals(x):
    values = x.astype(np.float32)
    smallest_normal = float(ml_dtypes.finfo(x.dtype).smallest_normal)
    return np.where(np.abs(values) < smallest_normal, 0.0, values).astype(np.float32)


@pytest.mark.parametrize(
    "lhs_dtype,rhs_dtype",
    [
        (jnp.float8_e4m3fn, jnp.float8_e4m3fn),
        (jnp.float8_e5m2, jnp.float8_e5m2),
        (jnp.float8_e4m3fn, jnp.float8_e5m2),
    ],
)
@pytest.mark.parametrize("m,k,n", [(32, 32, 32), (64, 256, 96), (256, 1024, 128)])
def test_native_float8_matmul(m, k, n, lhs_dtype, rhs_dtype):
    rng = np.random.default_rng(m + k + n)
    a, b = random_float8((m, k), lhs_dtype, rng), random_float8((k, n), rhs_dtype, rng)
    product = jax.jit(
        lambda a, b: jnp.dot(a, b, preferred_element_type=jnp.float32)
    )(a, b)
    a32, b32 = flush_subnormals(a).astype(np.float64), flush_subnormals(b).astype(np.float64)
    expected = a32 @ b32
    # The matrix unit's accumulation, about as for bfloat16 operands.
    scale = np.abs(a32) @ np.abs(b32)
    assert np.max(np.abs(product - expected) / np.maximum(scale, 1e-30)) < 2.0**-9


@pytest.mark.parametrize(
    "dtype,subnormal,one", [(jnp.float8_e4m3fn, 0x07, 0x38), (jnp.float8_e5m2, 0x03, 0x3C)]
)
def test_native_float8_matmul_flushes_subnormal_operands(dtype, subnormal, one):
    """The matrix unit reads the operands natively: subnormals count as zero."""
    a = np.full((32, 32), subnormal, np.uint8).view(dtype)
    b = np.full((32, 32), one, np.uint8).view(dtype)
    product = jax.jit(lambda a, b: jnp.dot(a, b, preferred_element_type=jnp.float32))(a, b)
    np.testing.assert_array_equal(product, np.zeros((32, 32), np.float32))


@pytest.mark.parametrize(
    "lhs_dtype,rhs_dtype", [(jnp.float8_e4m3fn, jnp.bfloat16), (jnp.bfloat16, jnp.float8_e5m2)]
)
def test_float8_matmul_on_decoded_values(lhs_dtype, rhs_dtype):
    """A float8 matrix times a bfloat16 one computes on the exactly decoded values."""
    rng = np.random.default_rng(3)
    a = rng.normal(size=(64, 128)).astype(np.float32).astype(lhs_dtype)
    b = rng.normal(size=(128, 32)).astype(np.float32).astype(rhs_dtype)
    product = jax.jit(lambda a, b: jnp.dot(a, b, preferred_element_type=jnp.float32))(a, b)
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    scale = np.abs(a64) @ np.abs(b64)
    assert np.max(np.abs(product - a64 @ b64) / scale) < 2.0**-8
