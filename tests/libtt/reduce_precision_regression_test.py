"""lax.reduce_precision runs as ttnn.reduce_precision, one kernel with XLA's
bit manipulation on the input's own bits: round to nearest even, flush below
the requested format's smallest normal, saturate to infinity, keep NaN. It used
to fail to legalize for formats other than float16, bfloat16 and float32.
Results must match XLA on the CPU bit for bit, including for subnormals of the
input format.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

SPECIAL = np.array(
    [0.0, -0.0, np.inf, -np.inf, np.nan, 1.0, 1.0625, 1.125, 1.1875, 3e38, -3e38, 1e-30, 448.0, 464.0, 57344.0, 0.015625],
    np.float32,
)

FORMATS = [(4, 3), (5, 2), (3, 4), (6, 8), (8, 0), (5, 10), (8, 7), (1, 2)]
FORMAT_IDS = ["e4m3", "e5m2", "e3m4", "e6m8", "e8m0", "e5m10", "e8m7", "e1m2"]


def values(dtype):
    x = np.random.default_rng(0).normal(scale=4.0, size=(8, 16)).astype(np.float32)
    x.reshape(-1)[: SPECIAL.size] = SPECIAL
    return x.astype(dtype)


def run(device, x, exponent_bits, mantissa_bits):
    f = jax.jit(lambda v: lax.reduce_precision(v, exponent_bits, mantissa_bits))
    return np.asarray(f(jax.device_put(x, jax.devices(device)[0])))


def assert_matches_cpu(x, bits):
    got, expected = run("tt", x, *bits), run("cpu", x, *bits)
    nan = np.isnan(expected.astype(np.float32))
    np.testing.assert_array_equal(np.isnan(got.astype(np.float32)), nan)
    uint = {2: np.uint16, 4: np.uint32}[x.dtype.itemsize]
    np.testing.assert_array_equal(got[~nan].view(uint), expected[~nan].view(uint))


@pytest.mark.parametrize("dtype", [np.float32, jnp.bfloat16, np.float16], ids=["f32", "bf16", "f16"])
@pytest.mark.parametrize("bits", FORMATS, ids=FORMAT_IDS)
def test_reduce_precision_matches_xla(dtype, bits):
    assert_matches_cpu(values(dtype), bits)


# Every 16-bit pattern, so subnormals of the input format are covered.
@pytest.mark.parametrize("dtype", [jnp.bfloat16, np.float16], ids=["bf16", "f16"])
@pytest.mark.parametrize("bits", FORMATS, ids=FORMAT_IDS)
def test_reduce_precision_all_16_bit_patterns(dtype, bits):
    assert_matches_cpu(np.arange(1 << 16, dtype=np.uint32).astype(np.uint16).view(dtype).reshape(256, 256), bits)


# Subnormal results depend on the input format: 2^-15 is normal in float32
# but subnormal in float16.
@pytest.mark.parametrize(
    "dtype, value, bits",
    [(np.float32, 2.0**-15, (5, 10)), (np.float16, 2.0**-15, (5, 2)), (np.float16, 2.0**-24, (6, 8))],
    ids=["f32-e5m10", "f16-e5m2", "f16-e6m8"],
)
def test_reduce_precision_subnormals(dtype, value, bits):
    assert_matches_cpu(np.full((4, 4), value, dtype), bits)


def float32_boundaries(exponent_bits, mantissa_bits):
    """Float32 bit patterns around the format's rounding, underflow and overflow
    boundaries, plus subnormals, infinities and NaN, with both signs."""
    dropped = max(23 - mantissa_bits, 0)
    half = (1 << dropped) >> 1
    last_kept = 1 << dropped if dropped < 23 else 0
    mantissas = {0, 1, 0x7FFFFF, 0x400000, half, half - 1, half + 1, last_kept | half, last_kept | (half - 1), last_kept | (half + 1)}
    reduced_bias = (1 << (exponent_bits - 1)) - 1
    min_exponent, max_exponent = 127 - reduced_bias, 127 + reduced_bias
    exponents = {0, 1, 127, 254, 255, min_exponent - 1, min_exponent, min_exponent + 1, max_exponent - 1, max_exponent, max_exponent + 1}
    bits = [
        sign << 31 | exponent << 23 | (mantissa & 0x7FFFFF)
        for sign in (0, 1)
        for exponent in sorted(e for e in exponents if 0 <= e <= 255)
        for mantissa in sorted(m for m in mantissas if m >= 0)
    ]
    return np.array(bits, np.uint32).view(np.float32)


# Raw float32 bit patterns at the boundaries of each format, including
# exponent_bits == 1 and the full-precision identity.
@pytest.mark.parametrize("bits", FORMATS + [(8, 23)], ids=FORMAT_IDS + ["e8m23"])
def test_reduce_precision_float32_boundaries(bits):
    assert_matches_cpu(float32_boundaries(*bits), bits)

