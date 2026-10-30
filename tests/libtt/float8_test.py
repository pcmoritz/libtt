"""float8 tensors: storage, conversions and native matmuls.

float8 tensors are stored as their bits. Conversions between float8 and f32
are exact on the device (ttnn's typecast), so they match ml_dtypes (and
PyTorch): rounding to nearest even, subnormals, NaN on e4m3fn overflow and
infinity on e5m2 overflow, at O0 and O1 (where casts through float8 must not
fold). A matmul of two float8 matrices reads them
natively on the matrix unit, which flushes subnormal operands to zero.
"""

import jax
from jax import core
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import hlo
from jax.extend.core import Primitive
from jax.interpreters import mlir
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest

FORMATS = [jnp.float8_e4m3fn, jnp.float8_e5m2]
LEVELS = pytest.mark.parametrize("level", ["O0", "O1"])


def jit(f, level):
    return jax.jit(f, compiler_options={"optimization_level": level})


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


@LEVELS
@pytest.mark.parametrize("dtype", FORMATS)
def test_decode_every_code(dtype, level):
    codes = all_codes(dtype)
    decoded = jit(lambda x: x.astype(jnp.float32), level)(codes)
    assert_same_bits(decoded, codes.astype(np.float32))
    to_bf16 = jit(lambda x: x.astype(jnp.bfloat16), level)(codes)
    assert_same_bits(to_bf16, codes.astype(jnp.bfloat16))


def midpoints(dtype):
    """Each rounding midpoint between neighboring values of the format (and past its largest),
    with its float32 neighbors."""
    values = np.arange(256, dtype=np.uint8).view(dtype).astype(np.float64)
    finite = np.unique(np.abs(values[np.isfinite(values)]))
    finite = np.append(finite, 2 * finite[-1] - finite[-2])
    mid = ((finite[:-1] + finite[1:]) / 2).astype(np.float32)
    return np.concatenate([mid, np.nextafter(mid, np.float32(0)), np.nextafter(mid, np.float32(np.inf))])


def encode_inputs():
    """Every bfloat16, random float32 values over the formats' ranges and random float32 bit
    patterns, both formats' rounding midpoints with their neighbors, and edges."""
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
    bits = rng.integers(0, 1 << 32, size=1 << 14, dtype=np.uint32).view(np.float32)
    mids = np.concatenate([midpoints(dtype) for dtype in FORMATS])
    values = np.concatenate([bf16, random, bits, mids, -mids, edges, -edges])
    return np.pad(values, (0, -len(values) % 1024)).reshape(-1, 32)


@LEVELS
@pytest.mark.parametrize("dtype", FORMATS)
def test_encode_rounds_like_ml_dtypes(dtype, level):
    values = encode_inputs()
    encoded = jit(lambda x: x.astype(dtype), level)(values)
    assert_same_bits(encoded, values.astype(dtype))


@LEVELS
@pytest.mark.parametrize("dtype", FORMATS)
def test_encode_rounding_survives_a_following_cast(dtype, level):
    """f32 -> float8 -> bfloat16 keeps the float8 rounding."""
    values = encode_inputs()
    rounded = jit(lambda x: x.astype(dtype).astype(jnp.bfloat16), level)(values)
    assert_same_bits(rounded, values.astype(dtype).astype(jnp.bfloat16))


@pytest.mark.parametrize("dtype", FORMATS)
def test_arithmetic_on_float8(dtype):
    rng = np.random.default_rng(1)
    x = rng.normal(size=(64, 32)).astype(np.float32).astype(dtype)
    y = rng.normal(size=(64, 32)).astype(np.float32).astype(dtype)
    total, larger = jax.jit(lambda x, y: (x + y, x > y))(x, y)
    assert_same_bits(total, (x.astype(np.float32) + y.astype(np.float32)).astype(dtype))
    np.testing.assert_array_equal(larger, x.astype(np.float32) > y.astype(np.float32))


# stablehlo.custom_call @mhlo.topk, which tt-mlir lowers to ttir.topk. (JAX's lax.top_k is a
# chlo.top_k composite of a sort instead.)
mhlo_topk_p = Primitive("test_mhlo_topk")
mhlo_topk_p.multiple_results = True
mhlo_topk_p.def_abstract_eval(
    lambda x, *, k: (
        core.ShapedArray(x.shape[:-1] + (k,), x.dtype),
        core.ShapedArray(x.shape[:-1] + (k,), np.int32),
    )
)


def mhlo_topk_lowering(ctx, x, *, k):
    op = hlo.CustomCallOp(
        [mlir.aval_to_ir_type(ctx.module_context, aval) for aval in ctx.avals_out],
        [x],
        call_target_name=ir.StringAttr.get("mhlo.topk"),
    )
    op.attributes["mhlo.attributes"] = ir.DictAttr.get(
        {"k": ir.IntegerAttr.get(ir.IntegerType.get_signless(64), k)}
    )
    return op.results


mlir.register_lowering(mhlo_topk_p, mhlo_topk_lowering)


@pytest.mark.parametrize("dtype", FORMATS)
def test_custom_call_top_k_compares_values(dtype):
    """A custom call that computes, such as mhlo.topk, gets values, not their bits: a negative
    value's code is the larger one (-1 is 0xb8 in float8_e4m3fn, +1 is 0x38)."""
    codes = np.concatenate([np.arange(0x30, 0x50), np.arange(0xB0, 0xD0)]).astype(np.uint8)
    x = np.random.default_rng(5).permuted(np.tile(codes, (2, 1)), axis=1).view(dtype)
    values, indices = jax.jit(lambda x: mhlo_topk_p.bind(x, k=8))(x)
    expected = np.argsort(-x.astype(np.float32), axis=1)[:, :8]
    np.testing.assert_array_equal(indices, expected)
    assert_same_bits(values, np.take_along_axis(x, expected, axis=1))


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


@LEVELS
@pytest.mark.parametrize("dtype", FORMATS)
def test_native_float8_matmul_rounds_output(dtype, level):
    """Exact integer products isolate rounding of the final float8 result."""
    rng = np.random.default_rng(7)
    a = rng.integers(-2, 3, (32, 64)).astype(dtype)
    b = rng.integers(-2, 3, (64, 32)).astype(dtype)
    product = jit(lambda a, b: jnp.dot(a, b), level)(a, b)
    expected = (a.astype(np.float32) @ b.astype(np.float32)).astype(dtype)
    assert_same_bits(product, expected)


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
