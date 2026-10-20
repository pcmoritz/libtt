"""Integer stablehlo.power runs as one exact device op: tt-metal's integer pow
kernels square and multiply Int32 and UInt32 tiles in registers, wrapping like
their multiplication, and narrower types widen to int32 and wrap back.
ttnn's float32 pow rounded results beyond 2^24 and saturated narrow types
(torch_tpu's pow on integers), and the previous fix expanded every power into
tensor ops for each exponent bit."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src.core import ShapedArray
from jax._src.lib.mlir.dialects import hlo
from jax.extend.core import Primitive
from jax.interpreters import mlir

# lax.pow only takes floating-point bases; stablehlo.power takes integers too.
integer_pow_p = Primitive("libtt_integer_pow")
integer_pow_p.def_abstract_eval(lambda x, y: ShapedArray(x.shape, x.dtype))
mlir.register_lowering(integer_pow_p, lambda ctx, x, y: [hlo.PowOp(x, y).result])

DTYPES = [np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32]
LEVELS = pytest.mark.parametrize("level", ["O0", "O1"])


def reference(x, y):
    # Exact powers with wrapping, as StableHLO defines them; negative
    # exponents give 1 for 1, +-1 for -1 and 0 otherwise.
    bits = np.iinfo(x.dtype).bits
    out = np.empty(x.shape, np.int64)
    for i, (b, e) in enumerate(zip(x.flat, y.flat)):
        b, e = int(b), int(e)
        if e < 0:
            out.flat[i] = 1 if b == 1 else (-1) ** (e & 1) if b == -1 else 0
        else:
            out.flat[i] = pow(b, e, 1 << bits)
    wrapped = out & ((1 << bits) - 1)
    if np.issubdtype(x.dtype, np.signedinteger):
        wrapped = np.where(wrapped >= 1 << (bits - 1), wrapped - (1 << bits), wrapped)
    return wrapped.astype(x.dtype)


def edge_cases(dtype):
    """(base, exponent) pairs at the edges of the exponent and value ranges."""
    info = np.iinfo(dtype)
    pairs = [(0, 0), (5, 0), (0, 3), (9, 9), (7, 9), (3, 7), (2, 7), (2, info.bits - 1),
             (2, info.bits), (3, info.max), (2, info.max), (info.max, 2), (info.max, info.max)]
    if np.issubdtype(dtype, np.signedinteger):
        pairs += [(1, -3), (-1, -3), (-1, -2), (5, -1), (0, -1), (-3, 3), (-2, info.bits - 1),
                  (info.min, 1), (info.min, 2), (-1, info.max), (-3, info.max)]
    else:
        # The top exponent bits: odd bases repeat with period 2^(bits-2).
        pairs += [(3, 1 << (info.bits - 1)), (3, (1 << (info.bits - 1)) + 1),
                  (2, 1 << (info.bits - 1)), (6, (1 << (info.bits - 1)) + 2)]
    if info.bits == 32:
        pairs += [(3, 1 << 30), (3, (1 << 30) + 5), (2, 1 << 30), (6, (1 << 30) + 1),
                  (12345, (1 << 30) - 1), (2, 31), (2, 32)]
    return pairs


def samples(dtype, shape=(32, 40)):
    rng = np.random.default_rng(0)
    info = np.iinfo(dtype)
    x = rng.integers(info.min, info.max, shape, dtype=np.int64, endpoint=True)
    y = rng.integers(max(info.min, -40), info.max, shape, dtype=np.int64, endpoint=True)
    # Small bases and exponents too, where powers neither vanish nor wrap.
    x[1::2] = rng.integers(max(info.min, -12), 12, x[1::2].shape, endpoint=True)
    y[1::2] = rng.integers(max(info.min, -3), 40, y[1::2].shape, endpoint=True)
    for i, (b, e) in enumerate(edge_cases(dtype)):
        x.flat[i], y.flat[i] = b, e
    return x.astype(dtype), y.astype(dtype)


def device_put(x):
    return jax.device_put(x, jax.devices("tt")[0])


@LEVELS
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: np.dtype(d).name)
def test_integer_pow(dtype, level):
    x, y = samples(dtype)
    run = jax.jit(integer_pow_p.bind, compiler_options={"optimization_level": level})
    got = np.asarray(run(device_put(x), device_put(y)))
    np.testing.assert_array_equal(got, reference(x, y))


@LEVELS
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: np.dtype(d).name)
def test_integer_pow_constant_exponent(tmp_path, dtype, level):
    """A constant exponent whose 32-bit sign bit is clear takes the scalar
    kernel (ttnn.pow_scalar), and one whose sign bit is set takes the tensor
    kernel (ttnn.pow_tensor). There negative signed exponents follow the
    negative-exponent rule, while uint32 exponents from 2^31 stay unsigned."""
    x, _ = samples(dtype)
    x.flat[:3] = np.array([0, 1, -1]).astype(dtype)
    info = np.iinfo(dtype)
    exponents = [0, 1, 2, 3, 9, info.bits - 1, info.bits, info.max]
    if info.bits == 32:
        exponents += [1 << 30, (1 << 30) + 3]
    if np.issubdtype(dtype, np.signedinteger):
        exponents += [-1, -2, info.min]
    if dtype == np.uint32:
        exponents += [1 << 31, (1 << 31) + 3]
    for exponent in exponents:
        export_path = tmp_path / str(exponent)
        run = jax.jit(
            lambda x: integer_pow_p.bind(x, jnp.full_like(x, exponent)),
            compiler_options={"optimization_level": level, "export_path": str(export_path)},
        )
        got = np.asarray(run(device_put(x)))
        np.testing.assert_array_equal(got, reference(x, np.full_like(x, exponent)), err_msg=f"exponent {exponent}")
        (ir,) = [path.read_text() for path in (export_path / "irs").glob("ttnn_[0-9]*.mlir")]
        if exponent in (0, 1):
            continue  # x**0 and x**1 may fold away.
        tensor = exponent < 0 or exponent >= 1 << 31
        routing = (ir.count("ttnn.pow_tensor"), ir.count("ttnn.pow_scalar"))
        assert routing == ((1, 0) if tensor else (0, 1)), f"exponent {exponent}: {ir}"


@pytest.mark.parametrize("level", ["O0", "O1"])
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16], ids=lambda d: np.dtype(d).name)
def test_unsigned_narrow_integer_pow_masks(tmp_path, dtype, level):
    """uint8 and uint16 powers widen to int32 and mask back to their width.
    (int8 and int16 are int32 on the device before the power is lowered.)"""
    x, y = samples(dtype)
    run = jax.jit(integer_pow_p.bind, compiler_options={"optimization_level": level, "export_path": str(tmp_path)})
    np.testing.assert_array_equal(np.asarray(run(device_put(x), device_put(y))), reference(x, y))
    (ir,) = [path.read_text() for path in (tmp_path / "irs").glob("ttnn_[0-9]*.mlir")]
    assert ir.count("ttnn.pow_tensor") == 1 and "ttnn.bitwise_and" in ir, ir


@pytest.mark.parametrize(
    "dtype,constant,op",
    [(np.int32, False, "ttnn.pow_tensor"), (np.int32, True, "ttnn.pow_scalar"), (np.uint32, False, "ttnn.pow_tensor")],
)
def test_integer_pow_is_one_op(tmp_path, dtype, constant, op):
    """The power stays one ttnn op on integer tiles, not a multiply chain."""
    x, y = samples(dtype)
    fn = (lambda x: integer_pow_p.bind(x, jnp.full_like(x, 7))) if constant else integer_pow_p.bind
    run = jax.jit(fn, compiler_options={"optimization_level": "O1", "export_path": str(tmp_path)})
    run(*(device_put(a) for a in ((x,) if constant else (x, y))))
    (ir,) = [path.read_text() for path in (tmp_path / "irs").glob("ttnn_[0-9]*.mlir")]
    assert ir.count(op) == 1, ir
    for expanded in ["ttnn.multiply", "ttnn.where", "ttnn.typecast"]:
        assert expanded not in ir, ir
