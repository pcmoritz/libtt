"""stablehlo.compare with compare_type TOTALORDER orders
-NaN < -inf < ... < -0 < +0 < ... < +inf < +NaN, NaNs by payload, which
ordinary float comparisons do not. libtt runs it as a native float32
total-order comparison on Blackhole.

The direct tests below emit such comparisons for all six directions. The
searchsorted test checks an integration: JAX canonicalizes zeros and NaNs
before comparing, so there it amounts to NumPy's NaNs-last convention."""

from functools import partial

import jax
from jax import core
from jax._src.lax import lax as lax_internal
from jax.extend.core import Primitive
from jax.interpreters import mlir
import jax.numpy as jnp
import numpy as np
import pytest

DIRECTIONS = ["EQ", "NE", "LT", "LE", "GT", "GE"]
LAX_OPS = {
    "EQ": jax.lax.eq,
    "NE": jax.lax.ne,
    "LT": jax.lax.lt,
    "LE": jax.lax.le,
    "GT": jax.lax.gt,
    "GE": jax.lax.ge,
}
NUMPY_OPS = {
    "EQ": np.equal,
    "NE": np.not_equal,
    "LT": np.less,
    "LE": np.less_equal,
    "GT": np.greater,
    "GE": np.greater_equal,
}


def _total_order_compare(direction):
    prim = Primitive(f"test_total_order_{direction.lower()}")
    prim.def_abstract_eval(lambda x, y: core.ShapedArray(x.shape, np.bool_))
    mlir.register_lowering(prim, partial(lax_internal._compare_lower_hlo, direction, True))
    return prim


TOTAL_ORDER_COMPARE = {d: _total_order_compare(d) for d in DIRECTIONS}

# float32 bit patterns: NaNs of both signs with different payloads, infinities,
# signed zeros, and adjacent values.
SPECIAL_BITS = np.array(
    [
        0xFFFFFFFF,  # -NaN, largest payload
        0xFFC00000,  # -NaN, quiet
        0xFF800001,  # -NaN, smallest payload
        0xFF800000,  # -inf
        0xBF800001,  # next float below -1
        0xBF800000,  # -1
        0x80000000,  # -0
        0x00000000,  # +0
        0x3F800000,  # 1
        0x3F800001,  # next float above 1
        0x7F800000,  # +inf
        0x7F800001,  # +NaN, smallest payload
        0x7FC00000,  # +NaN, quiet
        0x7FFFFFFF,  # +NaN, largest payload
    ],
    np.uint32,
)
SPECIAL = SPECIAL_BITS.view(np.float32)


# Every pair of special values, as two full operands.
X, Y = (np.ascontiguousarray(a) for a in np.meshgrid(SPECIAL, SPECIAL, indexing="ij"))


def device_put(x):
    return jax.device_put(x, jax.devices("tt")[0])


def total_order_keys(x):
    bits = x.view(np.int32)
    return bits ^ ((bits >> 31) & 0x7FFFFFFF)


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_total_order_compare(direction):
    got = jax.jit(TOTAL_ORDER_COMPARE[direction].bind)(device_put(X), device_put(Y))
    expected = NUMPY_OPS[direction](total_order_keys(X), total_order_keys(Y))
    np.testing.assert_array_equal(np.asarray(got), expected)


def test_total_order_compare_runs_natively(tmp_path):
    # The comparison reaches TTNN as one marked op, without bit-key arithmetic.
    run = jax.jit(TOTAL_ORDER_COMPARE["LT"].bind, compiler_options={"export_path": str(tmp_path)})
    run(device_put(X), device_put(Y))
    irs = [path.read_text() for path in (tmp_path / "irs").glob("ttnn*.mlir")]
    assert irs and any("ttcore.total_order" in ir for ir in irs)
    assert not any("bitcast_convert" in ir for ir in irs)


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_float_compare_unchanged(direction):
    got = jax.jit(LAX_OPS[direction])(device_put(X), device_put(Y))
    np.testing.assert_array_equal(np.asarray(got), NUMPY_OPS[direction](X, Y))


@pytest.mark.parametrize("side", ["left", "right"])
def test_searchsorted_special_values(side):
    # bfloat16 is not covered: on device its NaN compares equal to inf even
    # before the comparison, a separate issue.
    a = np.sort(np.array([np.nan, -np.inf, -1.5, -0.0, 0.0, 1e-30, 2.5, np.inf, 1.0, np.nan, 2.5], np.float32))
    v = np.array([np.nan, -np.inf, -1.5, -0.0, 0.0, 1e-30, 2.5, np.inf, 3.0, -2.0], np.float32)
    got = jax.jit(lambda a, v: jnp.searchsorted(a, v, side=side))(device_put(a), device_put(v))
    np.testing.assert_array_equal(np.asarray(got), np.searchsorted(a, v, side=side))
