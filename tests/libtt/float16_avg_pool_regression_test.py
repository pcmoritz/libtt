"""Native float16 average pooling preserves the input and output dtype."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

LEVELS = pytest.mark.parametrize("level", ["O0", "O1"])
# NHWC shape, window, strides, padding
WINDOWS = [
    ((1, 9, 11, 32), (1, 3, 3, 1), (1, 2, 2, 1), ((0, 0), (1, 1), (1, 1), (0, 0))),
    ((2, 8, 8, 48), (1, 2, 2, 1), (1, 2, 2, 1), ((0, 0),) * 4),
    ((1, 7, 9, 16), (1, 3, 2, 1), (1, 1, 1, 1), ((0, 0), (1, 1), (0, 1), (0, 0))),
    # The whole input: ttnn's global pooling.
    ((2, 7, 7, 64), (1, 7, 7, 1), (1, 1, 1, 1), ((0, 0),) * 4),
]
IDS = ["3x3s2p1", "2x2s2", "3x2s1p", "global"]


def run(fn, level, *args, export_path=None):
    options = {"optimization_level": level}
    if export_path:
        options["export_path"] = str(export_path)
    return np.asarray(jax.jit(fn, compiler_options=options)(*(jax.device_put(a, jax.devices("tt")[0]) for a in args)))


def sum_pool(window, strides, padding):
    return lambda x: lax.reduce_window(x, jnp.zeros((), x.dtype), lax.add, window, strides, padding)


@LEVELS
@pytest.mark.parametrize("shape,window,strides,padding", WINDOWS, ids=IDS)
def test_float16_avg_pool(shape, window, strides, padding, level):
    x = np.random.default_rng(1).standard_normal(shape).astype(np.float16)
    size = np.prod(window)
    fn = lambda x: sum_pool(window, strides, padding)(x) / size
    # float32 holds these sums of float16 values exactly. Summing the window in
    # float16, with the scaling and output roundings, may err by (size + 2)
    # float16 unit roundoffs of the sum of magnitudes.
    reference = jax.jit(sum_pool(window, strides, padding), backend="cpu")
    scale = 1 / size
    expected = np.asarray(reference(x.astype(np.float32))) * scale
    bound = (size + 2) * 2.0**-11 * np.asarray(reference(np.abs(x).astype(np.float32))) * scale
    got = run(fn, level, x)
    assert got.dtype == np.float16
    error = np.abs(got.astype(np.float32) - expected)
    assert np.all(error <= bound), f"{np.sum(error > bound)} beyond the bound, worst {np.max(error - bound)}"


@LEVELS
@pytest.mark.parametrize(
    "dtype,pool_dtype",
    [(np.float16, "f16"), (jnp.bfloat16, "bf16"), (np.float32, "bf16")],
    ids=["float16", "bfloat16", "float32-fallback"],
)
def test_avg_pool_dtype_routing(tmp_path, level, dtype, pool_dtype):
    shape, window, strides, padding = WINDOWS[0]
    x = np.random.default_rng(3).standard_normal(shape).astype(dtype)
    fn = lambda x: sum_pool(window, strides, padding)(x) / 9
    got = run(fn, level, x, export_path=tmp_path)
    assert got.dtype == x.dtype
    expected = jax.jit(fn, backend="cpu")(x.astype(np.float32))
    np.testing.assert_allclose(got.astype(np.float32), expected, rtol=2e-2, atol=2e-2)
    (ir,) = [path.read_text() for path in (tmp_path / "irs").glob("ttnn_[0-9]*.mlir")]
    pool = [line for line in ir.splitlines() if '"ttnn.avg_pool2d"' in line]
    assert len(pool) == 1 and f"x{pool_dtype}" in pool[0], ir
    # Native float32 pooling is not supported yet; its existing bf16 fallback
    # remains explicit while float16 and bfloat16 need no dtype conversions.
    if dtype == np.float32:
        assert "ttnn.typecast" in ir, ir
    else:
        assert "ttnn.typecast" not in ir, ir
