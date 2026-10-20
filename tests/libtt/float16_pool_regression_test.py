"""float16 average pooling of 2-D undilated windows runs on ttnn's pooling in
float16 instead of adding window slices. Plain float16 sums, other windows
and max pooling keep the exact window slices (ttnn's max pooling clears unused
rows to -inf, which float16 sources read as NaN)."""

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


def float16_values(shape, seed=0):
    """Random float16 values whose low mantissa bits bfloat16 drops, with
    ties, the largest finite values and infinities."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(shape).astype(np.float16)
    x = (x.view(np.uint16) | np.uint16(0x0005)).view(np.float16)
    flat = x.reshape(-1)
    flat[:6] = [65504, -65504, np.inf, -np.inf, 1.0009765625, 1.0009765625]
    flat[6:12] = flat[12:18]  # ties
    return x


def run(fn, level, *args, export_path=None):
    options = {"optimization_level": level}
    if export_path:
        options["export_path"] = str(export_path)
    return np.asarray(jax.jit(fn, compiler_options=options)(*(jax.device_put(a, jax.devices("tt")[0]) for a in args)))


def max_pool(window, strides, padding):
    return lambda x: lax.reduce_window(x, -jnp.inf, lax.max, window, strides, padding)


def sum_pool(window, strides, padding):
    return lambda x: lax.reduce_window(x, jnp.zeros((), x.dtype), lax.add, window, strides, padding)


# At O1 the float16 slices are typecast, and a padded window holding +inf
# gives NaN (as on the commit before float16 average pooling).
O1_PADDED_INF = pytest.mark.xfail(strict=True, reason="O1 typecasts float16 max-pool slices: +inf gives NaN")
MAX_CASES = [
    pytest.param(*case, level, id=f"{name}-{level}",
                 marks=[O1_PADDED_INF] if level == "O1" and any(map(any, case[3])) else [])
    for case, name in zip(WINDOWS, IDS)
    for level in ["O0", "O1"]
]


@pytest.mark.parametrize("shape,window,strides,padding,level", MAX_CASES)
def test_float16_max_pool(shape, window, strides, padding, level):
    x = float16_values(shape)
    fn = max_pool(window, strides, padding)
    np.testing.assert_array_equal(run(fn, level, x), np.asarray(jax.jit(fn, backend="cpu")(x)))


@LEVELS
@pytest.mark.parametrize("average", [False, True], ids=["sum", "avg"])
@pytest.mark.parametrize("shape,window,strides,padding", WINDOWS, ids=IDS)
def test_float16_sum_pool(shape, window, strides, padding, average, level):
    x = np.random.default_rng(1).standard_normal(shape).astype(np.float16)
    size = np.prod(window)
    fn = sum_pool(window, strides, padding)
    if average:
        fn = (lambda f: lambda x: f(x) / size)(fn)
    # float32 holds these sums of float16 values exactly. Summing the window in
    # float16, with the scaling and output roundings, may err by (size + 2)
    # float16 unit roundoffs of the sum of magnitudes.
    reference = jax.jit(sum_pool(window, strides, padding), backend="cpu")
    scale = 1 / size if average else 1
    expected = np.asarray(reference(x.astype(np.float32))) * scale
    bound = (size + 2) * 2.0**-11 * np.asarray(reference(np.abs(x).astype(np.float32))) * scale
    got = run(fn, level, x)
    assert got.dtype == np.float16
    error = np.abs(got.astype(np.float32) - expected)
    assert np.all(error <= bound), f"{np.sum(error > bound)} beyond the bound, worst {np.max(error - bound)}"


@pytest.mark.parametrize(
    "window,dilation",
    [((1, 2, 2, 1), (1, 2, 2, 1)), ((1, 2, 2, 2), (1, 1, 1, 1))],
    ids=["dilated", "3d"],
)
def test_float16_sum_pool_without_ttnn_pooling(window, dilation):
    """Windows ttnn does not pool keep adding window slices."""
    x = np.random.default_rng(2).standard_normal((1, 6, 6, 4)).astype(np.float16)
    fn = lambda x: lax.reduce_window(x, jnp.zeros((), x.dtype), lax.add, window, (1,) * 4, "VALID", window_dilation=dilation)
    expected = jax.jit(fn, backend="cpu")(x.astype(np.float32))
    np.testing.assert_allclose(run(fn, "O0", x).astype(np.float32), np.asarray(expected), rtol=2e-3, atol=1e-2)


def test_float16_avg_pool_stays_float16(tmp_path):
    shape, window, strides, padding = WINDOWS[0]
    x = np.random.default_rng(3).standard_normal(shape).astype(np.float16)
    run(lambda x: sum_pool(window, strides, padding)(x) / 9, "O1", x, export_path=tmp_path)
    (ir,) = [path.read_text() for path in (tmp_path / "irs").glob("ttnn_[0-9]*.mlir")]
    pool = [line for line in ir.splitlines() if '"ttnn.avg_pool2d"' in line]
    assert len(pool) == 1 and "xf16" in pool[0] and "bf16" not in pool[0], ir
    assert "ttnn.typecast" not in ir, ir
