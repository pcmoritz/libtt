"""lax.cummax and lax.cummin run on ttnn's accumulation kernel, like cumsum and
cumprod. They used to lower to a bf16 max pool, which rounded float32 results.

NaN propagates through both, from NaNs of either sign, as in JAX.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

OPS = {
    "cummax": (jax.lax.cummax, np.maximum.accumulate),
    "cummin": (jax.lax.cummin, np.minimum.accumulate),
}
SHAPE = (3, 40, 70)  # not tile-aligned


def device_put(x):
    return jax.device_put(x, jax.devices("tt")[0])


def values(dtype):
    rng = np.random.default_rng(0)
    if np.issubdtype(dtype, np.integer):
        # The full range: beyond 2**24, so float32 arithmetic would round them,
        # and for uint32 beyond INT32_MAX.
        info = np.iinfo(dtype)
        x = rng.integers(info.min, info.max, SHAPE, dtype=np.int64, endpoint=True).astype(dtype)
        x[0, 0, :2] = [info.min, info.max]
        if dtype == np.uint32:
            x[0, 1, :2] = [2**31, 2**31 - 1]
        return x
    # Values bf16 cannot hold, around 1e-3 as in testCumulativeReduce.
    x = rng.uniform(-1e-2, 1e-2, SHAPE).astype(np.float32)
    x[0, 0, :4] = [np.inf, -np.inf, -0.0, 1e-30]
    return x.astype(dtype)


def reference(np_op, x, axis, reverse):
    x = np.asarray(x, np.int64 if np.issubdtype(x.dtype, np.integer) else np.float64)
    if reverse:
        return np.flip(np_op(np.flip(x, axis), axis=axis), axis)
    return np_op(x, axis=axis)


@pytest.mark.parametrize("op", sorted(OPS))
@pytest.mark.parametrize(
    "dtype",
    [jnp.float32, jnp.bfloat16, jnp.float16, jnp.int32, jnp.uint32, jnp.uint16, jnp.uint8],
    ids=["f32", "bf16", "f16", "i32", "u32", "u16", "u8"],
)
@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("reverse", [False, True])
def test_cumulative_extremum(op, dtype, axis, reverse):
    lax_op, np_op = OPS[op]
    x = values(dtype)
    got = jax.jit(lambda x: lax_op(x, axis=axis, reverse=reverse))(device_put(x))
    assert got.dtype == x.dtype
    expected = reference(np_op, x, axis, reverse)
    # A running max or min only selects input values, so it is exact.
    np.testing.assert_array_equal(np.asarray(got).astype(expected.dtype), expected)


def test_cumulative_extremum_runs_on_accumulation(tmp_path):
    # A reverse scan runs in the kernel, without reversing the tensor.
    run = jax.jit(lambda x: jax.lax.cummax(x, axis=1, reverse=True), compiler_options={"export_path": str(tmp_path)})
    run(device_put(values(jnp.float32)))
    irs = [path.read_text() for path in (tmp_path / "irs").glob("ttnn*.mlir")]
    assert irs and any('"ttnn.cumulative"' in ir for ir in irs)
    assert not any("max_pool2d" in ir or '"ttnn.reverse"' in ir for ir in irs)


@pytest.mark.parametrize("op", sorted(OPS))
@pytest.mark.parametrize("sign", [1, -1], ids=["pos_nan", "neg_nan"])
# bfloat16 is not covered: on device, computing on a bfloat16 NaN turns it into
# inf before any op sees it (even x * 1), a separate issue.
@pytest.mark.parametrize("dtype", [jnp.float32], ids=["f32"])
@pytest.mark.parametrize("reverse", [False, True])
def test_cumulative_extremum_propagates_nan(op, sign, dtype, reverse):
    lax_op, _ = OPS[op]
    nan = np.copysign(np.float32(np.nan), np.float32(sign))
    x = np.array([[1.0, -np.inf, nan, 2.0, np.inf, -5.0, 0.5]], np.float32)
    if reverse:
        x = x[:, ::-1].copy()
    got = np.asarray(jax.jit(lambda x: lax_op(x, axis=1, reverse=reverse))(device_put(x.astype(dtype))))
    expected = [[False, False, True, True, True, True, True]]
    np.testing.assert_array_equal(np.isnan(got.astype(np.float32)), expected if not reverse else [expected[0][::-1]])
