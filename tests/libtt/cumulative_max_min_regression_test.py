"""lax.cummax and lax.cummin run on ttnn's accumulation kernel, like cumsum and
cumprod. They used to lower to a bf16 max pool, which rounded float32 results.

NaN inputs are not covered: like ttnn's elementwise max and min, the kernel
compares sign-magnitude bits, so cummin does not propagate +NaN (JAX does).
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
    if dtype == jnp.int32:
        # Beyond 2**24, so float32 arithmetic would round them.
        x = rng.integers(-(2**31), 2**31 - 1, SHAPE, dtype=np.int64).astype(np.int32)
        x[0, 0, :3] = [np.iinfo(np.int32).min, np.iinfo(np.int32).max, 16777217]
        return x
    # Values bf16 cannot hold, around 1e-3 as in testCumulativeReduce.
    x = rng.uniform(-1e-2, 1e-2, SHAPE).astype(np.float32)
    x[0, 0, :4] = [np.inf, -np.inf, -0.0, 1e-30]
    return x.astype(dtype)


def reference(np_op, x, axis, reverse):
    x = np.asarray(x, np.float64 if x.dtype != np.int32 else np.int64)
    if reverse:
        return np.flip(np_op(np.flip(x, axis), axis=axis), axis)
    return np_op(x, axis=axis)


@pytest.mark.parametrize("op", sorted(OPS))
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16, jnp.int32], ids=["f32", "bf16", "i32"])
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
    run = jax.jit(lambda x: jax.lax.cummax(x, axis=1), compiler_options={"export_path": str(tmp_path)})
    run(device_put(values(jnp.float32)))
    irs = [path.read_text() for path in (tmp_path / "irs").glob("ttnn*.mlir")]
    assert irs and any('"ttnn.cumulative"' in ir for ir in irs)
    assert not any("max_pool2d" in ir for ir in irs)
