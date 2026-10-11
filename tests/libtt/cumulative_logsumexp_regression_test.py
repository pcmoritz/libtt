"""lax.cumlogsumexp runs on ttnn's accumulation kernel as a running logaddexp,
computed as max(a, b) + log1p(exp(-|a - b|)), so large values don't overflow and
small ones aren't lost. It used to fail to lower (a reduce_window with JAX's
logaddexp as its body).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

SHAPE = (3, 40, 70)  # not tile-aligned
TOLERANCE = {jnp.float32: 1e-5, jnp.bfloat16: 2e-2, jnp.float16: 2e-3}


def device_put(x):
    return jax.device_put(x, jax.devices("tt")[0])


def reference(x, axis, reverse):
    x = np.asarray(x, np.float64)
    if reverse:
        return np.flip(np.logaddexp.accumulate(np.flip(x, axis), axis=axis), axis)
    return np.logaddexp.accumulate(x, axis=axis)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16, jnp.float16], ids=["f32", "bf16", "f16"])
@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("reverse", [False, True])
def test_cumlogsumexp(dtype, axis, reverse):
    x = (np.random.default_rng(0).normal(scale=3.0, size=SHAPE)).astype(dtype)
    got = jax.jit(lambda x: jax.lax.cumlogsumexp(x, axis=axis, reverse=reverse))(device_put(x))
    assert got.dtype == x.dtype
    tol = TOLERANCE[dtype]
    np.testing.assert_allclose(np.asarray(got).astype(np.float64), reference(x, axis, reverse), rtol=tol, atol=tol)


# log(0) = -inf leaves the sum unchanged, values far apart keep the larger one, large values don't
# overflow, +inf saturates and NaN propagates.
@pytest.mark.parametrize(
    "row",
    [
        [-np.inf, -np.inf, 0.0, -np.inf, 1.0],
        [1000.0, 1000.0, -1000.0, 999.0, 0.0],
        [-80.0, -100.0, 0.0, 88.0, 88.5],
        [1.0, np.inf, 2.0, -np.inf, 3.0],
        [1.0, 2.0, np.nan, 3.0, -np.inf],
    ],
    ids=["neg_inf", "large", "far_apart", "pos_inf", "nan"],
)
@pytest.mark.parametrize("reverse", [False, True])
def test_cumlogsumexp_special_values(row, reverse):
    x = np.array([row], np.float32)
    got = jax.jit(lambda x: jax.lax.cumlogsumexp(x, axis=1, reverse=reverse))(device_put(x))
    np.testing.assert_allclose(np.asarray(got).astype(np.float64), reference(x, 1, reverse), rtol=1e-5, atol=1e-5)


def test_cumlogsumexp_runs_on_accumulation(tmp_path):
    run = jax.jit(lambda x: jax.lax.cumlogsumexp(x, axis=1, reverse=True), compiler_options={"export_path": str(tmp_path)})
    run(device_put(np.zeros(SHAPE, np.float32)))
    irs = [path.read_text() for path in (tmp_path / "irs").glob("ttnn*.mlir")]
    assert irs and any('"ttnn.cumulative"' in ir and "logsumexp" in ir for ir in irs)
    assert not any('"ttnn.reverse"' in ir for ir in irs)
