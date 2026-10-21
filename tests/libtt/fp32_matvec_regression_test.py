"""Small float32 matrix-vector products retain all input mantissa bits."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize("dtype", [np.float32, np.float16, jnp.bfloat16])
@pytest.mark.parametrize("level", ["O0", "O1"])
@pytest.mark.parametrize("shape", [(5, 10), (32, 67), (1, 257)])
@pytest.mark.parametrize("vector_on_left", [False, True])
def test_small_fp32_matvec(level, shape, vector_on_left, dtype):
    rng = np.random.default_rng(12)
    a = rng.normal(size=shape).astype(dtype)
    v = rng.normal(size=shape[1]).astype(dtype)
    if vector_on_left:
        a = a.T
        run = lambda a, v: v @ a
        expected = v.astype(np.float64) @ a.astype(np.float64)
    else:
        run = lambda a, v: a @ v
        expected = a.astype(np.float64) @ v.astype(np.float64)
    got = jax.jit(run, compiler_options={"optimization_level": level})(
        *[jax.device_put(x, jax.devices("tt")[0]) for x in (a, v)])
    rtol = 2e-6 if dtype == np.float32 else (1e-3 if dtype == np.float16 else 1e-2)
    np.testing.assert_allclose(np.asarray(got).astype(np.float32),
                               expected.astype(dtype).astype(np.float32),
                               rtol=rtol, atol=2e-6)
