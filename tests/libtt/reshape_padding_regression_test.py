"""Reshape padding must not contribute stale values to a subsequent matmul."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize("level,shape", [
    ("O0", (3, 5, 4)), ("O1", (3, 5, 4)), ("O0", (33, 67, 35)),
])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16, jnp.float16])
def test_reshape_then_matmul(level, dtype, shape):
    m, k, n = shape
    a = (np.arange(m * k) % 5).astype(dtype)
    b = (np.arange(k * n) % 3).astype(dtype)
    expected = ((a.astype(np.float32) - 2).reshape(m, k)
                @ (b.astype(np.float32) - 1).reshape(k, n)).astype(dtype)
    options = {"optimization_level": level}
    subtract = jax.jit(lambda x, offset: x - offset, compiler_options=options)
    matmul = jax.jit(lambda x, y: x.reshape(m, k) @ y.reshape(k, n),
                     compiler_options=options)
    device = jax.devices("tt")[0]
    # Device-produced inputs and repeated calls expose reused scratch padding.
    for _ in range(3):
        x = subtract(jax.device_put(a, device), jax.device_put(np.array(2, dtype), device))
        y = subtract(jax.device_put(b, device), jax.device_put(np.array(1, dtype), device))
        np.testing.assert_array_equal(np.asarray(matmul(x, y)), expected)
