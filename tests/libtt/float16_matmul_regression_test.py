"""Float16 matmuls must stay exact when K spans several blocks under the optimizer's default configs."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize("level", ["O0", "O1"])
@pytest.mark.parametrize("k", [64, 96, 256])
@pytest.mark.parametrize("dtype", [jnp.float16, jnp.float32])
def test_matmul_to_float16(dtype, k, level):
    a = (np.arange(32 * k) % 5 - 2).reshape(32, k).astype(np.float32)
    b = (np.arange(k * 32) % 3 - 1).reshape(k, 32).astype(np.float32)
    expected = (a @ b).astype(np.float16)
    # Float16 operands, or float32 operands with a float16 result.
    matmul = jax.jit(lambda x, y: jax.lax.dot(x, y, preferred_element_type=jnp.float16),
                     compiler_options={"optimization_level": level})
    got = matmul(jnp.asarray(a, dtype), jnp.asarray(b, dtype))
    np.testing.assert_array_equal(np.asarray(got), expected)
