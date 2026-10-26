"""Tile padding of matmul operands must not contribute to the result along K."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

CASES = [("O0", (3, 5, 4)), ("O1", (3, 5, 4)), ("O0", (33, 67, 35))]
DTYPES = [jnp.float32, jnp.bfloat16, jnp.float16]


@pytest.mark.parametrize("level,shape", CASES)
@pytest.mark.parametrize("dtype", DTYPES)
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


@pytest.mark.parametrize("level,shape", CASES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_elementwise_then_matmul(level, dtype, shape):
    m, k, n = shape
    a = (np.arange(m * k) % 5).reshape(m, k).astype(np.float32)
    b = (np.arange(k * n) % 3).reshape(k, n).astype(np.float32)
    expected = ((a - 2) @ (b - 1)).astype(dtype)
    # The subtractions also map both operands' zero padding to -2 and -1.
    matmul = jax.jit(lambda x, y: (x - 2) @ (y - 1),
                     compiler_options={"optimization_level": level})
    got = matmul(jnp.asarray(a, dtype), jnp.asarray(b, dtype))
    np.testing.assert_array_equal(np.asarray(got), expected)


@pytest.mark.parametrize("level", ["O0", "O1"])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("junk", [np.nan, np.inf, -np.inf])
def test_zero_times_nonfinite_is_zero(level, dtype, junk):
    # Matmul zeroes only A's K padding; stale NaN or Inf in B's padding meets those zeros.
    a = (np.arange(32 * 64) % 5 - 2).reshape(32, 64).astype(np.float32)
    b = (np.arange(64 * 32) % 3 - 1).reshape(64, 32).astype(np.float32)
    a[:, 35:] = 0
    b[35:] = junk
    expected = (a[:, :35] @ b[:35]).astype(dtype)
    matmul = jax.jit(jnp.matmul, compiler_options={"optimization_level": level})
    got = matmul(jnp.asarray(a, dtype), jnp.asarray(b, dtype))
    np.testing.assert_array_equal(np.asarray(got), expected)
