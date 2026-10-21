"""Small float32 and float16 vector products retain their input precision."""

import jax
import numpy as np
import pytest


@pytest.mark.parametrize("dtype", [np.float32, np.float16])
@pytest.mark.parametrize("level", ["O0", "O1"])
@pytest.mark.parametrize("shape", [(5, 10), (32, 67), (1, 257), (1, 4096)])
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
    rtol = 2e-6 if dtype == np.float32 else 1e-3
    np.testing.assert_allclose(np.asarray(got).astype(np.float32),
                               expected.astype(dtype).astype(np.float32),
                               rtol=rtol, atol=2e-6)


@pytest.mark.parametrize("level", ["O0", "O1"])
def test_float16_matvec_preserves_low_mantissa_bit(level):
    # Native matmul loses this operand bit even with HiFi4 and FP32 accumulation.
    value = np.nextafter(np.float16(1), np.float16(2))
    a = np.ones((1, 2), dtype=np.float16)
    v = np.array([value, 0], dtype=np.float16)
    got = jax.jit(lambda a, v: a @ v, compiler_options={"optimization_level": level})(
        *[jax.device_put(x, jax.devices("tt")[0]) for x in (a, v)])
    np.testing.assert_array_equal(got, [value])
