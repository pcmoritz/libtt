"""Complex transposes preserve the real/imaginary pair and conjugate views."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize("level", ["O0", "O1"])
@pytest.mark.parametrize("conjugate", [False, True])
@pytest.mark.parametrize("shape,axes", [((2, 3), (1, 0)), ((2, 3, 4), (2, 0, 1))])
def test_complex_transpose(level, conjugate, shape, axes):
    real = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    x = (real + 1j * (0.5 - real)).astype(np.complex64)
    expected = (np.conj(x) if conjugate else x).transpose(axes)
    run = jax.jit(
        lambda x: jnp.transpose(jnp.conj(x) if conjugate else x, axes),
        compiler_options={"optimization_level": level},
    )
    got = run(jax.device_put(x, jax.devices("tt")[0]))
    np.testing.assert_array_equal(got, expected)
