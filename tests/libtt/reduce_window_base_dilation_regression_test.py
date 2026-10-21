"""Base dilation in reduce-window reuses interior padding."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize("level", ["O0", "O1"])
@pytest.mark.parametrize("shape,dilation,window", [
    ((1, 1, 4, 4), (1, 1, 2, 2), (1, 1, 2, 2)),
    ((1, 1, 1, 1, 1), (1, 1, 4, 4, 4), (1, 1, 2, 2, 2)),
    ((1, 1, 3, 4), (1, 1, 2, 3), (1, 1, 2, 2)),
])
def test_sum_window_with_base_dilation(level, shape, dilation, window):
    x = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) / 4
    padding = [(0, 0), (0, 0)] + [(1, 1)] * (len(shape) - 2)

    def reduce(a):
        return jax.lax.reduce_window(
            a, jnp.float32(0), jax.lax.add, window,
            (1,) * len(shape), padding, base_dilation=dilation)

    expected = jax.jit(reduce)(jax.device_put(x, jax.devices("cpu")[0]))
    actual = jax.jit(reduce, compiler_options={"optimization_level": level})(
        jax.device_put(x, jax.devices("tt")[0]))
    np.testing.assert_array_equal(actual, expected)
