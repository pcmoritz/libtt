"""uint8 argmax and argmin preserve values above 127 and first-index ties."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

X = np.array([[4, 5, 6, 4, 7], [9, 9, 5, 6, 0], [1, 200, 3, 255, 2]], np.uint8)
WIDE = np.random.default_rng(0).integers(0, 256, (8, 32768), dtype=np.uint8)


@pytest.mark.parametrize("level", ["O0", "O1"])
@pytest.mark.parametrize("op", ["argmax", "argmin"])
@pytest.mark.parametrize("x, axis", [(X, None), (X, 0), (X, 1), (WIDE, -1)], ids=["all", "0", "1", "wide"])
def test_uint8_argmax(x, axis, op, level):
    f = jax.jit(
        lambda a: getattr(jnp, op)(a, axis=axis),
        compiler_options={"optimization_level": level},
    )
    got = np.asarray(f(jax.device_put(x, jax.devices("tt")[0])))
    np.testing.assert_array_equal(got, getattr(np, op)(x, axis=axis))
