"""A permute that moves size-1 dimensions runs on the other dimensions alone,
between reshapes, since tiles pad a trailing size-1 dimension 32 times
(TorchTitan's attention permutes [S, H, D, 1] views of [S, H, D])."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize(
    "shape,permutation",
    [
        ((64, 8, 128, 1), (1, 0, 3, 2)),
        ((64, 8, 128, 1), (3, 1, 0, 2)),
        ((1, 64, 8, 128), (0, 2, 1, 3)),
        ((64, 1, 96), (2, 1, 0)),
        # Only size-1 dimensions move: a reshape.
        ((1, 1, 32, 64), (1, 0, 2, 3)),
        ((32, 1, 64, 1), (1, 0, 3, 2)),
    ],
)
@pytest.mark.parametrize("dtype", [jnp.int32, jnp.bfloat16])
def test_permute_unit_dims(shape, permutation, dtype):
    x = (np.arange(np.prod(shape)) % 251).astype(dtype).reshape(shape)
    device = jax.devices("tt")[0]
    got = jax.jit(lambda x: jnp.transpose(x + 1, permutation))(jax.device_put(x, device))
    np.testing.assert_array_equal(np.asarray(got), np.transpose(x + 1, permutation))
