"""Scatters of 8-bit integers along a non-last dimension run in int32: they
used to return all zeros (jnp.insert along axis 0 of a uint8 array). Results
must match XLA on the CPU.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def on(device, f, x):
    return np.asarray(jax.jit(f)(jax.device_put(x, jax.devices(device)[0])))


@pytest.mark.parametrize("dtype", [np.uint8, np.int8], ids=["u8", "i8"])
@pytest.mark.parametrize(
    "f",
    [
        lambda v: jnp.zeros((4, 3), v.dtype).at[jnp.array([0, 3])].set(v),
        lambda v: jnp.insert(v, 1, 9, axis=0),
        lambda v: jnp.insert(v, 1, 9, axis=1),
    ],
    ids=["scatter_rows", "insert_axis0", "insert_axis1"],
)
def test_8bit_scatter(dtype, f):
    x = np.arange(1, 7).reshape(2, 3).astype(dtype)
    np.testing.assert_array_equal(on("tt", f, x), on("cpu", f, x))
