"""A uint8 buffer that a row-major reader moved to the device can be a tiled
argument of a later program.

A scatter reads its uint8 update row-major, so its first use moves the buffer
to the device in that layout. Program arguments are otherwise tiled, and TTNN
can't tilize uint8 on device, so the next program aborted ("to_layout does not
support device to device tilize"), as torch_tpu's linspace and scalar_tensor
did with a uint8 scalar that index_fill_ had used. It is now tilized on host."""

import jax
import jax.numpy as jnp
import numpy as np


def test_uint8_scatter_update_then_tiled_argument():
    device = jax.devices("tt")[0]
    value = jax.device_put(np.uint8(50), device)
    target = jax.device_put(np.zeros(4, dtype=np.uint8), device)

    filled = jax.jit(lambda t, v: t.at[2].set(v))(target, value)
    np.testing.assert_array_equal(np.asarray(filled), [0, 0, 50, 0])

    plus_one = jax.jit(lambda v: v + jnp.uint8(1))(value)
    assert int(plus_one) == 51
