"""uint8 transposes that move the last dimension keep every element.

At O1 the optimizer keeps uint8 permutes row-major, where tt-metal transposes
32 x 32 blocks on the compute engine. With 16-bit Dest uint8 came back as
zeros, and with 1-byte elements a block row is half of Blackhole's 64-byte DRAM
alignment, so every other row was read 32 bytes off.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize("level", ["O0", "O1"])
@pytest.mark.parametrize(
    "shape, axes",
    [
        ((3, 5), (1, 0)),
        ((32, 64), (1, 0)),
        ((64, 128), (1, 0)),
        ((65, 33), (1, 0)),
        ((2, 7, 300), (0, 2, 1)),
        ((3, 37, 50), (2, 0, 1)),
        ((4, 224, 3), (2, 0, 1)),
        ((2, 96, 130), (0, 2, 1)),
        ((2, 3, 4, 40, 72), (4, 0, 3, 1, 2)),
    ],
)
def test_uint8_transpose(shape, axes, level):
    x = np.random.default_rng(0).integers(0, 256, size=shape, dtype=np.uint8)
    f = jax.jit(lambda a: jnp.transpose(a, axes), compiler_options={"optimization_level": level})
    np.testing.assert_array_equal(f(jax.device_put(x, jax.devices("tt")[0])), np.transpose(x, axes))
