"""At O1 the optimizer can give a small pooling output a width-sharded L1
layout whose shards are not whole tiles, and tilizing it for the next op
failed ("Physical shard shape (30, 32) must be tile {32, 32} sized"). It is
now tilized directly by TTNN's padded tilize path."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax


@pytest.mark.parametrize(
    "shape,window,stride,pad",
    [((1, 9, 11, 32), 3, 2, 1), ((2, 8, 8, 48), 2, 2, 0), ((1, 8, 8, 32), 2, 2, 0)],
    ids=["30x32", "32x48", "16x32"],
)
def test_small_avg_pool_output_at_o1(shape, window, stride, pad):
    x = np.random.default_rng(0).standard_normal(shape).astype(jnp.bfloat16)

    def avg_pool(x):
        padding = ((0, 0), (pad, pad), (pad, pad), (0, 0))
        total = lax.reduce_window(x, jnp.zeros((), x.dtype), lax.add, (1, window, window, 1), (1, stride, stride, 1), padding)
        return total / (window * window)

    got = jax.jit(avg_pool, compiler_options={"optimization_level": "O1"})(jax.device_put(x, jax.devices("tt")[0]))
    expected = jax.jit(avg_pool, backend="cpu")(x.astype(np.float32))
    np.testing.assert_allclose(np.asarray(got, np.float32), np.asarray(expected), rtol=2e-2, atol=2e-2)
