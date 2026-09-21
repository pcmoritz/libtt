"""Gather payloads must be tiled before conditional selection."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize("trace", [False, True])
@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
def test_where_after_gather(trace, dtype):
    device = jax.devices("tt")[0]

    def select(mask, pool, indices):
        values = pool[indices]
        return jnp.where(mask[:, None, None], values, jnp.zeros_like(values)), values

    compiled = jax.jit(
        select,
        compiler_options={"optimization_level": "O1", "enable_trace": str(trace).lower()},
    )
    indices = np.array([4, 2, 0, 1], dtype=np.int32)
    # Changing shapes exposed an invalid row-major where payload in optimized
    # programs. Returning the gather also keeps both consumers in the graph.
    for channels in (1024, 512, 128):
        pool = (np.arange(6 * channels * 3).reshape(6, channels, 3) % 121).astype(dtype)
        for step in range(3):
            mask = (np.arange(4) + step) % 2 == 0
            selected, gathered = compiled(
                *(jax.device_put(x, device) for x in (mask, pool, indices))
            )
            np.testing.assert_array_equal(np.asarray(gathered), pool[indices])
            np.testing.assert_array_equal(
                np.asarray(selected), np.where(mask[:, None, None], pool[indices], 0)
            )
