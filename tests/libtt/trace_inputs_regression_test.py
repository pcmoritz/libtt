"""Trace replay must refresh private inputs without modifying their owners."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float16, jnp.float32, jnp.int32])
@pytest.mark.parametrize("shape", [(4, 32), (2, 3, 17)])
def test_chained_traces_receive_current_inputs(dtype, shape):
    device = jax.devices("tt")[0]
    options = {"optimization_level": "1", "enable_trace": "true"}
    producer = jax.jit(lambda x: x + 1, compiler_options=options)
    consumer = jax.jit(
        lambda x: jnp.concatenate((x, x), axis=-1), compiler_options=options
    )
    inputs = []
    for value in range(6):
        values = np.full(shape, value, np.dtype(dtype))
        x = jax.device_put(values, device)
        inputs.append((x, values))
        # The producer reuses its output buffer on replay; the consumer must
        # copy its current contents even when the wrapper version is unchanged.
        actual = consumer(producer(x))
        expected = np.concatenate((values + 1, values + 1), axis=-1)
        np.testing.assert_array_equal(np.asarray(actual), expected)
    for x, values in inputs:
        np.testing.assert_array_equal(np.asarray(x), values)
