"""Feed one traced executable's device output into another traced executable.

The consumer's trace copies the input into its slot on device. The producer's
output is itself a trace output slot that every replay overwrites in place, so
the consumer must pick up the new values on every call rather than reuse the
first one.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize("shape", [(1, 32), (1, 4748), (32, 96)])
def test_traced_output_feeds_traced_input(shape):
    options = {"optimization_level": "O1", "enable_trace": "true"}
    produce = jax.jit(lambda x, y: x * 2 + y, compiler_options=options)
    consume = jax.jit(
        lambda logits: jnp.argmax(logits, axis=-1),
        compiler_options=options,
    )
    device = jax.devices("tt")[0]
    rng = np.random.default_rng(3)
    for _ in range(6):
        x = rng.integers(-8, 8, shape).astype(jnp.bfloat16)
        y = rng.integers(-8, 8, shape).astype(jnp.bfloat16)
        logits = produce(jax.device_put(x, device), jax.device_put(y, device))
        index = consume(logits)
        expected = x.astype(np.float32) * 2 + y.astype(np.float32)
        np.testing.assert_array_equal(np.asarray(logits, dtype=np.float32), expected)
        np.testing.assert_array_equal(
            np.take_along_axis(expected, np.asarray(index)[:, None], axis=-1)[:, 0],
            expected.max(axis=-1),
        )
