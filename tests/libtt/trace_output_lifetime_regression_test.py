"""A traced output must keep its values while the next replay runs.

Every replay writes into the same output slots. A caller that reads a result
after dispatching the next call, as an overlapping scheduler does, must still
see its own values rather than those of the later replay.
"""

import jax
import numpy as np


def test_output_survives_next_replay():
    options = {"optimization_level": "O1", "enable_trace": "true"}
    f = jax.jit(lambda x: x * 2 + 1, compiler_options=options)
    device = jax.devices("tt")[0]
    inputs = [np.full((32, 64), i, dtype=np.float32) for i in range(6)]
    previous = f(jax.device_put(inputs[0], device))
    for i in range(1, len(inputs)):
        current = f(jax.device_put(inputs[i], device))
        np.testing.assert_array_equal(np.asarray(previous), inputs[i - 1] * 2 + 1)
        previous = current
    np.testing.assert_array_equal(np.asarray(previous), inputs[-1] * 2 + 1)
