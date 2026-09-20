"""Row broadcasts preserve both faces and changing inputs during trace replay."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize("trace", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.int32])
@pytest.mark.parametrize("shape", [(2, 64, 65), (1, 256, 8192)])
def test_row_broadcast(trace, dtype, shape):
    device = jax.devices("tt")[0]

    def forward(x):
        row = jnp.sum(x, axis=-2, keepdims=True, dtype=x.dtype)
        return x + row, row - x, x * row

    run = jax.jit(
        forward,
        compiler_options={
            "optimization_level": "1",
            "enable_trace": str(trace).lower(),
        },
    )
    rng = np.random.default_rng(35)
    for _ in range(3):
        values = rng.integers(-32, 32, shape).astype(dtype)
        row = values.sum(axis=-2, keepdims=True, dtype=dtype)
        actual = run(jax.device_put(values, device))
        for result, expected in zip(actual, (values + row, row - values, values * row)):
            np.testing.assert_array_equal(np.asarray(result), expected)
