"""Row-major reshapes must preserve data across chunks and physical rows."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize("trace", [False, True])
@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32, jnp.int32])
@pytest.mark.parametrize(
    "shape,width",
    [
        ((1, 24576), 3),
        ((4, 24576), 3),
        ((7, 512), 8),
        ((3, 255), 5),  # No aligned subdivision: use the original copy path.
        ((128, 96), 3),  # Already enough input rows to occupy every core.
        ((64, 3), 192),  # Narrow input rows combined into a wide output row.
    ],
)
def test_reshape_rows(trace, dtype, shape, width):
    device = jax.devices("tt")[0]
    run = jax.jit(
        lambda x: x.reshape(-1, width),
        compiler_options={
            "optimization_level": "1",
            "enable_trace": str(trace).lower(),
        },
    )
    rng = np.random.default_rng(35)
    for _ in range(3):
        values = rng.integers(-(2**30), 2**30, size=shape, dtype=np.int32).astype(dtype)
        result = run(jax.device_put(values, device))
        np.testing.assert_array_equal(np.asarray(result), values.reshape(-1, width))
