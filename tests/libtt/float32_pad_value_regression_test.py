"""ttnn's row-major pad wrote a non-zero float32 pad value as two bfloat16
halves, so padding with 1.0 gave 1.0019379 (torch_tpu's avg_pool with
ceil_mode counts its divisor by padding ones)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize("value", [1.0, -2.75, 1e-3])
@pytest.mark.parametrize(
    "padding",
    [((0, 0), (1, 1)), ((2, 0), (0, 3)), ((0, 1), (0, 2))],
    ids=["front_and_back", "front_only", "back_only"],
)
def test_pad_float32_value(padding, value):
    x = np.random.default_rng(0).standard_normal((3, 20)).astype(np.float32)
    run = jax.jit(lambda x: jnp.pad(x, padding, constant_values=value))
    got = np.asarray(run(jax.device_put(x, jax.devices("tt")[0])))
    np.testing.assert_array_equal(got, np.pad(x, padding, constant_values=np.float32(value)))
