"""Depthwise convolution reads each kernel tap at its dilated offset."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize("level", ["O0", "O1"])
@pytest.mark.parametrize("dilation", [1, 2, 3])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_depthwise_conv_dilation(level, dilation, dtype):
    x = ((np.arange(2 * 2 * 16) % 7) - 3).reshape(2, 2, 16).astype(dtype)
    w = np.array([[[1, -2, 3]], [[-1, 2, 1]]], dtype=dtype)

    def conv(x, w):
        return jax.lax.conv_general_dilated(
            x, w, (1,), "VALID", rhs_dilation=(dilation,),
            dimension_numbers=("NCW", "OIW", "NCW"), feature_group_count=2,
        )

    device = jax.devices("tt")[0]
    actual = jax.jit(conv, compiler_options={"optimization_level": level})(
        jax.device_put(x, device), jax.device_put(w, device))
    length = x.shape[-1] - 2 * dilation
    expected = sum(x[:, :, tap * dilation:tap * dilation + length].astype(np.float32)
                   * w[:, 0, tap].astype(np.float32)[None, :, None] for tap in range(3))
    np.testing.assert_array_equal(np.asarray(actual), expected.astype(dtype))
