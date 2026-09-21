"""Conversions must preserve explicit truncation and floating-point special values."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("integer", [np.int32, np.uint32])
def test_float_integer_roundtrip(dtype, integer):
    values = np.array([0.1, 0.9, 1.1, 1.9, 2.5, 47.75], dtype=dtype)
    if integer == np.int32:
        values = np.concatenate((values, -values))
    device_values = jax.device_put(values, jax.devices("tt")[0])
    actual = jax.jit(lambda x: x.astype(integer).astype(dtype))(device_values)
    np.testing.assert_array_equal(
        np.asarray(actual), values.astype(integer).astype(dtype)
    )


@pytest.mark.parametrize("shape", [(0,), (0, 4), (8,), (2, 4), (33, 33)])
def test_bfloat16_layout_roundtrip(shape):
    values = np.array(
        [0.0, -0.0, 1.0, -2.0, np.nan, np.inf, -np.inf, 3.0], dtype=jnp.bfloat16
    )
    values = np.resize(values, shape)
    device_values = jax.device_put(values, jax.devices("tt")[0])
    actual = jax.jit(lambda x: x)(device_values)
    np.testing.assert_array_equal(
        np.asarray(actual).view(np.uint16), values.view(np.uint16)
    )
