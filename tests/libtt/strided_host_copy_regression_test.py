"""Non-contiguous host arrays must be copied to the device in logical order."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


VIEWS = {
    "column_slice": lambda x: x[:, 8:40],
    "column_stride": lambda x: x[:, ::2],
    "transpose": lambda x: x.T,
    "reversed": lambda x: x[::-1, ::-1],
    "inner_slice_3d": lambda x: x.reshape(4, 16, 64)[:, 2:10, 16:48],
}


@pytest.mark.parametrize("dtype", [jnp.bfloat16, np.float32])
@pytest.mark.parametrize("view", VIEWS.values(), ids=VIEWS.keys())
def test_strided_device_put(view, dtype):
    base = np.arange(64 * 64, dtype=np.float32).reshape(64, 64).astype(dtype)
    expected = view(base)
    assert not expected.flags.c_contiguous
    actual = jax.device_put(expected, jax.devices("tt")[0])
    np.testing.assert_array_equal(np.asarray(actual), expected)
