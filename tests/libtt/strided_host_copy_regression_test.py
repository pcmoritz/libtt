"""Host arrays must be copied to the device in logical order, whether strided or
contiguous, and whether the runtime supports their dtype or casts them."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


VIEWS = {
    "contiguous": lambda x: x,
    "column_slice": lambda x: x[:, 8:40],
    "column_stride": lambda x: x[:, ::2],
    "transpose": lambda x: x.T,
    "reversed": lambda x: x[::-1, ::-1],
    "inner_slice_3d": lambda x: x.reshape(4, 16, 64)[:, 2:10, 16:48],
    "singleton_axis": lambda x: x[:, 8:40, None],
}


# int8, int16 and bool go through the runtime's dtype-casting path.
@pytest.mark.parametrize("dtype", [jnp.bfloat16, np.float32, np.int8, np.int16, np.bool_])
@pytest.mark.parametrize("view", VIEWS.keys())
def test_strided_device_put(view, dtype):
    values = np.arange(64 * 64).reshape(64, 64)
    base = (values % 3 == 0) if dtype == np.bool_ else values.astype(dtype)
    expected = VIEWS[view](base)
    assert expected.flags.c_contiguous == (view == "contiguous")
    actual = jax.device_put(expected, jax.devices("tt")[0])
    np.testing.assert_array_equal(np.asarray(actual), expected)
