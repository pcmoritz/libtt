"""ttnn's tiled pad filled the input's implicit tile padding with the pad value on every call, a program of its own,
though that padding becomes data only where a tile dim grows from a size that is not a whole number of tiles. Back
pads of tiled tensors whose padding holds ones (from the add), with and without exposed padding."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32, jnp.int32])
@pytest.mark.parametrize("value", [0, 3])
@pytest.mark.parametrize(
    "shape,padding",
    [
        ((1, 8, 128), ((0, 0), (0, 0), (0, 128))),
        ((4, 6, 256), ((0, 0), (0, 0), (0, 256))),
        ((32, 31), ((0, 32), (0, 0))),
        ((32, 96), ((0, 32), (0, 32))),
        ((2, 8, 128), ((0, 1), (0, 0), (0, 0))),
        ((64, 31), ((0, 0), (0, 2))),
        ((8, 128), ((0, 40), (0, 0))),
        ((8, 100), ((0, 0), (0, 60))),
    ],
    ids=[
        "whole_tiles_last_dim",
        "whole_tiles_last_dim_wide",
        "whole_tiles_rows_of_partial_columns",
        "whole_tiles_both_dims",
        "upper_dim_only",
        "exposed_columns",
        "exposed_rows",
        "exposed_columns_partial_rows",
    ],
)
def test_tiled_pad(shape, padding, value, dtype):
    x = (np.arange(np.prod(shape)) % 23 - 11).reshape(shape).astype(dtype)
    run = jax.jit(lambda x: jnp.pad(x + 1, padding, constant_values=value))
    got = np.asarray(run(jax.device_put(x, jax.devices("tt")[0])))
    want = np.pad(np.asarray(x) + 1, padding, constant_values=np.asarray(value, dtype))
    np.testing.assert_array_equal(got, want)
