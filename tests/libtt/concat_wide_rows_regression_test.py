"""Row-major concatenations along the last dim with output rows wider than L1.

A last-dim concat whose widths are not all multiples of a tile runs row-major,
and its kernel builds each output row in L1. A row that doesn't fit together
with the widest input row is built in chunks instead. Before, such a concat
failed with "required DFB entry size ... exceeds per-core L1 capacity". It
happened, for example, for a 1-D concat of a model layer's flattened parameters,
as torch.cat or ravel_pytree does.

Chunks split input rows at offsets that aren't aligned, so every element here
has its own value: a byte read from or written to the wrong place changes the
result.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def distinct(shape, dtype, start):
    # Distinct values for float32 and int32; bfloat16 holds integers only up
    # to 256, so its values cycle through that range.
    values = start + np.arange(np.prod(shape)).reshape(shape)
    if dtype == jnp.bfloat16:
        values = values % 251 - 125
    return values.astype(dtype)


def concat_on_device(arrays, axis):
    device = jax.devices("tt")[0]
    fn = jax.jit(lambda *a: jnp.concatenate(a, axis=axis))
    return np.asarray(fn(*[jax.device_put(a, device) for a in arrays]))


def test_flattened_layer_parameters():
    # One MoE layer's parameters flattened into a 460064-element float32
    # vector (1.8 MB), as torchtitan's Qwen3 MoE test model's optimizer step
    # does. The 16-element norm weights make the concat row-major.
    widths = [49152, 16384, 16, 16, 131072, 131072, 131072, 1024, 128, 128]
    arrays, start = [], 0
    for width in widths:
        arrays.append(distinct((width,), np.float32, start))
        start += width
    np.testing.assert_array_equal(concat_on_device(arrays, 0), np.concatenate(arrays))


@pytest.mark.parametrize("dtype", [np.float32, np.int32, jnp.bfloat16])
def test_unaligned_widths_split_across_chunks(dtype):
    # Widths of a few elements put every later input at an unaligned byte
    # offset, so the chunks split the wide inputs at unaligned offsets too.
    widths = [3, 400001, 5, 7, 300003]
    arrays, start = [], 0
    for width in widths:
        arrays.append(distinct((width,), dtype, start))
        start += width
    np.testing.assert_array_equal(concat_on_device(arrays, 0), np.concatenate(arrays))


def test_rows_wider_than_l1():
    # Several rows, each built in chunks.
    shapes = [(3, 300000), (3, 17), (3, 200003)]
    arrays, start = [], 0
    for shape in shapes:
        arrays.append(distinct(shape, np.float32, start))
        start += np.prod(shape)
    np.testing.assert_array_equal(concat_on_device(arrays, 1), np.concatenate(arrays, axis=1))
