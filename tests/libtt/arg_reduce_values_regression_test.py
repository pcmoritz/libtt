"""Variadic max / min reductions that return both values and indices.

PyTorch's torch.max(x, dim) is a (value, index) reduce whose body compares with
>= (<= for min), keeps the maximum, and breaks ties towards the smaller index,
with the lowest (highest) finite value as init. Both results are used.
"""

import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
import pytest


def _arg_reduce(x, is_max):
    def body(a, b):
        (va, ia), (vb, ib) = a, b
        better = va >= vb if is_max else va <= vb
        value = jnp.maximum(va, vb) if is_max else jnp.minimum(va, vb)
        index = jnp.where(va == vb, jnp.minimum(ia, ib), jnp.where(better, ia, ib))
        return value, index

    info = jnp.finfo(x.dtype)
    init = (jnp.array(info.min if is_max else info.max, x.dtype), jnp.array(0, jnp.int32))
    iota = lax.broadcasted_iota(jnp.int32, x.shape, 1)
    return lax.reduce((x, iota), init, body, (1,))


@pytest.mark.parametrize("is_max", [True, False])
def test_values_and_indices(is_max):
    x = np.random.default_rng(0).standard_normal((4, 6)).astype(np.float32)
    values, indices = jax.jit(lambda x: _arg_reduce(x, is_max))(x)
    reduce, arg = (np.max, np.argmax) if is_max else (np.min, np.argmin)
    np.testing.assert_array_equal(values, reduce(x, axis=1))
    np.testing.assert_array_equal(indices, arg(x, axis=1))
