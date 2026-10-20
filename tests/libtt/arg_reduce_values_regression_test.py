"""Variadic max / min reductions that return both values and indices.

PyTorch's torch.max(x, dim) is a (value, index) reduce whose body compares with
>= (<= for min), keeps the maximum, and breaks ties towards the smaller index,
with the lowest (highest) finite value as init. Both results are used, and the
init takes part in the values: a row of -inf has the maximum finfo.min.
"""

import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
import pytest


def _arg_reduce(x, is_max, init_value):
    def body(a, b):
        (va, ia), (vb, ib) = a, b
        better = va >= vb if is_max else va <= vb
        value = jnp.maximum(va, vb) if is_max else jnp.minimum(va, vb)
        index = jnp.where(va == vb, jnp.minimum(ia, ib), jnp.where(better, ia, ib))
        return value, index

    init = (jnp.array(init_value, x.dtype), jnp.array(0, jnp.int32))
    iota = lax.broadcasted_iota(jnp.int32, x.shape, 1)
    return lax.reduce((x, iota), init, body, (1,))


@pytest.mark.parametrize("is_max", [True, False])
@pytest.mark.parametrize("finite_init", [True, False])
def test_values_and_indices(is_max, finite_init):
    x = np.random.default_rng(0).standard_normal((4, 6)).astype(np.float32)
    # A row of the reduction's identity and a row that reaches it once.
    extreme = -np.inf if is_max else np.inf
    x[1] = extreme
    x[2, 3] = extreme
    info = np.finfo(np.float32)
    if finite_init:
        init = info.min if is_max else info.max
    else:
        init = extreme
    values, indices = jax.jit(lambda x: _arg_reduce(x, is_max, init))(x)
    if is_max:
        expected_values = np.maximum(np.max(x, axis=1), init)
        expected_indices = np.argmax(x, axis=1)
    else:
        expected_values = np.minimum(np.min(x, axis=1), init)
        expected_indices = np.argmin(x, axis=1)
    np.testing.assert_array_equal(values, expected_values)
    np.testing.assert_array_equal(indices, expected_indices)
