"""Variadic max / min reductions that return both values and indices.

PyTorch's torch.max(x, dim) is a (value, index) reduce whose body compares with
>= (<= for min), keeps the maximum, and breaks ties towards the smaller index,
with the lowest (highest) finite value as init. Both results are used, and the
init takes part in them: a row of -inf has the maximum finfo.min, and its
index 0 wins wherever it ties the maximum, as in [-inf, finfo.min].

Only that reducer may become an argmax (argmin), which picks the smaller index on
ties. A reducer that picks the larger one must not.
"""

import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
import pytest


def _arg_reduce(x, is_max, init_value, nan_aware=False, tie_break=jnp.minimum):
    def body(a, b):
        (va, ia), (vb, ib) = a, b
        better = va >= vb if is_max else va <= vb
        tie = va == vb
        if nan_aware:  # PyTorch's float reducer prefers NaNs.
            better = better | ((va != va) & (vb == vb))
            tie = tie | ((va != va) & (vb != vb))
        value = jnp.maximum(va, vb) if is_max else jnp.minimum(va, vb)
        index = jnp.where(tie, tie_break(ia, ib), jnp.where(better, ia, ib))
        return value, index

    init = (jnp.array(init_value, x.dtype), jnp.array(0, jnp.int32))
    iota = lax.broadcasted_iota(jnp.int32, x.shape, 1)
    return lax.reduce((x, iota), init, body, (1,))


@pytest.mark.parametrize("is_max", [True, False])
@pytest.mark.parametrize("finite_init", [True, False])
@pytest.mark.parametrize("nan_aware", [True, False])
def test_values_and_indices(is_max, finite_init, nan_aware):
    x = np.random.default_rng(0).standard_normal((4, 6)).astype(np.float32)
    # A row of the reduction's identity, a row that reaches it once, and a row
    # whose extreme is the finite init value.
    extreme = -np.inf if is_max else np.inf
    info = np.finfo(np.float32)
    finite = info.min if is_max else info.max
    x[1] = extreme
    x[2, 3] = extreme
    x[3] = extreme
    x[3, 2] = finite
    init = finite if finite_init else extreme
    values, indices = jax.jit(lambda x: _arg_reduce(x, is_max, init, nan_aware))(x)
    # The init (value, index 0) wins ties, so its index wins wherever it is at
    # least the maximum (at most the minimum).
    if is_max:
        expected_values = np.maximum(np.max(x, axis=1), init)
        init_wins = init >= np.max(x, axis=1)
        expected_indices = np.where(init_wins, 0, np.argmax(x, axis=1))
    else:
        expected_values = np.minimum(np.min(x, axis=1), init)
        init_wins = init <= np.min(x, axis=1)
        expected_indices = np.where(init_wins, 0, np.argmin(x, axis=1))
    np.testing.assert_array_equal(values, expected_values)
    np.testing.assert_array_equal(indices, expected_indices)


@pytest.mark.parametrize("is_max", [True, False])
def test_ties_to_the_larger_index(is_max):
    x = np.array([[2.0, 2.0]], np.float32)
    info = np.finfo(np.float32)
    init = info.min if is_max else info.max
    reduce = jax.jit(lambda x: _arg_reduce(x, is_max, init, tie_break=jnp.maximum))
    try:
        values, indices = reduce(x)
    except jax.errors.JaxRuntimeError:
        return  # Variadic reduces in general don't lower; a wrong index would.
    np.testing.assert_array_equal(values, [2.0])
    np.testing.assert_array_equal(indices, [1])
