"""A boolean reduce_or is true or false, not a count.

On the device a boolean is a number, and reduce_or (jnp.any) ran as a sum:
an `any` over several true values stored 2 or more, so comparing it with true,
or with another `any`, said false. Casts and bitwise ops must stay right too.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# Every column has two or more true values, except the last, which has none.
FLAGS = np.zeros((6, 5), np.bool_)
FLAGS[:, 0] = True
FLAGS[1:4, 1] = True
FLAGS[::2, 2] = True
FLAGS[[0, 5], 3] = True


@pytest.mark.parametrize(
    "consumer",
    [
        lambda a: a == True,  # noqa: E712
        lambda a: a != True,  # noqa: E712
        lambda a: a[:, None] == a[None, :],
        lambda a: jnp.logical_xor(a, True),
        lambda a: jnp.logical_and(a, jnp.array([True, False, True, True, True])),
        lambda a: a ^ jnp.ones_like(a),
        lambda a: a.astype(jnp.int32),
        lambda a: a.astype(jnp.float32),
        lambda a: a.astype(jnp.uint8),
    ],
    ids=["eq_true", "ne_true", "pairwise_eq", "xor_true", "and", "bitwise_xor", "int32", "float32", "uint8"],
)
def test_any_feeds_consumers(consumer):
    got = jax.jit(lambda x: consumer(jnp.any(x, axis=0)))(FLAGS)
    want = consumer(jnp.any(jnp.asarray(FLAGS, device=jax.devices("cpu")[0]), axis=0))
    np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


@pytest.mark.parametrize("axis, keepdims", [(None, False), (1, True), ((0, 1), False)])
def test_any_counts_as_one(axis, keepdims):
    got = jax.jit(lambda x: jnp.any(x, axis=axis, keepdims=keepdims).astype(jnp.int32))(FLAGS)
    np.testing.assert_array_equal(np.asarray(got), np.any(FLAGS, axis=axis, keepdims=keepdims).astype(np.int32))
