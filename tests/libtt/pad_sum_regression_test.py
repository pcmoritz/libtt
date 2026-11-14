"""Sums of zero-padded slices run as concatenations.

The gradient of a tensor that a loop reads one slice at a time is each slice's
gradient zero-padded to the full shape, added up: PyTorch's select_backward, or
JAX's transpose of a slice. Adds of zero-paddings of adjacent regions of one
dimension become a padding of their concatenation, so a loop over every slice
sums to one concatenation.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

SHAPE = (1, 4, 8, 32, 64)


def _loop_gradient(chunks):
    # d/dx of sum_i sum(x[:, :, i] @ s) for the chunks in `chunks`.
    def loss(x, s):
        return sum(jnp.sum(x[:, :, i] @ s) for i in chunks)

    return jax.jit(jax.grad(loss))


@pytest.mark.parametrize(
    "chunks",
    [range(8), range(3, 8), [0, 1, 2, 5, 6], [7, 3, 4, 6, 5]],
    ids=["all", "suffix", "two-runs", "unordered"],
)
def test_loop_gradient(chunks):
    rng = np.random.default_rng(0)
    x = rng.standard_normal(SHAPE).astype(np.float32)
    s = rng.standard_normal((1, 4, 64, 16)).astype(np.float32)
    grad = np.asarray(_loop_gradient(list(chunks))(x, s))
    want = np.zeros(SHAPE, np.float32)
    for i in chunks:
        want[:, :, i] = s.sum(-1)[:, :, None, :]
    np.testing.assert_allclose(grad, want, rtol=1e-2, atol=1e-2)
