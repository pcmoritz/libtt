"""stablehlo.broadcast_in_dim with broadcast dimensions out of order also transposes its operand.

tt-mlir lowered it to a reshape and a broadcast, which keep the operand's dims in
their order: broadcast_in_dim(w[4, 8], [3, 8, 4], dims=[2, 1]) gave w reshaped,
not transposed. PyTorch's matmul backward broadcasts a transposed weight that
way: the gradient of x.transpose(0, 1) @ w used w instead of its transpose.
"""

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest


def _tt(*arrays):
    device = jax.devices("tt")[0]
    return [jax.device_put(a, device) for a in arrays]


@pytest.mark.parametrize(
    "operand_shape,shape,dims",
    [
        ((4, 8), (3, 8, 4), (2, 1)),
        ((4, 8), (8, 5, 4), (2, 0)),
        ((4, 8), (8, 4), (1, 0)),
        ((4, 8), (4, 5, 8), (0, 2)),
        ((2, 3, 4), (4, 6, 2, 3), (2, 3, 0)),
        ((64, 96), (2, 96, 64), (2, 1)),
    ],
)
def test_broadcast_in_dim(operand_shape, shape, dims):
    w = np.arange(np.prod(operand_shape), dtype=np.float32).reshape(operand_shape)
    (tw,) = _tt(w)
    got = jax.jit(lambda w: jax.lax.broadcast_in_dim(w, shape, dims))(tw)
    want = np.broadcast_to(np.transpose(w, np.argsort(dims)).reshape(
        [shape[d] if d in dims else 1 for d in range(len(shape))]), shape)
    np.testing.assert_array_equal(np.asarray(got), want)


def test_batched_matmul_with_transposed_broadcast_weight():
    # The gradient of x.transpose(0, 1) @ w as PyTorch writes it: dy @ w^T, with w^T broadcast over the batch.
    rng = np.random.default_rng(0)
    dy = rng.standard_normal((64, 4, 128)).astype(ml_dtypes.bfloat16)
    w = (rng.standard_normal((128, 128)) * 0.1).astype(ml_dtypes.bfloat16)

    def grad_x(dy, w):
        wt = jax.lax.broadcast_in_dim(w, (64, 128, 128), (2, 1))
        out = jax.lax.dot_general(dy, wt, (((2,), (1,)), ((0,), (0,))))
        return jnp.transpose(out, (1, 0, 2))

    got = np.asarray(jax.jit(grad_x)(*_tt(dy, w))).astype(np.float32)
    want = np.transpose(np.einsum("bik,jk->bij", dy.astype(np.float32), w.astype(np.float32)), (1, 0, 2))
    assert np.abs(got - want).max() < 0.05 * np.abs(want).max()
