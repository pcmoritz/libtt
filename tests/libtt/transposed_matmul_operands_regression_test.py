"""dot_general passes an lhs stored as (batch, contract, result) or an rhs
stored as (batch, result, contract) to the matmul transposed, instead of
permuting it first, as in attention's backward pass. Other orders still
permute."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize(
    "spec,lhs_shape,rhs_shape",
    [
        ("km,kn->mn", (96, 64), (96, 160)),
        ("mk,nk->mn", (64, 96), (160, 96)),
        ("km,nk->mn", (96, 64), (160, 96)),
        # Attention's backward: dV = P^T dO, dK = dS^T Q, dQ = dS K.
        ("bhqk,bhqd->bhkd", (1, 4, 256, 128), (1, 4, 256, 64)),
        ("bhqk,bhkd->bhqd", (1, 4, 256, 128), (1, 4, 128, 64)),
        ("bhqd,bhkd->bhqk", (1, 4, 256, 64), (1, 4, 128, 64)),
        # Contract dims that are not trailing or leading stay permuted.
        ("kbm,bkn->bmn", (64, 2, 32), (2, 64, 48)),
    ],
)
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_transposed_operands(spec, lhs_shape, rhs_shape, dtype):
    rng = np.random.default_rng(0)
    lhs = rng.normal(size=lhs_shape).astype(dtype)
    rhs = rng.normal(size=rhs_shape).astype(dtype)
    device = jax.devices("tt")[0]
    got = jax.jit(lambda a, b: jnp.einsum(spec, a, b))(jax.device_put(lhs, device), jax.device_put(rhs, device))
    want = np.einsum(spec, lhs.astype(np.float64), rhs.astype(np.float64))
    contraction = np.prod(lhs_shape) * np.prod(rhs_shape) / want.size
    tolerance = 2e-2 * np.sqrt(np.sqrt(contraction))
    np.testing.assert_allclose(np.asarray(got, np.float64), want, atol=tolerance, rtol=2e-2)
