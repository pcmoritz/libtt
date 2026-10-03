"""Check decode matmuls: one tile row of activations times DRAM weights."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def _bfp8(weights):
    weights = weights[None]
    return jax.ffi.ffi_call(
        "tt.weight_dtype_override",
        jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        vmap_method="sequential",
    )(weights, **{"ttcore.weight_dtype": "bfp_bf8"})[0]


@pytest.mark.parametrize(
    "rows,inner_size,width,residual,transposed,weight_dtype",
    [
        (1, 1024, 5120, False, False, "bfp_bf8"),
        # A residual add fuses into the matmul as a bias.
        (1, 1024, 5120, True, False, "bfp_bf8"),
        (4, 1024, 5120, True, False, "bfp_bf8"),
        # LM heads keep the embedding layout, [width, inner_size].
        (1, 1024, 5120, False, True, "bfp_bf8"),
        (4, 1024, 5120, False, True, "bfp_bf8"),
        # An odd K tile count and a prime output tile count.
        (1, 96, 32 * 163, True, False, "bfp_bf8"),
        (1, 1024, 5120, True, False, "bf16"),
        # A width that is not a whole number of tiles, as when a narrow
        # projection is fused into a wide one.
        (1, 1024, 5120 + 24, False, False, "bfp_bf8"),
        (1, 1024, 5120 + 24, False, True, "bfp_bf8"),
        (1, 1024, 5120 + 24, True, False, "bfp_bf8"),
    ],
)
def test_decode_matmul(rows, inner_size, width, residual, transposed, weight_dtype):
    def project(x, weights, bias):
        if weight_dtype == "bfp_bf8":
            weights = _bfp8(weights)
        out = x @ (weights.T if transposed else weights)
        return out + bias if residual else out

    run = jax.jit(project, compiler_options={"optimization_level": "O1"})
    device = jax.devices("tt")[0]
    rng = np.random.default_rng(7)
    for _ in range(3):
        # Exactly representable inputs: with FP32 accumulation, only the BF16
        # output rounds.
        x = (rng.integers(-4, 5, (rows, inner_size)) / 16).astype(jnp.bfloat16)
        shape = (width, inner_size) if transposed else (inner_size, width)
        weights = (rng.integers(-4, 5, shape) / 16).astype(jnp.bfloat16)
        bias = (rng.integers(-64, 65, (rows, width)) / 16).astype(jnp.bfloat16)
        actual = run(*(jax.device_put(v, device) for v in (x, weights, bias)))
        w = weights.astype(np.float32)
        expected = x.astype(np.float32) @ (w.T if transposed else w)
        if residual:
            expected += bias.astype(np.float32)
        np.testing.assert_allclose(
            np.asarray(actual).astype(np.float32), expected, atol=1 / 64, rtol=1 / 128
        )


def test_large_reduction_is_left_to_the_other_matmuls():
    """A reduction whose activation row alone (1.5 MiB of tiles) does not fit
    a core's L1 must not pick the decode matmul. The other matmuls round
    their partial sums to BF16 between K blocks, hence the tolerance."""

    def project(x, weights, bias):
        return x @ _bfp8(weights) + bias

    run = jax.jit(project, compiler_options={"optimization_level": "O1"})
    device = jax.devices("tt")[0]
    rng = np.random.default_rng(9)
    x = (rng.integers(-4, 5, (1, 24576)) / 16).astype(jnp.bfloat16)
    weights = (rng.integers(-4, 5, (24576, 5120)) / 16).astype(jnp.bfloat16)
    bias = (rng.integers(-64, 65, (1, 5120)) / 16).astype(jnp.bfloat16)
    actual = run(*(jax.device_put(v, device) for v in (x, weights, bias)))
    expected = x.astype(np.float32) @ weights.astype(np.float32) + bias.astype(np.float32)
    np.testing.assert_allclose(np.asarray(actual).astype(np.float32), expected, atol=0.5, rtol=0.05)
