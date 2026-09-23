"""SwiGLU fusion must leave unsupported projection widths on the generic path."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize("width", [6144, 12288])
def test_swiglu_projection_width(width):
    def swiglu(x, weights):
        weights = weights[None]
        weights = jax.ffi.ffi_call(
            "tt.weight_dtype_override",
            jax.ShapeDtypeStruct(weights.shape, weights.dtype),
            vmap_method="sequential",
        )(weights, **{"ttcore.weight_dtype": "bfp_bf8"})[0]
        up, gate = jnp.split(x @ weights, 2, axis=-1)
        return up * jax.nn.silu(gate)

    run = jax.jit(
        swiglu,
        compiler_options={"optimization_level": "O1", "enable_trace": "true"},
    )
    device = jax.devices("tt")[0]
    rng = np.random.default_rng(6)
    for _ in range(3):
        # Exactly representable BFP8 weights keep this focused on the fusion.
        x = (rng.integers(-4, 5, (1, 64)) / 16).astype(jnp.bfloat16)
        weights = (rng.integers(-4, 5, (64, 2 * width)) / 16).astype(jnp.bfloat16)
        actual = run(jax.device_put(x, device), jax.device_put(weights, device))
        up, gate = np.split(x.astype(np.float32) @ weights.astype(np.float32), 2, axis=-1)
        expected = up * gate / (1 + np.exp(-gate))
        np.testing.assert_allclose(
            np.asarray(actual).astype(np.float32), expected, atol=0.01, rtol=0.04
        )
