"""Alternating parameter layouts must preserve values and observe weight updates."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize("trace", [False, True])
def test_parameter_layouts(trace):
    device = jax.devices("tt")[0]
    options = {"optimization_level": "1", "enable_trace": str(trace).lower()}

    def forward(x, weight, cast):
        weight = jax.ffi.ffi_call(
            "tt.weight_dtype_override", jax.ShapeDtypeStruct(weight.shape, weight.dtype)
        )(weight, **{"ttcore.weight_dtype": "bf16"})
        if cast:
            x, weight = x.astype(jnp.float32), weight.astype(jnp.float32)
        return x * weight

    run = jax.jit(forward, static_argnums=(2,), compiler_options=options)
    update = jax.jit(
        lambda w: w + jnp.ones_like(w), donate_argnums=(0,), compiler_options=options
    )
    x = np.arange(256).reshape(2, 128).astype(jnp.bfloat16)
    weight_host = np.arange(128).astype(jnp.bfloat16)
    weight = jax.device_put(weight_host, device)
    for _ in range(3):
        for cast in (False, True, False, True):
            dtype = np.float32 if cast else jnp.bfloat16
            actual = run(jax.device_put(x, device), weight, cast)
            expected = (x.astype(np.float32) * weight_host.astype(np.float32)).astype(
                dtype
            )
            np.testing.assert_array_equal(np.asarray(actual), expected)
        weight = update(weight)
        weight_host = (weight_host.astype(np.float32) + 1).astype(jnp.bfloat16)
