"""Outlined searchsorted loops must compile with the TT optimizer enabled."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize("trace", [False, True])
@pytest.mark.parametrize("side", ["left", "right"])
def test_searchsorted_optimized(trace, side):
    device = jax.devices("tt")[0]
    search = jax.jit(
        lambda boundaries, values: jnp.searchsorted(boundaries, values, side=side),
        compiler_options={"optimization_level": "1", "enable_trace": str(trace).lower()},
    )
    # Duplicate boundaries represent empty/padded requests in ragged prefill.
    # Change both arguments after warmup so replay must use current inputs.
    for offset in (0, 3, -2):
        boundaries = np.array([0, 5, 7, 7, 24], dtype=np.int32) + offset
        values = np.arange(-4, 60, dtype=np.int32) - offset
        actual = search(jax.device_put(boundaries, device), jax.device_put(values, device))
        np.testing.assert_array_equal(np.asarray(actual), np.searchsorted(boundaries, values, side))
