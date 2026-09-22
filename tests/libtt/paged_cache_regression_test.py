"""Shared paged-cache writes must preserve every row, including trace replay."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize("trace", [False, True])
@pytest.mark.parametrize("tokens", [16, 32])
def test_shared_page_updates(trace, tokens):
    device = jax.devices("tt")[0]
    dtype = jnp.bfloat16
    rng = np.random.default_rng(0)
    expected = np.asarray(rng.normal(size=(17, 8, 32, 128)), dtype=dtype)
    cache = jax.device_put(expected.copy(), device)
    pages = np.tile(np.arange(16, dtype=np.int32)[::-1], (tokens, 1))
    positions = np.arange(tokens, dtype=np.int32) + 24
    # Exercise skipped writes before, between and after valid updates.
    positions[[0, tokens // 2, -1]] = -1

    def update(cache, values, positions, pages):
        return jax.ffi.ffi_call(
            "tt.paged_update_cache",
            jax.ShapeDtypeStruct(cache.shape, cache.dtype),
            input_output_aliases={0: 0},
        )(cache, values, positions, pages, share_cache=True)

    update = jax.jit(
        update,
        donate_argnums=(0,),
        compiler_options={"optimization_level": "O1", "enable_trace": str(trace).lower()},
    )
    for offset in [0, 5, 1]:
        indices = np.where(positions < 0, -1, positions + offset).astype(np.int32)
        values = np.asarray(rng.normal(size=(1, tokens, 8, 128)), dtype=dtype)
        cache = update(cache, *[jax.device_put(x, device) for x in (values, indices, pages)])
        for row, index in enumerate(indices):
            if index >= 0:
                expected[pages[row, index // 32], :, index % 32, :] = values[0, row]
        np.testing.assert_array_equal(np.asarray(cache), expected)
