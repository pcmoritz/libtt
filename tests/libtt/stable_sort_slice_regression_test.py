"""Stable sort slices must retain tie order and support large boolean masks."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize("level", ["O0", "O1"])
@pytest.mark.parametrize("tail", [False, True])
@pytest.mark.parametrize("dtype,n,k", [(np.bool_, 31, 8), (np.float32, 31, 8),
                                      (np.float32, 257, 64), (np.float32, 257, 65),
                                      (np.bool_, 65537, 32768),
                                      (np.bool_, 131072, 65452)])
def test_stable_sort_slice(level, tail, dtype, n, k):
    keys = np.random.default_rng(5).integers(0, 2, n).astype(dtype)
    selected = slice(-k, None) if tail else slice(None, k)
    expected = np.argsort(keys, kind="stable")[selected]

    def sort_slice(x):
        values, indices = jax.lax.sort_key_val(
            x, jnp.arange(n, dtype=jnp.int32), is_stable=True)
        return values[selected], indices[selected]

    run = jax.jit(sort_slice, compiler_options={"optimization_level": level})
    values, indices = run(jax.device_put(keys, jax.devices("tt")[0]))
    np.testing.assert_array_equal(indices, expected)
    np.testing.assert_array_equal(values, keys[expected])


@pytest.mark.parametrize("level", ["O0", "O1"])
@pytest.mark.parametrize("stable", [False, True])
def test_sort_suffix_keeps_ascending_order(level, stable):
    x = np.random.default_rng(7).permutation(31).astype(np.float32)
    run = jax.jit(lambda a: jnp.sort(a, stable=stable)[-8:],
                  compiler_options={"optimization_level": level})
    got = run(jax.device_put(x, jax.devices("tt")[0]))
    np.testing.assert_array_equal(got, np.sort(x)[-8:])
