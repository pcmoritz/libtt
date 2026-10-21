"""Boolean selections retain tie order without changing floating-point TopK."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize("level", ["O0", "O1"])
@pytest.mark.parametrize("tail", [False, True])
@pytest.mark.parametrize("n,k", [(31, 8), (4096, 2048), (4096, 2049),
                                  (65537, 32768), (131072, 65452)])
def test_boolean_sort_slice(level, tail, n, k):
    keys = np.random.default_rng(5).integers(0, 2, n).astype(np.bool_)
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


@pytest.mark.parametrize("dtype,n,k", [
    (np.float32, 257, 65), (np.float32, 4096, 65), (np.float32, 4096, 128),
    (jnp.bfloat16, 257, 65), (jnp.bfloat16, 257, 128),
    (jnp.bfloat16, 4096, 65), (jnp.bfloat16, 4096, 128),
])
def test_float_topk_retains_special_value_order(dtype, n, k):
    patterns = np.array([
        0x7fc00000, 0x7fc00123, 0xffc00000, 0x7f800000, 0xff800000,
        0x7f7fffff, 0xff7fffff, 0x3f800000, 0xbf800000, 0x00000001,
        0x80000001, 0x00000000, 0x80000000,
    ], np.uint32).view(np.float32)
    x = np.resize(patterns, n).copy()
    np.random.default_rng(4).shuffle(x)
    x = x.astype(dtype)
    run = jax.jit(lambda a: jax.lax.top_k(a, k),
                  compiler_options={"optimization_level": "O0"})
    expected = jax.jit(lambda a: jax.lax.top_k(a, k))(
        jax.device_put(x, jax.devices("cpu")[0]))
    got = run(jax.device_put(x, jax.devices("tt")[0]))
    np.testing.assert_array_equal(got[1], expected[1])
    np.testing.assert_array_equal(np.asarray(got[0], np.float32),
                                  np.asarray(expected[0], np.float32))
