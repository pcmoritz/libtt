"""Sorts on several keys (lax.sort with num_keys > 1, jnp.lexsort) order equal
earlier keys by the later ones. Only the first key used to be sorted. Results
must match XLA on the CPU.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax


def on(device, f, *args):
    out = jax.jit(f)(*(jax.device_put(a, jax.devices(device)[0]) for a in args))
    return [np.asarray(o) for o in out] if isinstance(out, (tuple, list)) else np.asarray(out)


def assert_matches_cpu(f, *args):
    got, expected = on("tt", f, *args), on("cpu", f, *args)
    for g, e in zip(got if isinstance(got, list) else [got], expected if isinstance(expected, list) else [expected]):
        np.testing.assert_array_equal(g, e)


# Few distinct values, so most keys tie.
def keys(shape, dtype, seed=0):
    return np.random.default_rng(seed).integers(0, 3, size=shape).astype(dtype)


@pytest.mark.parametrize("dtype", [np.int32, np.float32, np.uint32, np.int16], ids=["i32", "f32", "u32", "i16"])
def test_lexsort(dtype):
    assert_matches_cpu(jnp.lexsort, keys((3, 40), dtype))


@pytest.mark.parametrize("axis", [0, -1], ids=["d0", "d1"])
def test_lexsort_2d(axis):
    x = keys((2, 6, 5), np.int32, seed=1)
    assert_matches_cpu(lambda v: jnp.lexsort(v, axis=axis), x)


# Two keys and a value operand, which follows the order of the keys.
def test_sort_two_keys_with_value():
    k0, k1 = keys((37,), np.float32, seed=2), keys((37,), np.int32, seed=3)
    values = np.arange(37, dtype=np.float32) * 0.5
    assert_matches_cpu(lambda a, b, v: lax.sort((a, b, v), num_keys=2), k0, k1, values)
