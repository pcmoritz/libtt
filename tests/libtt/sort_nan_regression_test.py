"""Float sorts order like stablehlo's comparators: -0.0 and +0.0 tie, and every
NaN (of either sign) sorts after +inf. NaNs used to be dropped behind the +inf
row padding or sorted first when negative. Results must match XLA on the CPU.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

SPECIAL = [np.nan, -np.nan, np.inf, -np.inf, 0.0, -0.0, 1.0, -1.0]


def on(device, f, *args):
    return np.asarray(jax.jit(f)(*(jax.device_put(a, jax.devices(device)[0]) for a in args)))


def assert_matches_cpu(f, x):
    got, expected = on("tt", f, x), on("cpu", f, x)
    np.testing.assert_array_equal(np.isnan(got.astype(np.float32)), np.isnan(expected.astype(np.float32)))
    np.testing.assert_array_equal(got, expected)


def values(n, dtype, seed=0):
    x = np.random.default_rng(seed).normal(size=n).astype(np.float32)
    special = np.array(SPECIAL, np.float32)
    x[np.random.default_rng(seed + 1).choice(n, size=min(n, 3 * len(SPECIAL)), replace=False)] = np.resize(
        special, min(n, 3 * len(SPECIAL))
    )
    return x.astype(dtype)


DTYPES = [np.float32, np.float16, jnp.bfloat16]
DTYPE_IDS = ["f32", "f16", "bf16"]


# Row lengths that need padding, including more than one tile row.
@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("n", [6, 40, 100], ids=["n6", "n40", "n100"])
@pytest.mark.parametrize("descending", [False, True], ids=["asc", "desc"])
def test_sort_special_values(dtype, n, descending):
    assert_matches_cpu(lambda v: jnp.sort(v, descending=descending), values(n, dtype))


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("descending", [False, True], ids=["asc", "desc"])
def test_stable_argsort_special_values(dtype, descending):
    x = values(40, dtype, seed=3)
    assert_matches_cpu(lambda v: jnp.argsort(v, stable=True, descending=descending), x)


# The case from JAX's lax_test: stable argsort of special values is the identity.
def test_sort_key_val_special_values_identity():
    x = np.array([-np.inf, 0.0, -0.0, np.inf, np.nan, -np.nan], np.float32)
    got = on("tt", lambda v: lax.sort_key_val(v, lax.iota(np.int32, v.size), is_stable=True)[1], x)
    np.testing.assert_array_equal(got, np.arange(x.size))


@pytest.mark.parametrize("axis", [0, 1], ids=["d0", "d1"])
def test_sort_2d_special_values(axis):
    x = values(5 * 37, np.float32, seed=5).reshape(5, 37)
    assert_matches_cpu(lambda v: jnp.sort(v, axis=axis), x)
