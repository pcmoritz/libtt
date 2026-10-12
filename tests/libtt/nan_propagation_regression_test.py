"""Elementwise maximum, minimum, clamp and power propagate NaN (backports of
tt-metal PRs #58100 and #52599). minimum used to drop a NaN operand and power
returned infinity. Results must match XLA on the CPU.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

nan, inf = np.nan, np.inf
X = np.array([nan, 1.0, -2.0, nan, 0.5, -inf, 3.0, 0.0], np.float32)
Y = np.array([1.0, nan, nan, nan, 2.0, 1.0, -1.0, 0.0], np.float32)


def on(device, f, *args):
    return np.asarray(jax.jit(f)(*(jax.device_put(a, jax.devices(device)[0]) for a in args)))


def assert_matches_cpu(f, *args, rtol=0.0):
    got, expected = (on(d, f, *args).astype(np.float32) for d in ("tt", "cpu"))
    np.testing.assert_array_equal(np.isnan(got), np.isnan(expected))
    finite = ~np.isnan(expected)
    np.testing.assert_allclose(got[finite], expected[finite], rtol=rtol)


DTYPES = [np.float32, jnp.bfloat16]


@pytest.mark.parametrize("dtype", DTYPES, ids=["f32", "bf16"])
@pytest.mark.parametrize("op", [jnp.maximum, jnp.minimum], ids=["max", "min"])
def test_max_min_propagate_nan(dtype, op):
    assert_matches_cpu(op, X.astype(dtype), Y.astype(dtype))


@pytest.mark.parametrize("dtype", DTYPES, ids=["f32", "bf16"])
def test_clip_propagates_nan(dtype):
    assert_matches_cpu(lambda v: jnp.clip(v, -1.0, 1.0), X.astype(dtype))


# NaN base or exponent gives NaN, except pow(x, 0) = 1 and pow(1, y) = 1.
@pytest.mark.parametrize("dtype", DTYPES, ids=["f32", "bf16"])
def test_power_propagates_nan(dtype):
    base = np.array([nan, nan, 2.0, 1.0, 0.5, 3.0], np.float32)
    exponent = np.array([2.0, 0.0, nan, nan, 2.0, 2.0], np.float32)
    assert_matches_cpu(jnp.power, base.astype(dtype), exponent.astype(dtype), rtol=2e-2)
