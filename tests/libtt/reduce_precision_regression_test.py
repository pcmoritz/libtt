"""lax.reduce_precision to formats other than float16, bfloat16 and float32
(e.g. the float8 formats) uses XLA's bit manipulation on the Float32 bits:
round to nearest even, flush below the smallest normal, saturate to infinity,
keep NaN. It used to fail to legalize. Results must match XLA on the CPU.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

SPECIAL = np.array(
    [0.0, -0.0, np.inf, -np.inf, np.nan, 1.0, 1.0625, 1.125, 1.1875, 3e38, -3e38, 1e-30, 448.0, 464.0, 57344.0, 0.015625],
    np.float32,
)


def values(dtype):
    x = np.random.default_rng(0).normal(scale=4.0, size=(8, 16)).astype(np.float32)
    x.reshape(-1)[: SPECIAL.size] = SPECIAL
    return x.astype(dtype)


def run(device, x, exponent_bits, mantissa_bits):
    f = jax.jit(lambda v: lax.reduce_precision(v, exponent_bits, mantissa_bits))
    return np.asarray(f(jax.device_put(x, jax.devices(device)[0])))


@pytest.mark.parametrize("dtype", [np.float32, jnp.bfloat16, np.float16], ids=["f32", "bf16", "f16"])
@pytest.mark.parametrize("bits", [(4, 3), (5, 2), (3, 4), (6, 8), (8, 0)], ids=["e4m3", "e5m2", "e3m4", "e6m8", "e8m0"])
def test_reduce_precision_matches_xla(dtype, bits):
    x = values(dtype)
    got, expected = run("tt", x, *bits), run("cpu", x, *bits)
    np.testing.assert_array_equal(np.isnan(got.astype(np.float32)), np.isnan(expected.astype(np.float32)))
    numbers = ~np.isnan(expected.astype(np.float32))
    np.testing.assert_array_equal(got[numbers], expected[numbers])
