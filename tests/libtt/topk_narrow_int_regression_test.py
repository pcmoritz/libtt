"""top_k on 8- and 16-bit integers runs in float32, which holds them exactly.
It used to run in bfloat16, whose 8-bit mantissa merged large values (65534 and
65535 both became 65536), so uint16 jnp.partition picked the wrong elements.
Results must match XLA on the CPU.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax


def on(device, f, x):
    return np.asarray(jax.jit(f)(jax.device_put(x, jax.devices(device)[0])))


@pytest.mark.parametrize("dtype", [np.uint16, np.uint8, np.int16], ids=["u16", "u8", "i16"])
def test_top_k_near_max(dtype):
    info = np.iinfo(dtype)
    x = np.array([[info.max, info.max - 1, 0, info.max - 2, 1, info.max, 5, info.max - 3]], dtype)
    for k in (1, 3, 5):
        np.testing.assert_array_equal(on("tt", lambda v: lax.top_k(v, k)[0], x), on("cpu", lambda v: lax.top_k(v, k)[0], x))


# The case from JAX's lax_numpy_test (jax-ml/jax#22137).
@pytest.mark.parametrize("kth", [1, 3, 8])
def test_partition_unsigned_with_zeros(kth):
    x = np.array([[6, 65535, 0, 4, 3, 1, 0, 7, 5, 2]], np.uint16)
    got = on("tt", lambda v: jnp.partition(v, kth), x)
    expected = np.partition(x, kth)
    assert got[0, kth] == expected[0, kth]
    np.testing.assert_array_equal(np.sort(got[0, :kth]), np.sort(expected[0, :kth]))
    np.testing.assert_array_equal(np.sort(got[0, kth + 1 :]), np.sort(expected[0, kth + 1 :]))
