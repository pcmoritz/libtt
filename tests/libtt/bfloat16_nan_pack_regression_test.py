"""bfloat16 results keep NaN and -0.0: packing bfloat16 to bfloat16 takes the
packer's identity path (the rounding path turned NaN into inf and -0.0 into
+0.0, after every op). Results must match XLA on the CPU.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

VALUES = np.array([np.nan, -np.nan, np.inf, -np.inf, 0.0, -0.0, 1.5, -2.5], np.float32).astype(jnp.bfloat16)


def on(device, f, x):
    return np.asarray(jax.jit(f)(jax.device_put(x, jax.devices(device)[0])))


# Each op keeps NaN; trunc and abs also keep -0.0. Arithmetic on the FPU (x + 0, x * 1) and sign's
# zero still lose them before the pack.
@pytest.mark.parametrize(
    "f, keeps_signed_zero",
    [(jnp.trunc, True), (jnp.abs, True), (jnp.sign, False)],
    ids=["trunc", "abs", "sign"],
)
def test_bfloat16_ops_keep_nan_and_signed_zero(f, keeps_signed_zero):
    got, expected = on("tt", f, VALUES), on("cpu", f, VALUES)
    got32, expected32 = got.astype(np.float32), expected.astype(np.float32)
    np.testing.assert_array_equal(np.isnan(got32), np.isnan(expected32))
    finite = ~np.isnan(expected32)
    np.testing.assert_array_equal(got32[finite], expected32[finite])
    if keeps_signed_zero:
        np.testing.assert_array_equal(np.signbit(got32[finite]), np.signbit(expected32[finite]))
