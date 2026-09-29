"""Concatenations of slices of one tensor, which the compiler may fold into one slice."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize(
    "bounds",
    [
        # Adjacent ranges, like [q | k | v] rebuilt from a fused projection.
        [(0, 512), (512, 1024), (1024, 2560)],
        # A gap and a reversed order must stay concatenations.
        [(0, 512), (1024, 2560)],
        [(512, 1024), (0, 512)],
    ],
)
def test_concat_of_slices(bounds):
    def run(x):
        y = x @ jnp.eye(x.shape[1], dtype=x.dtype)
        return jnp.concatenate([y[:, None, a:b] for a, b in bounds], axis=2)

    x = (np.arange(2 * 4096) % 97 / 8).astype(jnp.bfloat16).reshape(2, 4096)
    out = jax.jit(run)(jax.device_put(x, jax.devices("tt")[0]))
    expected = np.concatenate([x[:, None, a:b] for a, b in bounds], axis=2)
    np.testing.assert_array_equal(np.asarray(out), expected)
