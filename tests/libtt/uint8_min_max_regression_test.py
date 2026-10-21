"""uint8 min/max and tensor clamp preserve broadcast bounds and output dtype."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

RNG = np.random.default_rng(0)
X = RNG.integers(0, 256, (64, 96), dtype=np.uint8)
CASES = {
    "maximum": (jnp.maximum, np.maximum, [X, RNG.integers(0, 256, (64, 96), dtype=np.uint8)]),
    "minimum_row": (jnp.minimum, np.minimum, [X, RNG.integers(0, 256, (1, 96), dtype=np.uint8)]),
    "maximum_scalar": (jnp.maximum, np.maximum, [X, np.uint8(200)]),
    "clamp": (
        lambda x, lo, hi: jax.lax.clamp(lo, x, hi),
        lambda x, lo, hi: np.minimum(np.maximum(x, lo), hi),
        [X, RNG.integers(0, 128, (64, 96), dtype=np.uint8), RNG.integers(128, 256, (64, 96), dtype=np.uint8)],
    ),
    "clamp_scalar_bounds": (
        lambda x, lo, hi: jax.lax.clamp(lo, x, hi),
        lambda x, lo, hi: np.minimum(np.maximum(x, lo), hi),
        [X, np.uint8(30), np.uint8(220)],
    ),
}


@pytest.mark.parametrize("level", ["O0", "O1"])
@pytest.mark.parametrize("name", list(CASES))
def test_uint8_min_max(name, level):
    op, ref, args = CASES[name]
    f = jax.jit(op, compiler_options={"optimization_level": level})
    got = np.asarray(f(*(jax.device_put(a, jax.devices("tt")[0]) for a in args)))
    assert got.dtype == np.uint8
    np.testing.assert_array_equal(got, ref(*args))
