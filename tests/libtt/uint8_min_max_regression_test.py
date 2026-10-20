"""uint8 maximum, minimum and tensor clamp (a maximum and a minimum) run
natively: ttnn compares uint8 operands, broadcast ones included, so neither O0
nor O1 converts them to int32 first."""

import re

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
def test_uint8_min_max(name, level, tmp_path):
    op, ref, args = CASES[name]
    f = jax.jit(op, compiler_options={"optimization_level": level, "export_path": str(tmp_path)})
    got = np.asarray(f(*(jax.device_put(a, jax.devices("tt")[0]) for a in args)))
    assert got.dtype == np.uint8
    np.testing.assert_array_equal(got, ref(*args))
    ir = "\n".join(path.read_text() for path in (tmp_path / "irs").glob("ttnn_[0-9]*.mlir"))
    assert re.search(r'"ttnn\.(maximum|minimum|clamp_tensor)"\([^)]*\)[^\n]*: \(tensor<[0-9x]*xui8', ir)
