"""uint8 argmax and argmin run natively: ttnn argmax compares uint8 rows, so
neither O0 nor O1 converts them to int32 first. tt-mlir lowers argmin as the
argmax of the bitwise not, and an argmax over another axis as a permute and an
argmax over the last; ties return the first index."""

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

X = np.array([[4, 5, 6, 4, 7], [9, 9, 5, 6, 0], [1, 200, 3, 255, 2]], np.uint8)
WIDE = np.random.default_rng(0).integers(0, 256, (8, 32768), dtype=np.uint8)


@pytest.mark.parametrize("level", ["O0", "O1"])
@pytest.mark.parametrize("op", ["argmax", "argmin"])
@pytest.mark.parametrize("x, axis", [(X, None), (X, 0), (X, 1), (WIDE, -1)], ids=["all", "0", "1", "wide"])
def test_uint8_argmax(x, axis, op, level, tmp_path):
    f = jax.jit(
        lambda a: getattr(jnp, op)(a, axis=axis),
        compiler_options={"optimization_level": level, "export_path": str(tmp_path)},
    )
    got = np.asarray(f(jax.device_put(x, jax.devices("tt")[0])))
    np.testing.assert_array_equal(got, getattr(np, op)(x, axis=axis))
    ir = "\n".join(path.read_text() for path in (tmp_path / "irs").glob("ttnn_[0-9]*.mlir"))
    assert re.search(r'"ttnn\.argmax"\(%\w+\)[^\n]*: \(tensor<[0-9x]*xui8', ir)
