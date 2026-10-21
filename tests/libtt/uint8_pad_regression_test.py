"""uint8 pads run natively: ttnn.pad takes uint8 sticks of any byte width, front
padding and whole tiles, so neither O0 nor O1 converts them to int32 (or, at O1,
float32) and back."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize("level", ["O0", "O1"])
@pytest.mark.parametrize(
    "shape, padding, value",
    [
        ((3, 5), ((1, 0), (2, 3)), 7),
        ((3, 5), ((0, 2), (0, 1)), 7),
        ((5, 7), ((1, 0), (2, 3)), 7),
        ((5, 7), ((0, 2), (0, 1)), 255),
        ((3, 1), ((0, 0), (5, 0)), 9),
        ((2, 9, 33), ((1, 1), (2, 0), (3, 30)), 200),
        ((300, 100), ((4, 5), (1, 27)), 128),
        ((64, 32), ((0, 32), (0, 64)), 0),
    ],
)
def test_uint8_pad(shape, padding, value, level, tmp_path):
    x = (np.arange(np.prod(shape)) * 37 % 256).astype(np.uint8).reshape(shape)
    f = jax.jit(
        lambda a: jnp.pad(a, padding, constant_values=np.uint8(value)),
        compiler_options={"optimization_level": level, "export_path": str(tmp_path)},
    )
    got = f(jax.device_put(x, jax.devices("tt")[0]))
    assert got.dtype == np.uint8
    np.testing.assert_array_equal(got, np.pad(x, padding, constant_values=value))
    ir = "\n".join(path.read_text() for path in (tmp_path / "irs").glob("ttnn_[0-9]*.mlir"))
    assert "ttnn.pad" in ir
    assert "ttnn.typecast" not in ir
