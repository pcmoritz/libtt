"""Reshapes must preserve data across chunks, physical rows and tiles."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize("trace", [False, True])
@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32, jnp.int32])
@pytest.mark.parametrize(
    "shape,width",
    [
        ((1, 24576), 3),
        ((4, 24576), 3),
        ((7, 512), 8),
        ((3, 255), 5),  # No aligned subdivision: use the original copy path.
        ((128, 96), 3),  # Already enough input rows to occupy every core.
        ((64, 3), 192),  # Narrow input rows combined into a wide output row.
    ],
)
def test_reshape_rows(trace, dtype, shape, width):
    device = jax.devices("tt")[0]
    run = jax.jit(
        lambda x: x.reshape(-1, width),
        compiler_options={
            "optimization_level": "O1",
            "enable_trace": str(trace).lower(),
        },
    )
    rng = np.random.default_rng(35)
    for _ in range(3):
        values = rng.integers(-(2**30), 2**30, size=shape, dtype=np.int32).astype(dtype)
        result = run(jax.device_put(values, device))
        np.testing.assert_array_equal(np.asarray(result), values.reshape(-1, width))


@pytest.mark.parametrize("trace", [False, True])
@pytest.mark.parametrize("level", ["O0", "O1"])
@pytest.mark.parametrize("shape,width", [((1, 24576), 3), ((7, 512), 8), ((3, 255), 5), ((64, 3), 192)])
def test_uint8_reshape(trace, level, shape, width, tmp_path):
    """TTNN's device reshape copies elements by their size, so uint8 reshapes as it is.

    It cast to float32 and back at O1, and to int32 and back at O0."""
    run = jax.jit(
        lambda x: x.reshape(-1, width),
        compiler_options={
            "optimization_level": level,
            "enable_trace": str(trace).lower(),
            "export_path": str(tmp_path),
        },
    )
    values = np.random.default_rng(37).integers(0, 256, size=shape, dtype=np.uint8)
    result = np.asarray(run(jax.device_put(values, jax.devices("tt")[0])))
    assert result.dtype == np.uint8
    np.testing.assert_array_equal(result, values.reshape(-1, width))
    (ir,) = (tmp_path / "irs").glob("ttnn_runtime_*.mlir")
    assert '"ttnn.typecast"' not in ir.read_text(), "the reshape was cast"


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32, jnp.int32])
@pytest.mark.parametrize(
    "shape,new_shape",
    [
        ((4096, 1), (1, 4096)),  # A per-channel weight scale, reshaped to scale a row.
        ((1, 1536), (1536, 1)),
        ((1000, 1), (1, 1000)),
        ((3, 100, 1), (3, 1, 100)),
        ((2, 64, 1), (1, 2, 64)),  # Leading dims differ: not a transpose.
    ],
)
def test_reshape_column_to_row(dtype, shape, new_shape):
    """A tiled column reshaped to a row, or back, runs as a transpose of the last two dims.

    Small integers keep the BF16 products exact."""
    device = jax.devices("tt")[0]
    run = jax.jit(lambda x, y: (x + x).reshape(new_shape) * y)
    rng = np.random.default_rng(36)
    values = rng.integers(-8, 9, size=shape, dtype=np.int32).astype(dtype)
    scales = rng.integers(-8, 9, size=new_shape, dtype=np.int32).astype(dtype)
    result = run(jax.device_put(values, device), jax.device_put(scales, device))
    np.testing.assert_array_equal(np.asarray(result), (values + values).reshape(new_shape) * scales)
