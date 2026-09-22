"""Batched argmax must retain the row-major multicore path at O1."""

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize("dtype", [jnp.bfloat16, np.float16, np.float32])
@pytest.mark.parametrize("trace", [False, True])
@pytest.mark.parametrize("shape", [(16, 151936), (33, 1056)])
def test_batched_argmax(dtype, trace, shape, tmp_path):
    values = np.random.default_rng(0).uniform(-1, 1, shape).astype(dtype)
    for row, x in enumerate(values):
        case = row % 8
        if case == 0:
            x[0], x[-1] = 9, 9.001
        elif case == 1:
            x[shape[1] // 3] = x[-1] = 10
        elif case == 2:
            x[:] = -1
            x[-1] = 0
        elif case == 3:
            x[shape[1] // 3] = x[-1] = np.inf
        elif case == 4:
            x[:] = -np.inf
        elif case == 5:
            x[:] = 0.0
            x[::2] = -0.0
        elif case == 6:
            x[shape[1] // 3], x[-1] = -np.nan, np.nan
        else:
            x[:] = np.nan

    run = jax.jit(
        lambda x: jnp.argmax(x, axis=-1),
        compiler_options={
            "optimization_level": "O1",
            "enable_trace": str(trace).lower(),
            "export_path": str(tmp_path),
            "export_model_name": "argmax",
            "export_tensors": "false",
        },
    )
    x = jax.device_put(values, jax.devices("tt")[0])
    expected = np.argmax(values.astype(np.float32), axis=-1)
    for _ in range(2):
        np.testing.assert_array_equal(np.asarray(run(x)), expected)

    # Numerical checks alone do not catch the slow tiled implementation.
    (ir,) = (tmp_path / "irs").glob("ttnn_runtime_argmax_*.mlir")
    text = ir.read_text()
    operand_layouts = re.findall(r'"ttnn.argmax".* : \(tensor<[^>]*, (#[\w]+)>\)', text)
    assert operand_layouts, "Expected a compiled argmax operation"
    for layout in operand_layouts:
        definition = next(
            line for line in text.splitlines() if line.startswith(layout + " =")
        )
        assert "!ttcore.tile" not in definition, definition
