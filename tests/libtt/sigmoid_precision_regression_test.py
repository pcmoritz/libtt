"""Expanded sigmoid must retain promotion to float32 before its arithmetic."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize("level", ["O0", "O1"])
@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float16])
@pytest.mark.parametrize("backward", [False, True])
def test_promoted_sigmoid(level, dtype, backward, tmp_path):
    x = np.array([-6.03125, -2.125, -.890625, .375, 2.71875, 5.5], dtype=dtype)
    grad = np.array([1.2, -3.5, 2.3, -.5, 1.1, 4], np.float32)

    def compute(x, grad):
        value = x.astype(jnp.float32)
        sigmoid = 1 / (1 + jnp.exp(-value))
        return grad * sigmoid * (1 + value * (1 - sigmoid)) if backward else sigmoid

    run = jax.jit(compute, compiler_options={
        "optimization_level": level, "export_path": str(tmp_path),
    })
    device = jax.devices("tt")[0]
    actual = run(jax.device_put(x, device), jax.device_put(grad, device))
    value = x.astype(np.float64)
    sigmoid = 1 / (1 + np.exp(-value))
    expected = grad * sigmoid * (1 + value * (1 - sigmoid)) if backward else sigmoid
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-4, atol=1e-4)

    ir = "\n".join(p.read_text() for p in (tmp_path / "irs").glob("ttnn_[0-9]*.mlir"))
    sigmoid_ops = [line for line in ir.splitlines() if '"ttnn.sigmoid"' in line]
    assert sigmoid_ops, "Expected the expanded expression to remain fused"
    assert all("xf32," in line.split("->")[0] for line in sigmoid_ops)
