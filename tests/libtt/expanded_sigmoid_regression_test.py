"""An expanded sigmoid, 1 / (1 + exp(-x)), as jax.nn.sigmoid reaches the
compiler under explicit sharding, becomes one sigmoid op."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def expanded_sigmoid(x):
    return 1 / (1 + jnp.exp(-x))


def compile_and_run(fn, tmp_path, *args):
    run = jax.jit(fn, compiler_options={"optimization_level": "O1", "export_path": str(tmp_path)})
    device = jax.devices("tt")[0]
    out = run(*(jax.device_put(a, device) for a in args))
    (ir,) = [path.read_text() for path in (tmp_path / "irs").glob("ttnn_[0-9]*.mlir")]
    return np.asarray(out).astype(np.float32), ir


@pytest.mark.parametrize("dtype", [jnp.bfloat16, np.float32])
def test_gated_output(tmp_path, dtype):
    """The gated attention output of Qwen3.5: out * sigmoid(gate)."""
    rng = np.random.default_rng(0)
    out = rng.normal(0, 1, (1, 6, 256)).astype(dtype)
    gate = rng.normal(0, 2, (1, 6, 256)).astype(dtype)
    actual, ir = compile_and_run(lambda o, g: o * expanded_sigmoid(g), tmp_path, out, gate)
    assert "ttnn.sigmoid" in ir
    assert "ttnn.exp" not in ir and "ttnn.divide" not in ir
    o, g = out.astype(np.float32), gate.astype(np.float32)
    np.testing.assert_allclose(actual, o / (1 + np.exp(-g)), atol=2e-2, rtol=2e-2)


def test_silu_still_fuses(tmp_path):
    """x * sigmoid(x) stays one silu after the sigmoid is fused first."""
    x = np.random.default_rng(1).normal(0, 2, (32, 256)).astype(jnp.bfloat16)
    actual, ir = compile_and_run(lambda v: v * expanded_sigmoid(v), tmp_path, x)
    assert "ttnn.silu" in ir and "ttnn.sigmoid" not in ir
    v = x.astype(np.float32)
    np.testing.assert_allclose(actual, v / (1 + np.exp(-v)), atol=3e-2, rtol=2e-2)
