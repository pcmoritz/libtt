"""jnp.mean reaches the compiler as sum(x) / N; it becomes one mean op instead
of a sum and a divide."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def compile_and_run(fn, tmp_path, *args):
    run = jax.jit(fn, compiler_options={"optimization_level": "O1", "export_path": str(tmp_path)})
    device = jax.devices("tt")[0]
    out = run(*(jax.device_put(a, device) for a in args))
    (ir,) = [path.read_text() for path in (tmp_path / "irs").glob("ttnn_[0-9]*.mlir")]
    return np.asarray(out).astype(np.float32), ir


@pytest.mark.parametrize("axis", [-1, -2, (0, 1)])
@pytest.mark.parametrize("dtype", [jnp.bfloat16, np.float32])
def test_mean(tmp_path, axis, dtype):
    x = np.random.default_rng(0).normal(size=(2, 4, 256)).astype(dtype)
    actual, ir = compile_and_run(lambda v: jnp.mean(v, axis), tmp_path, x)
    assert "ttnn.mean" in ir
    assert "ttnn.sum" not in ir and "ttnn.divide" not in ir
    np.testing.assert_allclose(actual, x.astype(np.float64).mean(axis), atol=1e-2, rtol=1e-2)
