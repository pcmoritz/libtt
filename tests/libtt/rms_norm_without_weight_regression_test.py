"""x * rsqrt(mean(x^2) + eps) with no gamma multiply after it becomes one
ttnn.rms_norm without a weight, as in Qwen3.8-Flash-Next's grouped
hyper-connection norm, whose per-element weight follows a reshape."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

EPS = 1e-6


def compile_and_run(fn, tmp_path, *args):
    run = jax.jit(fn, compiler_options={"optimization_level": "O1", "export_path": str(tmp_path)})
    device = jax.devices("tt")[0]
    out = run(*(jax.device_put(a, device) for a in args))
    (ir,) = [path.read_text() for path in (tmp_path / "irs").glob("ttnn_[0-9]*.mlir")]
    return np.asarray(out).astype(np.float32), ir


def grouped_norm(x, w, groups):
    tokens, width = x.shape
    g = x.astype(jnp.float32).reshape(tokens, groups, width // groups)
    normed = g * jax.lax.rsqrt(jnp.mean(g * g, -1, keepdims=True) + EPS)
    return (normed.reshape(tokens, width) * (1 + w)).astype(x.dtype)


@pytest.mark.parametrize("tokens", [1, 8])
@pytest.mark.parametrize("dtype", [jnp.bfloat16, np.float32])
def test_grouped_norm(tmp_path, tokens, dtype):
    rng = np.random.default_rng(0)
    x = (rng.normal(size=(tokens, 4 * 2560)) * rng.uniform(0.5, 4, size=4 * 2560)).astype(dtype)
    w = (0.1 * rng.normal(size=4 * 2560)).astype(np.float32)
    actual, ir = compile_and_run(lambda v: grouped_norm(v, jnp.asarray(w), 4), tmp_path, x)
    assert "ttnn.rms_norm" in ir
    assert "ttnn.rsqrt" not in ir
    g = x.astype(np.float64).reshape(tokens, 4, 2560)
    want = (g / np.sqrt((g * g).mean(-1, keepdims=True) + EPS)).reshape(tokens, -1) * (1 + w)
    np.testing.assert_allclose(actual, want, atol=2e-2, rtol=2e-2)


def test_weight_after_the_norm_stays_in_it(tmp_path):
    """A [D] gamma right after the product is still the rms_norm's weight."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=(8, 2560)).astype(jnp.bfloat16)
    w = (0.1 * rng.normal(size=2560)).astype(np.float32)

    def norm(v):
        v32 = v.astype(jnp.float32)
        return (v32 * jax.lax.rsqrt(jnp.mean(v32 * v32, -1, keepdims=True) + EPS) * (1 + jnp.asarray(w))).astype(v.dtype)

    actual, ir = compile_and_run(norm, tmp_path, x)
    assert "ttnn.rms_norm" in ir
    assert "ttnn.multiply" not in ir
    v = x.astype(np.float64)
    want = v / np.sqrt((v * v).mean(-1, keepdims=True) + EPS) * (1 + w)
    np.testing.assert_allclose(actual, want, atol=3e-2, rtol=2e-2)
