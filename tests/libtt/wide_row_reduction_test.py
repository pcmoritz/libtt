"""Wide vocabulary reductions preserve row boundaries and padding semantics."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize("width", [32768, 32769, 151936])
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("keepdims", [False, True])
def test_wide_row_reductions(width, batch, keepdims):
    device = jax.devices("tt")[0]
    run = jax.jit(
        lambda x: (
            jnp.max(x, axis=-1, keepdims=keepdims),
            jnp.sum(x, axis=-1, keepdims=keepdims),
        ),
        compiler_options={"enable_trace": "true", "optimization_level": "O1"},
    )
    for step in range(4):
        x = -((np.arange(batch * width).reshape(batch, width) + step) % 31 + 1)
        x = x.astype(np.float32)
        maximum, total = run(jax.device_put(x, device))
        np.testing.assert_array_equal(
            np.asarray(maximum), x.max(axis=-1, keepdims=keepdims)
        )
        np.testing.assert_array_equal(
            np.asarray(total), x.sum(axis=-1, keepdims=keepdims)
        )


def test_wide_row_log_softmax():
    device = jax.devices("tt")[0]
    run = jax.jit(
        lambda x: jax.nn.log_softmax(x, axis=-1),
        compiler_options={"enable_trace": "true", "optimization_level": "O1"},
    )
    rng = np.random.default_rng(42)
    for scale in (1, 10, 0.1, 1):
        x = rng.normal(size=(1, 151936)).astype(np.float32) * scale
        shifted = x.astype(np.float64) - x.max(axis=-1, keepdims=True)
        expected = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
        np.testing.assert_allclose(
            np.asarray(run(jax.device_put(x, device))), expected,
            rtol=1e-4, atol=2e-3,
        )


@pytest.mark.parametrize("batch", [1, 2])
def test_wide_bfloat16_maximum(batch):
    device = jax.devices("tt")[0]
    run = jax.jit(
        lambda x: jnp.max(x, axis=-1),
        compiler_options={"enable_trace": "true", "optimization_level": "O1"},
    )
    for step in range(4):
        x = -((np.arange(batch * 151936).reshape(batch, 151936) + step) % 31 + 1)
        x = x.astype(jnp.bfloat16)
        np.testing.assert_array_equal(
            np.asarray(run(jax.device_put(x, device))), x.max(axis=-1)
        )


def test_wide_float32_sum_with_cancellation():
    device = jax.devices("tt")[0]
    run = jax.jit(
        lambda x: jnp.sum(x, axis=-1),
        compiler_options={"enable_trace": "true", "optimization_level": "O1"},
    )
    rng = np.random.default_rng(91)
    for _ in range(4):
        x = rng.normal(size=(2, 151936)).astype(np.float32)
        expected = x.astype(np.float64).sum(axis=-1)
        np.testing.assert_allclose(
            np.asarray(run(jax.device_put(x, device))), expected,
            rtol=2e-5, atol=1e-3,
        )
