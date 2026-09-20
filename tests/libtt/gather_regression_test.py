"""Regression tests for gather indexing and payload precision."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def test_cross_entropy_gather_after_matmul():
    rng = np.random.default_rng(0)
    inputs = rng.normal(size=(32, 32)).astype(np.float32)
    weights = rng.normal(size=(32, 256)).astype(np.float32) * 0.1
    labels = np.arange(1, 33, dtype=np.int32)

    @jax.jit
    def select_log_probs(inputs, weights, labels):
        log_probs = jax.nn.log_softmax(inputs @ weights, axis=-1)
        # Match the singleton index dimensions in TorchAX's torch.gather.
        selected = log_probs[jnp.arange(inputs.shape[0])[:, None], labels[:, None]]
        return log_probs, selected[:, 0]

    device = jax.devices("tt")[0]
    args = [jax.device_put(value, device) for value in (inputs, weights, labels)]
    log_probs, selected = map(np.asarray, select_log_probs(*args))
    # Index the actual device output to isolate gather from matmul accuracy.
    np.testing.assert_array_equal(selected, log_probs[np.arange(32), labels])


@pytest.mark.parametrize("trace", [False, True])
@pytest.mark.parametrize("shape,axis", [((5, 32, 128, 128), 0), ((2, 5, 32, 128), 1)])
def test_large_fp32_gather(trace, shape, axis):
    device = jax.devices("tt")[0]
    run = jax.jit(
        lambda x, i: jnp.take(x, i, axis=axis, mode="clip"),
        compiler_options={"optimization_level": "1", "enable_trace": str(trace).lower()},
    )
    rng = np.random.default_rng(35)
    for indices in ([3], [1], [4]):
        values = rng.uniform(-1, 1, shape).astype(np.float32)
        indices = np.array(indices, np.int32)
        result = run(jax.device_put(values, device), jax.device_put(indices, device))
        np.testing.assert_array_equal(np.asarray(result), np.take(values, indices, axis=axis))


@pytest.mark.parametrize("trace", [False, True])
@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
@pytest.mark.parametrize("shape", [(2, 16), (32, 129), (3, 5, 32)])
def test_scalar_gather(trace, dtype, shape):
    """Scalar indexing preserves tile boundaries, clipping and index shape."""
    device = jax.devices("tt")[0]

    def select(x, indices):
        values = x * x
        return values, values.at[indices].get(mode="clip")

    run = jax.jit(
        select,
        compiler_options={
            "optimization_level": "1",
            "enable_trace": str(trace).lower(),
        },
    )
    rng = np.random.default_rng(35)
    for _ in range(3):
        values = rng.normal(size=shape).astype(dtype)
        indices = tuple(
            rng.integers(-2 * size, 2 * size, size=(3, 4), dtype=np.int32)
            for size in shape
        )
        clipped = tuple(
            np.clip(np.where(index < 0, index + size, index), 0, size - 1)
            for index, size in zip(indices, shape)
        )
        actual, selected = run(
            jax.device_put(values, device),
            tuple(jax.device_put(index, device) for index in indices),
        )
        np.testing.assert_array_equal(np.asarray(selected), np.asarray(actual)[clipped])
