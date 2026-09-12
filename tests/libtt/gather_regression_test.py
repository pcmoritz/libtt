"""Regression for gather offset rounding in cross-entropy graphs."""

import jax
import jax.numpy as jnp
import numpy as np


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
