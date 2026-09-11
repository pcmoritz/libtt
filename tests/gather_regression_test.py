"""Device regressions for exact gather offsets, including cross-entropy."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("vocab_size", [256, 2048])
@pytest.mark.parametrize("ignore_targets", [False, True])
def test_cross_entropy_after_matmul(dtype, vocab_size, ignore_targets):
    # A preceding model matmul exposed rounding in the floating-point matmul
    # formerly used to flatten gather indices. Testing loss alone missed it.
    rng = np.random.default_rng(0)
    inputs = rng.normal(size=(32, 32)).astype(np.float32)
    weights = rng.normal(size=(32, vocab_size)).astype(np.float32) * 0.1
    targets = np.arange(1, 33, dtype=np.int32)
    if ignore_targets:
        targets[::7] = -100

    def loss_fn(weights, inputs, targets):
        log_probs = jax.nn.log_softmax((inputs @ weights).astype(jnp.float32))
        valid = targets != -100
        safe_targets = jnp.where(valid, targets, 0)
        # Match TorchAX's torch.gather lowering, including the singleton dim.
        picked = log_probs[jnp.arange(inputs.shape[0])[:, None], safe_targets[:, None]]
        losses = jnp.where(valid, -picked[:, 0], 0)
        return losses.sum() / valid.sum(), (log_probs, losses)

    evaluate = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))
    results = []
    for backend in ("cpu", "tt"):
        device = jax.devices(backend)[0]
        args = [jax.device_put(a, device) for a in (weights, inputs, targets)]
        args[:2] = [a.astype(dtype) for a in args[:2]]
        results.append(jax.tree.map(np.asarray, evaluate(*args)))

    (loss, (log_probs, losses)), gradient = results[1]
    valid = targets != -100
    expected = np.zeros(32, dtype=np.float32)
    expected[valid] = -log_probs[np.arange(32)[valid], targets[valid]]
    # Compare with the actual device log-probabilities to isolate indexing
    # correctness from ordinary matmul/softmax arithmetic differences.
    np.testing.assert_array_equal(losses, expected)
    np.testing.assert_allclose(loss, expected.sum() / valid.sum(), rtol=1e-6)
    np.testing.assert_allclose(
        gradient.astype(np.float32),
        results[0][1].astype(np.float32),
        rtol=2e-2,
        atol=2e-4,
    )


@pytest.mark.parametrize("reverse_indices", [False, True])
def test_multi_component_gather_clamps_before_flattening(reverse_indices):
    values = np.arange(32 * 64 * 8, dtype=np.float32).reshape(32, 64, 8)
    # Index non-adjacent dimensions and retain the middle dimension.
    indices = np.array([[0, 0], [8, 3], [31, 7], [-1, 4], [33, 9]], dtype=np.int32)
    index_map = (0, 2)
    if reverse_indices:
        indices = indices[:, ::-1].copy()
        index_map = (2, 0)
    dimensions = jax.lax.GatherDimensionNumbers(
        offset_dims=(1,),
        collapsed_slice_dims=(0, 2),
        start_index_map=index_map,
    )
    gather = jax.jit(
        lambda x, i: jax.lax.gather(
            x,
            i,
            dimensions,
            slice_sizes=(1, 64, 1),
            mode="clip",
        )
    )
    device = jax.devices("tt")[0]
    actual = np.asarray(
        gather(jax.device_put(values, device), jax.device_put(indices, device))
    )
    coordinates = indices[:, ::-1] if reverse_indices else indices
    expected = np.stack(
        [values[np.clip(i, 0, 31), :, np.clip(k, 0, 7)] for i, k in coordinates]
    )
    np.testing.assert_array_equal(actual, expected)


def test_multi_dimensional_gather_windows():
    # This lowering shares the index-flattening helper and expands its integer
    # offsets into a window; keep the window offsets integer as well.
    values = np.arange(16 * 32 * 8, dtype=np.float32).reshape(16, 32, 8)
    indices = np.array([[0, 0], [3, 7], [13, 27], [6, 16]], dtype=np.int32)
    dimensions = jax.lax.GatherDimensionNumbers(
        offset_dims=(0, 1, 2),
        collapsed_slice_dims=(),
        start_index_map=(0, 1),
    )
    gather = jax.jit(
        lambda x, i: jax.lax.gather(
            x,
            i,
            dimensions,
            slice_sizes=(3, 5, 8),
            mode="clip",
        )
    )
    device = jax.devices("tt")[0]
    actual = np.asarray(
        gather(jax.device_put(values, device), jax.device_put(indices, device))
    )
    expected = np.stack([values[i : i + 3, j : j + 5, :] for i, j in indices], axis=-1)
    np.testing.assert_array_equal(actual, expected)
