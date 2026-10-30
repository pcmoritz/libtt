"""A float32 cumsum along a long last dimension runs in blocks: sums within each
block as a matmul with a triangular matrix of ones (FP32 sums), plus a short
cumsum over the block totals. ttnn's accumulation kernel walks the dimension a
tile at a time, which took 300 ms over a 248K vocabulary (the CDF a sampler
draws from), against 4 ms in blocks.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def device_put(x):
    return jax.device_put(x, jax.devices("tt")[0])


@pytest.mark.parametrize(
    "shape",
    [(3, 248320), (2, 4, 8192), (5, 4096), (2, 8200)],
    ids=["vocabulary", "rank3", "shortest", "no-whole-tile-blocks"],
)
@pytest.mark.parametrize("reverse", [False, True])
def test_cumsum_along_a_long_dimension(shape, reverse):
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, shape).astype(np.float32) / shape[-1]
    got = jax.jit(lambda x: jax.lax.cumsum(x, axis=x.ndim - 1, reverse=reverse))(device_put(x))
    x64 = np.flip(x, -1) if reverse else x
    expected = np.cumsum(x64.astype(np.float64), axis=-1)
    if reverse:
        expected = np.flip(expected, -1)
    np.testing.assert_allclose(np.asarray(got, np.float64), expected, rtol=0, atol=1e-6)


def test_sampling_from_a_cdf():
    # The CDF draw of sglang-jax's speculative rejection sampling.
    rng = np.random.default_rng(1)
    probs = rng.random((12, 248320)).astype(np.float32)
    probs /= probs.sum(axis=-1, keepdims=True)
    u = rng.random((12,)).astype(np.float32)

    def draw(probs, u):
        cdf = jnp.cumsum(probs, axis=-1)
        return jnp.sum((cdf <= (u * cdf[:, -1])[:, None]).astype(jnp.int32), axis=-1)

    run = jax.jit(draw)
    args = device_put(probs), device_put(u)
    got = np.asarray(run(*args))
    cdf = np.cumsum(probs.astype(np.float64), axis=-1)
    expected = (cdf <= (u * cdf[:, -1])[:, None]).sum(axis=-1)
    # A draw may move by one token where the threshold falls within rounding
    # of a CDF step.
    assert np.abs(got - expected).max() <= 1, (got, expected)

    start = time.perf_counter()
    for _ in range(3):
        out = run(*args)
    np.asarray(out)
    assert (time.perf_counter() - start) / 3 < 0.1
