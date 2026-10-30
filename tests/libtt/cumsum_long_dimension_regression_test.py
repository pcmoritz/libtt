"""A forward float cumsum along the last dimension scans each row of tiles along
its width, transposing every tile so that cumsum_tile runs down its columns,
instead of permuting that dimension to the front: the permuted scan padded every
element of a thin tensor out to its own tile, and took 300 ms over a 248K
vocabulary (the CDF a sampler draws from), against 4 ms along the width.
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


def test_infinities_and_nan():
    # No compensation term: an infinity stays infinite, a NaN propagates forward only.
    x = np.ones((3, 8192), np.float32)
    x[0, 1000], x[1, 1000], x[2, 1000] = np.nan, np.inf, -np.inf
    got = np.asarray(jax.jit(lambda x: jnp.cumsum(x, axis=-1))(device_put(x)))
    expected = np.cumsum(x.astype(np.float64), axis=-1)
    np.testing.assert_array_equal(got, expected.astype(np.float32))


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
    got_cdf = np.asarray(jax.jit(lambda p: jnp.cumsum(p, axis=-1))(args[0]), np.float64)
    tolerance = np.abs(got_cdf - cdf).max()
    # A sequential float32 sum over the row drifts by a few 1e-6, about one token's probability
    # here: the draw is the token whose float64 CDF step holds the threshold, up to that drift.
    assert tolerance < 2e-5, tolerance
    threshold = u * cdf[:, -1]
    lower = np.take_along_axis(np.pad(cdf, ((0, 0), (1, 0))), got[:, None], axis=-1)[:, 0]
    upper = np.take_along_axis(cdf, got[:, None], axis=-1)[:, 0]
    assert np.all((lower <= threshold + tolerance) & (threshold <= upper + tolerance)), (got, threshold)

    start = time.perf_counter()
    for _ in range(3):
        out = run(*args)
    np.asarray(out)
    assert (time.perf_counter() - start) / 3 < 0.1
