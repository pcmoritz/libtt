"""Binary ops broadcast a one-element operand over the other one.

TTNN's scalar-broadcast readers fill a tile from the operand's first element in
L1 before the compute kernel sees it; a wrong fill changes every element of the
result. The scalar is on either side, changes from tile to tile when it is one
per batch (and between two runs of the same program), and also feeds the
ternary where.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def check_binary(run, op, a, b, dtype):
    device = jax.devices("tt")[0]
    got = np.asarray(run(jax.device_put(a, device), jax.device_put(b, device)))
    want = getattr(np, op)(np.asarray(a).astype(np.float64), np.asarray(b).astype(np.float64))
    rtol = 1e-5 if dtype == np.float32 else 2e-2
    np.testing.assert_allclose(got.astype(np.float64), want, rtol=rtol, atol=1e-6)


def positive(rng, shape, dtype):
    return rng.uniform(0.5, 2, shape).astype(np.float32).astype(dtype)


@pytest.mark.parametrize("dtype", [np.float32, "bfloat16"])
@pytest.mark.parametrize("op", ["divide", "multiply", "subtract"])
@pytest.mark.parametrize("shape", [(1, 8), (1, 128), (33, 70), (64, 64), (2, 3, 40)])
def test_scalar_broadcast(dtype, op, shape):
    rng = np.random.default_rng(shape[-1])
    x, s = positive(rng, shape, dtype), positive(rng, (1,) * len(shape), dtype)
    check_binary(jax.jit(getattr(jnp, op)), op, x, s, dtype)


@pytest.mark.parametrize("dtype", [np.float32, "bfloat16"])
@pytest.mark.parametrize("op", ["divide", "subtract"])
@pytest.mark.parametrize("shape", [(1, 8), (33, 70)])
def test_scalar_broadcast_on_the_left(dtype, op, shape):
    rng = np.random.default_rng(shape[-1] + 1)
    x, s = positive(rng, shape, dtype), positive(rng, (1,) * len(shape), dtype)
    check_binary(jax.jit(getattr(jnp, op)), op, s, x, dtype)


@pytest.mark.parametrize("dtype", [np.float32, "bfloat16"])
@pytest.mark.parametrize("op", ["divide", "multiply", "subtract"])
@pytest.mark.parametrize("batch", [(2, 3), (8, 16)])
def test_scalar_per_batch(dtype, op, batch):
    # One distinct scalar per batch: each reader fills a tile with a different value for each of
    # its tiles (8 x 16 batches make 768 output tiles, several per core). The same compiled
    # function then runs again with the scalars reversed, so a stale fill shows.
    rng = np.random.default_rng(6)
    n = int(np.prod(batch))
    x = positive(rng, batch + (33, 70), dtype)
    s = (0.5 + np.arange(n, dtype=np.float32) / n).reshape(batch + (1, 1)).astype(dtype)
    run = jax.jit(getattr(jnp, op))
    check_binary(run, op, x, s, dtype)
    check_binary(run, op, x, s.reshape(-1)[::-1].reshape(s.shape).copy(), dtype)


@pytest.mark.parametrize("dtype", [np.float32, "bfloat16"])
def test_where_of_scalars(dtype):
    rng = np.random.default_rng(7)
    x = rng.uniform(-1, 1, (33, 70)).astype(np.float32)
    s1 = np.array([[1.5]], dtype=np.float32).astype(dtype)
    s2 = np.array([[-0.75]], dtype=np.float32).astype(dtype)
    device = jax.devices("tt")[0]
    got = np.asarray(
        jax.jit(lambda c, a, b: jnp.where(c > 0, a, b))(*(jax.device_put(v, device) for v in (x, s1, s2)))
    )
    want = np.where(x > 0, np.asarray(s1).astype(np.float64), np.asarray(s2).astype(np.float64))
    np.testing.assert_array_equal(got.astype(np.float64), want)


@pytest.mark.parametrize("dtype", [np.int32, np.uint32])
def test_integer_scalar_broadcast(dtype):
    x = np.arange(33 * 70, dtype=np.int64).reshape(33, 70).astype(dtype)
    s = np.array([[7]], dtype=dtype)
    device = jax.devices("tt")[0]
    got = np.asarray(jax.jit(jnp.add)(jax.device_put(x, device), jax.device_put(s, device)))
    np.testing.assert_array_equal(got, x + s)
