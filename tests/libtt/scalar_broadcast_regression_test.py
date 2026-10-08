"""Binary ops broadcast a one-element operand over the other one.

TTNN's scalar-broadcast readers fill a tile from the operand's first element in
L1 before the compute kernel sees it; a wrong fill changes every element of the
result. The scalar is on either side, changes from tile to tile when it is one
per batch, and also feeds the ternary where.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize("dtype", [np.float32, "bfloat16"])
@pytest.mark.parametrize("op", ["divide", "multiply", "subtract"])
@pytest.mark.parametrize("shape", [(1, 8), (1, 128), (33, 70), (64, 64), (2, 3, 40)])
def test_scalar_broadcast(dtype, op, shape):
    rng = np.random.default_rng(shape[-1])
    x = rng.uniform(0.5, 2, shape).astype(np.float32).astype(dtype)
    s = rng.uniform(0.5, 2, (1,) * len(shape)).astype(np.float32).astype(dtype)
    device = jax.devices("tt")[0]
    got = np.asarray(jax.jit(getattr(jnp, op))(jax.device_put(x, device), jax.device_put(s, device)))
    want = getattr(np, op)(np.asarray(x).astype(np.float64), np.asarray(s).astype(np.float64))
    rtol = 1e-5 if dtype == np.float32 else 2e-2
    np.testing.assert_allclose(got.astype(np.float64), want, rtol=rtol, atol=1e-6)


@pytest.mark.parametrize("dtype", [np.float32, "bfloat16"])
@pytest.mark.parametrize("op", ["divide", "subtract"])
@pytest.mark.parametrize("shape", [(1, 8), (33, 70)])
def test_scalar_broadcast_on_the_left(dtype, op, shape):
    rng = np.random.default_rng(shape[-1] + 1)
    x = rng.uniform(0.5, 2, shape).astype(np.float32).astype(dtype)
    s = rng.uniform(0.5, 2, (1,) * len(shape)).astype(np.float32).astype(dtype)
    device = jax.devices("tt")[0]
    got = np.asarray(jax.jit(getattr(jnp, op))(jax.device_put(s, device), jax.device_put(x, device)))
    want = getattr(np, op)(np.asarray(s).astype(np.float64), np.asarray(x).astype(np.float64))
    rtol = 1e-5 if dtype == np.float32 else 2e-2
    np.testing.assert_allclose(got.astype(np.float64), want, rtol=rtol, atol=1e-6)


@pytest.mark.parametrize("dtype", [np.float32, "bfloat16"])
@pytest.mark.parametrize("op", ["divide", "multiply", "subtract"])
def test_scalar_per_batch(dtype, op):
    # One scalar per (2, 3) batch: the reader fills a tile with a different value for each.
    rng = np.random.default_rng(6)
    x = rng.uniform(0.5, 2, (2, 3, 33, 70)).astype(np.float32).astype(dtype)
    s = (np.arange(6, dtype=np.float32).reshape(2, 3, 1, 1) / 4 + 0.5).astype(dtype)
    device = jax.devices("tt")[0]
    got = np.asarray(jax.jit(getattr(jnp, op))(jax.device_put(x, device), jax.device_put(s, device)))
    want = getattr(np, op)(np.asarray(x).astype(np.float64), np.asarray(s).astype(np.float64))
    rtol = 1e-5 if dtype == np.float32 else 2e-2
    np.testing.assert_allclose(got.astype(np.float64), want, rtol=rtol, atol=1e-6)


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
