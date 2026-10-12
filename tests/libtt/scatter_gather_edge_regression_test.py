"""Gather with uint8 indices (cast to uint32 for ttnn.gather), scatter with an
empty update (a no-op), and float16 scatter reductions: max and min compare
the bits, add and multiply round to float16 after every update, as float16
arithmetic does."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax


def on(device, f, *args):
    return np.asarray(jax.jit(f)(*(jax.device_put(a, jax.devices(device)[0]) for a in args)))


def assert_matches_cpu(f, *args):
    np.testing.assert_array_equal(on("tt", f, *args), on("cpu", f, *args))


# Start indices above 127 catch a sign-extending widening.
def test_vmapped_dynamic_slice_with_uint8_indices():
    x = np.arange(4 * 260, dtype=np.float32).reshape(4, 260)
    starts = np.array([0, 127, 128, 255], np.uint8)
    assert_matches_cpu(jax.vmap(lambda row, start: lax.dynamic_slice(row, (start,), (2,))), x, starts)


# Gathering along a non-last axis transposes the index first.
def test_take_along_first_axis_with_uint8_indices():
    x = np.arange(6 * 40, dtype=np.float32).reshape(6, 40)
    indices = np.random.default_rng(0).integers(0, 6, size=(6, 40)).astype(np.uint8)
    assert_matches_cpu(lambda a, i: jnp.take_along_axis(a, i, axis=0), x, indices)


def test_vmapped_dynamic_update_slice_with_empty_update():
    operand = np.arange(2 * 7 * 5 * 3, dtype=np.float32).reshape(2, 7, 5, 3)
    update = np.zeros((2, 2, 0, 1), np.float32)

    def f(o, u):
        return jax.vmap(lambda a, b: lax.dynamic_update_slice(a, b, (4, 1, 0)))(o, u)

    assert_matches_cpu(f, operand, update)


def test_float16_scatter_add():
    x = np.arange(8, dtype=np.float16)
    indices = np.array([0, 2, 2, 7], np.int32)
    updates = np.array([0.5, 1.25, -2.0, 3.0], np.float16)
    assert_matches_cpu(lambda a, i, u: a.at[i].add(u), x, indices, updates)


# Each update rounds to float16: 2048 + 1 rounds back to 2048 twice, where
# accumulating in float32 and rounding once would give 2050.
def test_float16_scatter_add_rounds_every_update():
    x = np.array([2048], np.float16)
    indices = np.array([0, 0], np.int32)
    updates = np.array([1, 1], np.float16)
    got = on("tt", lambda a, i, u: a.at[i].add(u), x, indices, updates)
    np.testing.assert_array_equal(got, np.array([2048], np.float16))


# The gradient of overlapping max-pooling windows is a float16
# select_and_scatter, which accumulates several weighted updates per element.
def test_float16_select_and_scatter_add():
    x = np.random.default_rng(1).permutation(64).astype(np.float16).reshape(8, 8)
    w = (np.random.default_rng(2).integers(1, 64, size=(7, 7)) * 0.37).astype(np.float16)

    def grad(a, weights):
        def pooled(v):
            windows = lax.reduce_window(v, np.float16(-np.inf), lax.max, (2, 2), (1, 1), "VALID")
            return jnp.sum((windows * weights).astype(np.float32))

        return jax.grad(pooled)(a)

    assert_matches_cpu(grad, x, w)


# Float16 max and min compare the bits: negatives, both zeros and NaN.
@pytest.mark.parametrize("op", ["max", "min"])
def test_float16_scatter_max_min(op):
    x = np.array([-3.0, 0.0, -0.0, 2.5, 1.0, np.nan], np.float16)
    indices = np.array([0, 0, 1, 2, 3, 3, 4, 5], np.int32)
    updates = np.array([-2.0, -65504.0, -0.0, 0.0, np.nan, 7.0, -np.inf, 1.0], np.float16)
    assert_matches_cpu(lambda a, i, u: getattr(a.at[i], op)(u), x, indices, updates)
