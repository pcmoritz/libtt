"""Gather with uint8 indices (widened to 32 bits for ttnn.gather), scatter with
an empty update (a no-op), and float16 scatter reductions (computed in
float32, as tt-metal has no float16 scatter reductions)."""

import jax
import jax.numpy as jnp
import numpy as np
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
