"""ttnn.arange supports uint8 (jnp.indices and packbits with uint8 used to throw
"Unsupported dtype")."""

import jax
import jax.numpy as jnp
import numpy as np


def run(f, *args):
    return np.asarray(jax.jit(f)(*(jax.device_put(a, jax.devices("tt")[0]) for a in args)))


def test_indices_uint8():
    with jax.default_device(jax.devices("tt")[0]):
        got = jax.jit(lambda: jnp.indices((4, 5, 6), dtype=np.uint8))()
    np.testing.assert_array_equal(np.asarray(got), np.indices((4, 5, 6), dtype=np.uint8))


def test_packbits_unpackbits():
    x = (np.random.default_rng(0).random((3, 16)) < 0.5).astype(np.uint8)
    np.testing.assert_array_equal(run(jnp.packbits, x), np.packbits(x))
    packed = np.packbits(x, axis=1)
    np.testing.assert_array_equal(run(lambda v: jnp.unpackbits(v, axis=1), packed), np.unpackbits(packed, axis=1))
