"""64-bit programs run with 32-bit values on device.

PyTorch frontends use float64 scalars and uint64 random keys, which reach the
compiler as f64 constants and 64-bit sort keys.
"""

import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize("level", ["O0", "O1"])
@pytest.mark.parametrize("op", ["matmul", "conv1d", "conv2d"])
def test_float64_contraction(op, level):
    """f64 contractions reach the same f32 normalization as elementwise ops."""
    with jax.enable_x64(True):
        if op == "matmul":
            x = np.arange(12, dtype=np.float64).reshape(3, 4) / 8
            w = np.arange(8, dtype=np.float64).reshape(4, 2) / 8
            run = lambda x, w: x @ w
        else:
            shape = (1, 5, 2) if op == "conv1d" else (1, 4, 5, 2)
            kernel = (2, 2, 3) if op == "conv1d" else (2, 2, 2, 3)
            dims = ("NWC", "WIO", "NWC") if op == "conv1d" else (
                "NHWC", "HWIO", "NHWC"
            )
            x = np.arange(np.prod(shape), dtype=np.float64).reshape(shape) / 8
            w = np.ones(kernel, dtype=np.float64) / 4
            run = lambda x, w: jax.lax.conv_general_dilated(
                x, w, (1,) * (len(shape) - 2), "VALID", dimension_numbers=dims
            )
        expected = jax.jit(run, device=jax.devices("cpu")[0])(x, w)
        actual = jax.jit(run, compiler_options={"optimization_level": level})(
            *[jax.device_put(a, jax.devices("tt")[0]) for a in (x, w)]
        )
        assert actual.dtype == np.float64
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def test_float64_constant():
    with jax.enable_x64(True):
        out = jax.jit(lambda x: (x.astype(jnp.float64) * 1.5 + 0.25).astype(jnp.float32))(
            jnp.arange(4, dtype=jnp.float32)
        )
    np.testing.assert_allclose(out, np.arange(4) * 1.5 + 0.25)


def test_sort_uint64_keys():
    keys = np.random.default_rng(0).integers(0, 2**32, size=(1, 100), dtype=np.uint64)
    payload = np.arange(100, dtype=np.int32)[None]
    with jax.enable_x64(True):
        sorted_keys, sorted_payload = jax.jit(
            lambda k, p: lax.sort((k, p), dimension=1, num_keys=1, is_stable=True)
        )(keys, payload)
    order = np.argsort(keys, axis=1, kind="stable")
    np.testing.assert_array_equal(sorted_keys, np.take_along_axis(keys, order, 1))
    np.testing.assert_array_equal(sorted_payload, np.take_along_axis(payload, order, 1))
