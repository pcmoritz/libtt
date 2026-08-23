import jax
import jax.numpy as jnp
import numpy as np


X = np.array([1.0 + 2.0j, -3.5 + 0.25j, 0.0 - 4.0j], dtype=np.complex64)
Y = np.array([-2.0 + 0.5j, 1.5 - 2.0j, 3.0 + 0.0j], dtype=np.complex64)


def assert_complex_close(actual, expected):
    np.testing.assert_allclose(
        np.asarray(actual), expected, rtol=2e-5, atol=2e-5
    )


def test_complex_add():
    result = jax.jit(lambda x, y: x + y)(X, Y)
    assert_complex_close(result, X + Y)


def test_complex_subtract():
    result = jax.jit(lambda x, y: x - y)(X, Y)
    assert_complex_close(result, X - Y)


def test_complex_multiply():
    result = jax.jit(lambda x, y: x * y)(X, Y)
    assert_complex_close(result, X * Y)


def test_complex_select():
    pred = np.array([True, False, True])
    result = jax.jit(jnp.where)(pred, X, Y)
    assert_complex_close(result, np.where(pred, X, Y))


def test_complex_iota():
    result = jnp.arange(8, dtype=jnp.complex64)
    assert_complex_close(result, np.arange(8, dtype=np.complex64))


def test_complex_negate():
    result = jax.jit(lambda x: -x)(X)
    assert_complex_close(result, -X)
