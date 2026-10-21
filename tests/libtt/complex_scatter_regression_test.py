"""Complex scatter assignments preserve both components and untouched values."""

import jax
import numpy as np
import pytest


@pytest.mark.parametrize("level", ["O0", "O1"])
@pytest.mark.parametrize("case", ["vector", "rows", "elements", "empty"])
def test_complex_scatter_assignment(case, level):
    shape = (9,) if case == "vector" else (5, 7)
    values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    x = (values + 1j * (100 - values)).astype(np.complex64)
    indices = np.array([0, shape[0] - 1, 2], dtype=np.int32)
    if case == "empty":
        indices = indices[:0]
    update_shape = (len(indices),) if case in ("vector", "elements") else (
        len(indices), shape[1]
    )
    values = np.arange(np.prod(update_shape), dtype=np.float32).reshape(update_shape)
    updates = ((values - 3) + 1j * (17 + 2 * values)).astype(np.complex64)
    expected = x.copy()
    if case == "elements":
        expected[indices, indices] = updates
        run = lambda x, i, u: x.at[i, i].set(u)
    else:
        expected[indices] = updates
        run = lambda x, i, u: x.at[i].set(u)

    args = [jax.device_put(a, jax.devices("tt")[0]) for a in (x, indices, updates)]
    actual = jax.jit(run, compiler_options={"optimization_level": level})(*args)
    assert actual.dtype == np.complex64
    np.testing.assert_array_equal(actual, expected)
