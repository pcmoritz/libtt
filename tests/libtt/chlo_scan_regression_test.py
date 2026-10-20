"""Scans with a combiner run as running reductions.

PyTorch frontends emit cumulative ops as chlo.scan composites. When the scan
body only combines its two arguments, recorded as the composite attribute
"tt.scan_combiner", the scan runs as a cumulative reduction. An empty scan has
no last position, so its carry is the init value.
"""

import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
import pytest

INIT = np.array([1.0, -1.0], np.float32)


def _cumsum_scan(x, init, reverse):
    def decomposition(x, init, **_):
        return init[:, None] + lax.cumsum(x, axis=1, reverse=reverse), init + x.sum(axis=1)

    return lax.composite(decomposition, name="chlo.scan")(
        x, init, dimension=1, is_reverse=reverse, **{"tt.scan_combiner": "add"}
    )


@pytest.mark.parametrize("reverse", [False, True])
def test_cumsum(reverse):
    x = np.arange(10, dtype=np.float32).reshape(2, 5)
    out, carry = jax.jit(lambda x, init: _cumsum_scan(x, init, reverse))(x, INIT)
    sums = np.cumsum(x[:, ::-1], axis=1)[:, ::-1] if reverse else np.cumsum(x, axis=1)
    np.testing.assert_array_equal(out, INIT[:, None] + sums)
    np.testing.assert_array_equal(carry, INIT + x.sum(axis=1))


@pytest.mark.parametrize("reverse", [False, True])
def test_empty_scan(reverse):
    carry = jax.jit(lambda init: _cumsum_scan(jnp.zeros((2, 0)), init, reverse)[1])(INIT)
    np.testing.assert_array_equal(carry, INIT)
