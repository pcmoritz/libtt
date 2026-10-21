"""Float32 inner products keep the input precision, including batched contractions."""

import jax
import numpy as np
import pytest


@pytest.mark.parametrize("level", ["O0", "O1"])
@pytest.mark.parametrize("shape,contract,batch", [
    ((5,), (0,), ()),
    ((67,), (0,), ()),
    ((5, 7), (0,), (1,)),
    ((3, 5, 7), (2,), (0, 1)),
    ((3, 5, 7), (1, 2), (0,)),
    ((3, 5, 7), (1,), (2, 0)),
])
def test_fp32_inner_product(level, shape, contract, batch):
    rng = np.random.default_rng(7)
    x, y = [rng.normal(size=shape).astype(np.float32) for _ in range(2)]
    dims = ((contract, contract), (batch, batch))
    run = lambda x, y: jax.lax.dot_general(x, y, dims)
    result = jax.jit(run, compiler_options={"optimization_level": level})(
        *[jax.device_put(v, jax.devices("tt")[0]) for v in (x, y)])
    # Sum in float64, then arrange the remaining axes in the requested batch order.
    expected = (x.astype(np.float64) * y).sum(axis=contract)
    remaining = [axis for axis in range(len(shape)) if axis not in contract]
    expected = np.transpose(expected, [remaining.index(axis) for axis in batch])
    np.testing.assert_allclose(np.asarray(result), expected, rtol=2e-6, atol=1e-6)
