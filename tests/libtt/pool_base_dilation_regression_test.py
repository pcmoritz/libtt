"""reduce_window with base dilations, as in torch_tpu's avg_pool backward,
failed to legalize (ttnn pooling has no base dilation). Sum and max windows
now interleave init values and reduce window slices."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

CASES = [
    # shape, window, strides, padding, base dilation
    ((2, 3, 5), (1, 1, 3), (1, 1, 1), ((0, 0), (0, 0), (2, 2)), (1, 1, 2)),
    ((1, 2, 4, 5), (1, 1, 2, 3), (1, 1, 1, 2), ((0, 0), (0, 0), (1, 1), (2, 2)), (1, 1, 3, 2)),
    # A 3D avg_pool gradient with stride 4, kernel 2 and padding 1.
    ((1, 1, 1, 1, 1), (1, 1, 2, 2, 2), (1, 1, 1, 1, 1), ((0, 0), (0, 0)) + ((1, 1),) * 3, (1, 1, 4, 4, 4)),
]


@pytest.mark.parametrize("dtype", [np.float32, np.float16, jnp.bfloat16])
@pytest.mark.parametrize("op", ["sum", "max"])
@pytest.mark.parametrize("case", CASES, ids=lambda case: "x".join(map(str, case[0])))
def test_base_dilated_reduce_window(case, op, dtype):
    shape, window, strides, padding, base = case
    x = np.random.default_rng(0).standard_normal(shape).astype(dtype)
    init = np.array(0 if op == "sum" else -np.inf, dtype)
    computation = lax.add if op == "sum" else lax.max

    def pool(x):
        return lax.reduce_window(x, init, computation, window, strides, padding, base_dilation=base)

    got = jax.jit(pool)(jax.device_put(x, jax.devices("tt")[0]))
    expected = jax.jit(pool, backend="cpu")(x)
    tolerance = 1e-5 if dtype == np.float32 else 2e-2
    np.testing.assert_allclose(
        np.asarray(got, np.float32), np.asarray(expected, np.float32), rtol=tolerance, atol=tolerance
    )
