"""A square of an absolute value doesn't compute the absolute value.

torch's vector norm squares abs(x), also through a cast to float32, so
gradient clipping took a pass over every gradient for abs alone: 56 ms of a
Qwen3-1.7B training step on one Blackhole. abs(x) * abs(x) is x * x.
"""

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest


@pytest.mark.parametrize("cast", [False, True])
def test_sum_of_squared_absolute_values(tmp_path, cast):
    rng = np.random.default_rng(0)
    x = rng.integers(-8, 9, (256, 512)).astype(ml_dtypes.bfloat16)

    def norm(x):
        a = jnp.abs(x)
        if cast:
            a = a.astype(jnp.float32)
        return jnp.sqrt(jnp.sum(a * a, dtype=jnp.float32)), a * a

    run = jax.jit(norm, compiler_options={"export_path": str(tmp_path)})
    total, squares = run(jax.device_put(x, jax.devices("tt")[0]))
    xf = x.astype(np.float32)
    np.testing.assert_array_equal(np.asarray(squares).astype(np.float32), xf * xf)
    np.testing.assert_allclose(float(total), np.sqrt(np.sum(xf * xf)), rtol=1e-6)
    irs = [path.read_text() for path in (tmp_path / "irs").glob("ttnn_[0-9]*.mlir")]
    assert irs and all('"ttnn.abs"' not in ir for ir in irs)
