"""A scalar combined from full-size broadcasts stays a scalar.

torch_tpu's fused AdamW broadcasts its scalars to each parameter's shape before
it combines them: 1 - beta is a subtract of a full-size constant and a
full-size broadcast of beta. Folding the constant to a scalar (after the
implicit broadcast folding) left a broadcast on the subtract's output, which
became a full-size ttnn.repeat; CSE shared it between all parameters of that
shape, so it stayed allocated for the whole update: two float32 copies of
Qwen3-1.7B's 151936 x 2048 embedding, and 61 ms of repeats.
"""

import jax
import jax.numpy as jnp
import numpy as np


def test_one_minus_broadcast_scalar_is_not_materialized(tmp_path):
    shape = (512, 256)
    rng = np.random.default_rng(0)
    p, g = (rng.standard_normal(shape).astype(np.float32) for _ in range(2))

    def update(p, g, beta):
        rest = jnp.broadcast_to(jnp.float32(1), shape) - jnp.broadcast_to(beta, shape)
        return rest * g, rest * p

    run = jax.jit(update, compiler_options={"export_path": str(tmp_path)})
    device = jax.devices("tt")[0]
    beta = np.float32(0.875)
    got = run(jax.device_put(p, device), jax.device_put(g, device), beta)
    np.testing.assert_allclose(np.asarray(got[0]), (1 - beta) * g, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(got[1]), (1 - beta) * p, rtol=1e-6)
    irs = [path.read_text() for path in (tmp_path / "irs").glob("ttnn_[0-9]*.mlir")]
    assert irs and all('"ttnn.repeat"' not in ir for ir in irs)
