"""Check the fused residual add and RMSNorm of a decode step."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax


@pytest.mark.parametrize("trace", [False, True])
@pytest.mark.parametrize(
    "rows,width,weight_dtype",
    [
        # Qwen3.8-27B and Qwen3.5-9B hidden sizes, with the model's FP32 weight.
        (1, 5120, jnp.float32),
        (1, 4096, jnp.float32),
        # Several rows, rows in the lower faces of the tile, and all rows.
        (2, 5120, jnp.float32),
        (17, 5120, jnp.float32),
        (32, 5120, jnp.float32),
        # One core, fewer cores than a row of the grid, fewer tiles than the
        # 32 cores, and a width that is not their multiple.
        (1, 32, jnp.float32),
        (1, 160, jnp.float32),
        (1, 512, jnp.float32),
        (1, 1120, jnp.float32),
        (1, 5120, jnp.bfloat16),
    ],
)
def test_rms_norm_residual(trace, rows, width, weight_dtype, tmp_path):
    eps = 1e-6

    def layer(x, r, w):
        # The residual stream pattern of a decoder layer, with the arithmetic
        # of sglang-jax's RMSNorm: statistics in FP32, the output in BF16.
        h = x + r
        hf = h.astype(jnp.float32)
        var = jnp.mean(hf * hf, axis=-1, keepdims=True)
        n = (hf * (lax.rsqrt(var + eps) * w.reshape(1, -1))).astype(jnp.bfloat16)
        return n, h

    run = jax.jit(
        layer,
        compiler_options={
            "optimization_level": "O1",
            "enable_trace": str(trace).lower(),
            "export_path": str(tmp_path),
        },
    )
    device = jax.devices("tt")[0]
    rng = np.random.default_rng(3)
    for _ in range(3):
        x = (rng.normal(0, 1, (rows, width))).astype(jnp.bfloat16)
        r = (rng.normal(0, 1, (rows, width))).astype(jnp.bfloat16)
        w = (rng.uniform(0.5, 1.5, (width,))).astype(weight_dtype)
        actual_norm, actual_sum = run(*(jax.device_put(v, device) for v in (x, r, w)))

        expected_sum = (x.astype(np.float32) + r.astype(np.float32)).astype(jnp.bfloat16)
        hf = expected_sum.astype(np.float32)
        var = np.mean(hf * hf, axis=-1, keepdims=True)
        expected_norm = hf / np.sqrt(var + eps) * w.astype(np.float32)
        # The sum is exact in FP32; packing it to BF16 rounds ties up where
        # NumPy rounds them to even, one ulp apart.
        np.testing.assert_allclose(
            np.asarray(actual_sum).astype(np.float32), expected_sum.astype(np.float32), rtol=2**-7, atol=0
        )
        np.testing.assert_allclose(
            np.asarray(actual_norm).astype(np.float32), expected_norm, atol=1e-2, rtol=1e-2
        )

    # A numerical pass alone could also come from the separate add and norm.
    assert any(
        "ttnn.rms_norm_residual" in path.read_text()
        for path in (tmp_path / "irs").glob("ttnn*.mlir")
    ), "the add and the norm were not fused"
