"""Check decode projections served by the streaming decode matmul.

Single-tile-row BF16 activations against BFP8 DRAM weights use a dedicated
TT-Metal program; other shapes keep the generic matmul factories. The shapes
cover even and uneven column splits, multi-group outputs, the largest resident
activation, a partial last core row, and FP32 logits.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def _bf8(weights):
    weights = weights[None]
    return jax.ffi.ffi_call(
        "tt.weight_dtype_override",
        jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        vmap_method="sequential",
    )(weights, **{"ttcore.weight_dtype": "bfp_bf8"})[0]


@pytest.mark.parametrize(
    "rows,inner_size,width,out_dtype",
    [
        # Qwen3-32B TP4 projections: QKV, O, down, LM head.
        (1, 5120, 2560, jnp.bfloat16),
        (1, 2048, 5120, jnp.bfloat16),
        (1, 6400, 5120, jnp.bfloat16),
        (1, 5120, 37984, jnp.float32),
        # TP2 down projection keeps the largest activation resident in L1.
        (1, 12800, 5120, jnp.bfloat16),
        # 89 output tiles leave a single core in the last row.
        (1, 1024, 89 * 32, jnp.bfloat16),
        (2, 5120, 5120, jnp.bfloat16),
        (32, 512, 4096, jnp.bfloat16),
        # Two tile rows use the generic implementation.
        (33, 512, 4096, jnp.bfloat16),
    ],
)
def test_decode_stream_matmul(rows, inner_size, width, out_dtype):
    def project(x, weights):
        return jnp.matmul(x, _bf8(weights), preferred_element_type=out_dtype)

    run = jax.jit(
        project,
        compiler_options={"optimization_level": "O1", "enable_trace": "true"},
    )
    device = jax.devices("tt")[0]
    rng = np.random.default_rng(11)
    weights = (rng.integers(-4, 5, (inner_size, width)) / 16).astype(jnp.bfloat16)
    weights_device = jax.device_put(weights, device)
    for _ in range(3):
        # Exactly representable BFP8 weights keep this focused on accumulation.
        x = (rng.integers(-4, 5, (rows, inner_size)) / 16).astype(jnp.bfloat16)
        actual = run(jax.device_put(x, device), weights_device)
        expected = x.astype(np.float32) @ weights.astype(np.float32)
        assert actual.dtype == out_dtype
        # Within about one BF16 ulp of the exact result.
        np.testing.assert_allclose(
            np.asarray(actual).astype(np.float32), expected, rtol=2**-7, atol=2**-5
        )
