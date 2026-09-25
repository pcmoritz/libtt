"""Check fused SwiGLU decode projections across widths and reduction sizes."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize(
    "rows,inner_size,width",
    [
        # One K tile, an odd K tile count, a prime output tile count, and a
        # partial core row.
        (1, 32, 6144),
        (1, 96, 6144),
        (1, 64, 3104),
        (1, 512, 256),
        # Several and all 32 rows of the activation tile.
        (2, 512, 12800),
        (32, 512, 6144),
        # Qwen3-8B TP1, Qwen3-32B TP4/TP2 and Qwen3-14B TP1 per-chip shapes;
        # 14B needs more output pairs per core than fit in DST at once.
        (1, 4096, 12288),
        (1, 5120, 6400),
        (1, 5120, 12800),
        (1, 5120, 17408),
        # Reductions far beyond typical hidden sizes.
        (1, 12800, 6144),
        (1, 16384, 3072),
    ],
)
def test_swiglu_projection_width(rows, inner_size, width, tmp_path):
    def swiglu(x, weights):
        weights = weights[None]
        weights = jax.ffi.ffi_call(
            "tt.weight_dtype_override",
            jax.ShapeDtypeStruct(weights.shape, weights.dtype),
            vmap_method="sequential",
        )(weights, **{"ttcore.weight_dtype": "bfp_bf8"})[0]
        up, gate = jnp.split(x @ weights, 2, axis=-1)
        return up * jax.nn.silu(gate)

    run = jax.jit(
        swiglu,
        compiler_options={
            "optimization_level": "O1",
            "enable_trace": "true",
            "export_path": str(tmp_path),
        },
    )
    device = jax.devices("tt")[0]
    rng = np.random.default_rng(6)
    for _ in range(3):
        # Exactly representable BFP8 weights keep this focused on the fusion.
        x = (rng.integers(-4, 5, (rows, inner_size)) / 16).astype(jnp.bfloat16)
        weights = (rng.integers(-4, 5, (inner_size, 2 * width)) / 16).astype(jnp.bfloat16)
        actual = run(jax.device_put(x, device), jax.device_put(weights, device))
        up, gate = np.split(x.astype(np.float32) @ weights.astype(np.float32), 2, axis=-1)
        expected = up * gate / (1 + np.exp(-gate))
        np.testing.assert_allclose(
            np.asarray(actual).astype(np.float32), expected, atol=0.01, rtol=0.04
        )

    # A numerical pass alone could also come from the unfused path.
    assert any(
        "ttnn.fused_swiglu" in path.read_text()
        for path in (tmp_path / "irs").glob("ttnn*.mlir")
    ), "projection did not select fused SwiGLU"
