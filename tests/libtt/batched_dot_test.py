"""Batched dots with little work per batch.

A batched matmul whose per-batch output cannot fill the core grid spreads whole
batches over the cores (MatmulMultiCoreReuseProgramConfig), and a batched
matrix-vector product whose per-batch output is a single tile becomes an f32
multiply and sum (XLA's dot strength reduction). Both must match the CPU.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# (einsum spec, lhs shape, rhs shape, lowering): "sum" for a multiply and sum,
# "batched" for a matmul spread over the batches.
CASES = [
    # The N-gram short conv at decode: 2560 channels of 4 taps.
    ("bck,ck->bc", (1, 2560, 4), (2560, 4), "sum"),
    ("bck,ck->bc", (8, 256, 4), (256, 4), "sum"),
    # Per-row dot products.
    ("bk,bk->b", (300, 128), (300, 128), "sum"),
    # Several contracting dims, the vector on the left, batch dims not leading.
    ("kbj,jbk->b", (3, 40, 5), (5, 40, 3), "sum"),
    # The same conv at prefill: 40 tokens per channel, more than one tile.
    ("ctk,ck->ct", (256, 40, 4), (256, 4), "batched"),
    # One query per head against its keys.
    ("hsd,hd->hs", (8, 100, 64), (8, 64), "batched"),
    # Small batched matmuls.
    ("bij,bjk->bik", (64, 8, 16), (64, 16, 8), "batched"),
]


def _ids(case):
    return f"{case[0]}-{'x'.join(map(str, case[1]))}"


def _compile(tmp_path, fn, *args):
    device = jax.devices("tt")[0]
    options = {"export_path": str(tmp_path), "export_model_name": "dot", "export_tensors": "false"}
    compiled = jax.jit(fn, compiler_options=options)
    result = np.asarray(compiled(*[jax.device_put(a, device) for a in args]))
    (ir,) = (tmp_path / "irs").glob("ttnn_runtime_dot_*.mlir")
    return result, ir.read_text()


@pytest.mark.parametrize("dtype", [jnp.bfloat16, np.float32])
@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_batched_dot_matches_cpu(tmp_path, case, dtype):
    spec, lhs_shape, rhs_shape, lowering = case
    rng = np.random.default_rng(0)
    lhs = rng.normal(size=lhs_shape).astype(dtype)
    rhs = rng.normal(size=rhs_shape).astype(dtype)

    def fn(x, y):
        return jnp.einsum(spec, x, y)

    got, ir = _compile(tmp_path, fn, lhs, rhs)
    cpu = jax.devices("cpu")[0]
    want = jax.jit(lambda x, y: jnp.einsum(spec, x, y, preferred_element_type=jnp.float32))(
        *[jax.device_put(a, cpu) for a in (lhs, rhs)]
    )
    np.testing.assert_allclose(got.astype(np.float32), np.asarray(want), rtol=2e-2, atol=5e-2)

    if lowering == "sum":
        assert '"ttnn.matmul"' not in ir, spec
        assert '"ttnn.sum"' in ir, spec
    else:
        assert "matmul_multi_core_reuse_program_config" in ir, spec


@pytest.mark.parametrize(
    "lhs_shape,rhs_shape",
    [
        # Two batches of [64, 2048] outputs fill the grid.
        ((2, 64, 512), (2, 512, 2048)),
        # Two batches of 16 output tiles: split over the batches, only two
        # cores would work.
        ((2, 128, 64), (2, 64, 128)),
    ],
)
def test_few_large_batches_keep_the_mcast_config(tmp_path, lhs_shape, rhs_shape):
    rng = np.random.default_rng(1)
    lhs = rng.normal(size=lhs_shape).astype(jnp.bfloat16)
    rhs = rng.normal(size=rhs_shape).astype(jnp.bfloat16)
    got, ir = _compile(tmp_path, jnp.matmul, lhs, rhs)
    want = np.matmul(lhs.astype(np.float32), rhs.astype(np.float32))
    np.testing.assert_allclose(got.astype(np.float32), want, rtol=2e-2, atol=0.5)
    assert '"ttnn.matmul"' in ir
    assert "matmul_multi_core_reuse_program_config" not in ir
