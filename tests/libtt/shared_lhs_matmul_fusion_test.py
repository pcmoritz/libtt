"""Matmuls that share an activation fuse over their concatenated weights only
when const-eval concatenates the weights once."""

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def weight(w):
    """Marks w as a weight, prepared once rather than per call."""
    return jax.ffi.ffi_call("tt.weight_dtype_override", jax.ShapeDtypeStruct(w.shape, w.dtype))(
        w, **{"ttcore.weight_dtype": "bf16"}
    )


@pytest.mark.parametrize("marked", [False, True])
def test_shared_lhs_matmuls(marked, tmp_path):
    def projections(x, a, b):
        if marked:
            a, b = weight(a), weight(b)
        # A GDN layer's two input projections: a wide one and a narrow one.
        return x @ a.T, x @ b.T

    run = jax.jit(projections, compiler_options={"export_path": str(tmp_path)})
    device = jax.devices("tt")[0]
    rng = np.random.default_rng(3)
    x = (rng.integers(-4, 5, (1, 512)) / 16).astype(jnp.bfloat16)
    a = (rng.integers(-4, 5, (1024, 512)) / 16).astype(jnp.bfloat16)
    b = (rng.integers(-4, 5, (24, 512)) / 16).astype(jnp.bfloat16)
    actual = run(*(jax.device_put(v, device) for v in (x, a, b)))
    for out, w in zip(actual, (a, b)):
        expected = x.astype(np.float32) @ w.astype(np.float32).T
        np.testing.assert_allclose(np.asarray(out).astype(np.float32), expected, atol=0.01, rtol=0.01)

    ir = "\n".join(p.read_text() for p in (tmp_path / "irs").glob("ttnn_[0-9]*.mlir"))
    assert ir, "no IR was exported"
    # Weights the program takes as inputs would be concatenated on every call.
    concat = re.search(r'"ttnn\.concat"', ir) is not None
    assert concat == marked, "weights were concatenated" if concat else "marked weights were not fused"
