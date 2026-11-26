"""An FP32 matmul reads a BF16 operand that was only cast up to FP32.

A router's FP32 gate applied to BF16 hidden states casts the hidden states to
FP32 first. The cast is exact and TTNN's matmul unpacks each operand from its
own format, so tt-mlir drops the cast. TTNN must still run it at its FP32
fidelity, which it otherwise picks only when both inputs are FP32, so the
result must equal that of an FP32 matmul of the same values bit for bit, both
at O0, where tt-mlir sets the fidelity, and at O1, where TTNN picks it. With
both operands cast, the casts stay: dropping both would make the matmul a BF16
one.

At the highest precision the matmul runs on the vector unit, which reads FP32
operands only, so tt-mlir keeps every cast.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax


@pytest.mark.parametrize("precision", ["DEFAULT", "HIGHEST"])
@pytest.mark.parametrize("optimization_level", ["O0", "O1"])
@pytest.mark.parametrize("a_dtype,b_dtype", [("bfloat16", "float32"), ("float32", "bfloat16"), ("bfloat16", "bfloat16")])
@pytest.mark.parametrize("m,k,n", [(1, 2048, 128), (33, 512, 96)])
def test_matmul_reads_bf16_operand(m, k, n, a_dtype, b_dtype, optimization_level, precision, tmp_path):
    rng = np.random.default_rng(m + k + n)
    # FP32 values with their full mantissa, so a lower fidelity shows.
    a = rng.standard_normal((m, k)).astype(np.float32).astype(a_dtype)
    b = (rng.standard_normal((k, n)) / np.sqrt(k)).astype(np.float32).astype(b_dtype)
    device = jax.devices("tt")[0]

    def matmul(x, w):
        return jnp.dot(x.astype(jnp.float32), w.astype(jnp.float32), precision=getattr(lax.Precision, precision))

    def run(x, w, export_path):
        options = {"optimization_level": optimization_level, "export_path": str(export_path)}
        return np.asarray(jax.jit(matmul, compiler_options=options)(jax.device_put(x, device), jax.device_put(w, device)))

    got = run(a, b, tmp_path / "cast")
    assert got.dtype == np.float32
    want = run(np.asarray(a, np.float32), np.asarray(b, np.float32), tmp_path / "fp32")
    np.testing.assert_array_equal(got, want)

    ir = "\n".join(path.read_text() for path in (tmp_path / "cast" / "irs").glob("ttnn_[0-9]*.mlir"))
    assert ir, "no IR was exported"
    casts = ir.count('"ttnn.typecast"')
    if precision == "HIGHEST":
        assert casts == [a_dtype, b_dtype].count("bfloat16"), "a cast of a BF16 operand of the exact matmul was dropped"
    elif "float32" in (a_dtype, b_dtype):
        assert casts == 0, "the BF16 operand was still cast"
    else:
        assert casts == 2, "a cast of the BF16 operands was dropped"


def test_shared_weight_cast_is_read_by_every_matmul(tmp_path):
    """A BF16 weight cast to FP32 once and read by several FP32 matmuls (a router a loop's passes share) is read as
    BF16 by each of them: no cast stays."""
    rng = np.random.default_rng(5)
    x1, x2 = (rng.standard_normal((4, 2048)).astype(jnp.bfloat16) for _ in range(2))
    w = rng.standard_normal((128, 2048)).astype(jnp.bfloat16)

    def logits(x1, x2, w):
        wf = w.astype(jnp.float32)
        return x1.astype(jnp.float32) @ wf.T, x2.astype(jnp.float32) @ wf.T

    run = jax.jit(logits, compiler_options={"optimization_level": "O0", "export_path": str(tmp_path)})
    device = jax.devices("tt")[0]
    out1, out2 = run(*(jax.device_put(v, device) for v in (x1, x2, w)))
    wf = w.astype(np.float64)
    for out, x in ((out1, x1), (out2, x2)):
        expected = x.astype(np.float64) @ wf.T
        np.testing.assert_allclose(np.asarray(out), expected, atol=1e-3 * np.abs(expected).max(), rtol=1e-3)
    irs = [path.read_text() for path in (tmp_path / "irs").glob("ttnn_[0-9]*.mlir")]
    assert irs and not any('"ttnn.typecast"' in ir for ir in irs)
