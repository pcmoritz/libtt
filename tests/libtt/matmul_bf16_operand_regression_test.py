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
operands only, so tt-mlir keeps the cast of a BF16 operand next to an FP32 one.
With both operands cast from BF16 it drops both casts and runs the matmul on
the matrix unit at HiFi4, so the result equals that of an FP32 matmul of the
same values at the default precision.
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

    def run(x, w, export_path, precision):
        def matmul(x, w):
            return jnp.dot(x.astype(jnp.float32), w.astype(jnp.float32), precision=getattr(lax.Precision, precision))

        options = {"optimization_level": optimization_level, "export_path": str(export_path)}
        return np.asarray(jax.jit(matmul, compiler_options=options)(jax.device_put(x, device), jax.device_put(w, device)))

    both_bf16 = a_dtype == b_dtype == "bfloat16"
    got = run(a, b, tmp_path / "cast", precision)
    assert got.dtype == np.float32
    want_precision = "DEFAULT" if both_bf16 else precision
    want = run(np.asarray(a, np.float32), np.asarray(b, np.float32), tmp_path / "fp32", want_precision)
    np.testing.assert_array_equal(got, want)

    ir = "\n".join(path.read_text() for path in (tmp_path / "cast" / "irs").glob("ttnn_[0-9]*.mlir"))
    assert ir, "no IR was exported"
    casts = ir.count('"ttnn.typecast"')
    if precision == "HIGHEST" and both_bf16:
        assert casts == 0, "a cast of the BF16 operands was kept"
        assert "ttcore.fp32_exact" not in ir, "the BF16 matmul still runs on the vector unit"
        assert "math_fidelity = hifi4" in ir and "fp32_dest_acc_en = true" in ir
    elif precision == "HIGHEST":
        assert casts == 1, "the cast of the BF16 operand of the exact matmul was dropped"
    elif "float32" in (a_dtype, b_dtype):
        assert casts == 0, "the BF16 operand was still cast"
    else:
        assert casts == 2, "a cast of the BF16 operands was dropped"
