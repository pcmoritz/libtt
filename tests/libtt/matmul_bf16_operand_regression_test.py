"""An FP32 matmul reads a BF16 operand that was only cast up to FP32.

A router's FP32 gate applied to BF16 hidden states casts the hidden states to
FP32 first. The cast is exact and TTNN's matmul unpacks each operand from its
own format, so tt-mlir drops the cast; the result must still match an FP32
matmul. With both operands cast, the cast stays: dropping both would make the
matmul a BF16 one.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax


@pytest.mark.parametrize("cast_both", [False, True])
@pytest.mark.parametrize("m,k,n", [(1, 2048, 128), (33, 512, 96)])
def test_matmul_reads_bf16_operand(m, k, n, cast_both, tmp_path):
    rng = np.random.default_rng(m + k + n)
    a = rng.standard_normal((m, k)).astype(jnp.bfloat16)
    b = (rng.standard_normal((k, n)) / np.sqrt(k)).astype(jnp.bfloat16 if cast_both else np.float32)
    run = jax.jit(
        lambda x, w: jnp.dot(x.astype(jnp.float32), w.astype(jnp.float32), precision=lax.Precision.HIGHEST),
        compiler_options={"export_path": str(tmp_path)},
    )
    device = jax.devices("tt")[0]
    got = np.asarray(run(jax.device_put(a, device), jax.device_put(b, device)))
    want = np.asarray(a).astype(np.float64) @ np.asarray(b).astype(np.float64)
    assert got.dtype == np.float32
    np.testing.assert_allclose(got, want, rtol=1e-2, atol=1e-2)

    irs = [path.read_text() for path in (tmp_path / "irs").glob("ttnn_[0-9]*.mlir")]
    assert irs, "no IR was exported"
    ir = "\n".join(irs)
    if not cast_both:
        assert '"ttnn.typecast"' not in ir, "the BF16 operand was still cast"
