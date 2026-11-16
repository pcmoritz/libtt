"""PyTorch's softmax: exp(x - max) times the reciprocal of its sum.

tt-mlir fuses it into TTNN's softmax in BF16 only. TTNN's softmax rejects FP16,
and in FP32 it is about 1e-3 off where the exp / sum / reciprocal decomposition
is within 1e-6, as torch_tpu's ops_test test_softmax checks. An FP32 router's
softmax still becomes a softmax over its top k
(router_topk_softmax_regression_test.py).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax


def torch_softmax(x, axis):
    # As PyTorch lowers softmax: the sum stays in x's dtype.
    e = jnp.exp(x - x.max(axis=axis, keepdims=True))
    total = lax.reduce(e, np.array(0, x.dtype), lax.add, (axis,))
    return e * (np.array(1, x.dtype) / jnp.expand_dims(total, axis))


def run(x, axis, tmp_path):
    f = jax.jit(lambda v: torch_softmax(v, axis), compiler_options={"export_path": str(tmp_path)})
    got = np.asarray(f(jax.device_put(x, jax.devices("tt")[0]))).astype(np.float64)
    x = x.astype(np.float64)
    want = np.exp(x - x.max(axis, keepdims=True))
    ir = "\n".join(path.read_text() for path in (tmp_path / "irs").glob("ttnn_[0-9]*.mlir"))
    return got, want / want.sum(axis, keepdims=True), ir


@pytest.mark.parametrize("shape,axis", [((5,), 0), ((5, 5), 1), ((5, 5), 0), ((4, 5, 5), 1), ((64, 128), 1)])
def test_float32(shape, axis, tmp_path):
    x = np.random.default_rng(0).standard_normal(shape).astype(np.float32) * 2
    got, want, _ = run(x, axis, tmp_path)
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-7)


def test_float16(tmp_path):
    x = np.random.default_rng(0).standard_normal((5, 5)).astype(np.float16) * 2
    got, want, _ = run(x, 1, tmp_path)
    np.testing.assert_allclose(got, want, rtol=5e-3, atol=1e-3)


def test_bfloat16_is_fused(tmp_path):
    x = (np.random.default_rng(0).standard_normal((5, 5)) * 2).astype(jnp.bfloat16)
    got, want, ir = run(x, 1, tmp_path)
    np.testing.assert_allclose(got, want, rtol=3e-2, atol=3e-2)
    assert '"ttnn.softmax"' in ir, "the softmax wasn't fused"
