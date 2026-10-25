"""A softmax over rows wider than 1024 stays decomposed.

TTNN's softmax kernels split their work across cores by tile rows, so each row
runs on one core. With the exact exp and FP32 accumulation they use for FP32
inputs, rows wider than 1024 take longer than the exp / sum / divide
decomposition, which reduces each row across cores: 1.3 ms against 0.15 ms for
64 rows of 16384, and far longer for the two vocab-wide rows of a decode
batch's logits ([2, 151936]). The TTIR softmax fusion therefore keeps the
decomposition for rows wider than 1024 (tt_mlir_jax_stable_softmax.patch).
"""

import jax
import numpy as np
import pytest


def _jit(fn, trace, export_path):
    options = {
        "optimization_level": "O1",
        "enable_trace": str(trace).lower(),
        "export_path": str(export_path),
    }
    return jax.jit(fn, compiler_options=options)


def _put(value):
    return jax.device_put(np.asarray(value), jax.devices("tt")[0])


def _runtime_ir(export_path):
    (ir,) = (export_path / "irs").glob("ttnn_runtime_*.mlir")
    return ir.read_text()


def _softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


@pytest.mark.parametrize("trace", [False, True])
@pytest.mark.parametrize("rows", [2, 64])
def test_wide_rows_stay_decomposed(rows, trace, tmp_path):
    x = np.random.default_rng(1).standard_normal((rows, 16384), dtype=np.float32) * 3
    run = _jit(lambda v: jax.nn.softmax(v, axis=-1), trace, tmp_path)
    result = np.asarray(run(_put(x)))
    np.testing.assert_allclose(result, _softmax(x), rtol=2e-3, atol=1e-6)
    assert "ttnn.softmax" not in _runtime_ir(tmp_path)


def test_rows_of_1024_still_fuse(tmp_path):
    x = np.random.default_rng(1).standard_normal((64, 1024), dtype=np.float32) * 3
    result = np.asarray(_jit(lambda v: jax.nn.softmax(v, axis=-1), False, tmp_path)(_put(x)))
    assert "ttnn.softmax" in _runtime_ir(tmp_path)
    # TTNN's FP32 softmax defaults (exact exp, FP32 accumulation) keep the
    # fused rows normalized; the runtime's generic defaults left them 4% off.
    np.testing.assert_allclose(result.sum(axis=-1), 1.0, atol=5e-3)
    np.testing.assert_allclose(result, _softmax(x), rtol=5e-3, atol=1e-6)
