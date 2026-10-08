"""An FP32 jax.nn.softmax over the last dim runs as one numerically stable TTNN
softmax, and every FP32 softmax uses the exact exp.

JAX subtracts the row maximum first, as maximum(-inf, max(x)) reduced without
keeping the dim and reshaped back. tt-mlir folds that into
softmax(numericStable=true) instead of running the max reduction and the
subtraction as programs of their own, and asks for the exact exp on FP32
inputs. TTNN subtracts the row maximum of an FP32 input in FP32 and masks the
padding of a row that does not fill its last tile the same way. Logits around a
thousand overflow a softmax that does not subtract the maximum, and lose their
low bits where the subtraction runs at the matrix unit's 19 bits. BF16 softmax
(JAX sums its exponentials in FP32) and softmax over other dims keep JAX's ops.
"""

import jax
import numpy as np
import pytest


def reference(x, axis):
    x = x.astype(np.float64)
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


@pytest.mark.parametrize("dtype", [np.float32, "bfloat16"])
@pytest.mark.parametrize("offset", [0.0, 1000.0])
@pytest.mark.parametrize(
    "shape,axis",
    [
        ((1, 128), -1),  # a mixture-of-experts router's probabilities
        ((33, 70), -1),  # rows that do not fill their last tile
        ((2, 3, 40), -1),
        ((40, 6), 0),
    ],
)
def test_softmax_is_one_accurate_stable_program(dtype, offset, shape, axis, tmp_path):
    rng = np.random.default_rng(len(shape) * 100 + shape[-1])
    x = (rng.standard_normal(shape) * 4 + offset).astype(np.float32).astype(dtype)
    run = jax.jit(lambda a: jax.nn.softmax(a, axis=axis), compiler_options={"export_path": str(tmp_path)})
    got = np.asarray(run(jax.device_put(x, jax.devices("tt")[0]))).astype(np.float64)
    want = reference(np.asarray(x), axis)
    if dtype == np.float32:
        # The approximate exp alone leaves 3% here, the 19-bit subtraction 25% at offset 1000.
        np.testing.assert_allclose(got, want, rtol=1e-2, atol=1e-4)
    else:
        np.testing.assert_allclose(got, want, rtol=6e-2, atol=4e-3)

    if dtype != np.float32:
        return
    irs = [path.read_text() for path in (tmp_path / "irs").glob("ttnn_[0-9]*.mlir")]
    assert irs, "no IR was exported"
    ir = "\n".join(irs)
    assert "math_approx_mode = false" in ir, "the FP32 softmax uses the approximate exp"
    if axis == -1:
        assert "numericStable = true" in ir, "the softmax was not made numerically stable"
        assert '"ttnn.max"' not in ir and '"ttnn.subtract"' not in ir, "the max subtraction was left over"
