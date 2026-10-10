"""An FP32 matmul at the highest precision gets FP32 products and sums.

The matrix unit reads FP32 operands as TF32, so a plain FP32 matmul is off by
about 1e-3 relative. With precision=HIGHEST, tt-mlir marks the matmul
(ttcore.fp32_exact) and the runtime computes it on the vector unit
(ttnn::experimental::matmul_fp32), within a few units in the last place of the
exact result. The default precision keeps the fast matrix unit.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax


def error(got, a, b):
    """The largest error relative to the sum of the products' magnitudes, which bounds FP32 rounding."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    exact = a @ b
    bound = np.abs(a) @ np.abs(b)
    return np.max(np.abs(np.asarray(got, np.float64) - exact) / np.maximum(bound, 1e-30))


def ttnn_ir(export_path):
    ir = "\n".join(path.read_text() for path in (export_path / "irs").glob("ttnn_[0-9]*.mlir"))
    assert ir, "no IR was exported"
    return ir


CASES = {
    "plain": (lambda a, b, p: jnp.dot(a, b, precision=p), (64, 96), (96, 128)),
    "transposed_rhs": (lambda a, b, p: jnp.dot(a, b.T, precision=p), (64, 96), (128, 96)),
    "batched": (lambda a, b, p: jnp.einsum("bij,jk->bik", a, b, precision=p), (3, 64, 96), (96, 128)),
    "unaligned_k": (lambda a, b, p: jnp.dot(a, b, precision=p), (40, 50), (50, 70)),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_highest_precision_is_fp32_exact(case, tmp_path):
    f, a_shape, b_shape = CASES[case]
    rng = np.random.default_rng(len(case))
    # FP32 values with their full mantissa, so TF32 rounding shows.
    a = rng.standard_normal(a_shape).astype(np.float32)
    b = rng.standard_normal(b_shape).astype(np.float32)
    device = jax.devices("tt")[0]
    k = a_shape[-1]

    def run(precision, name):
        options = {"export_path": str(tmp_path / name)}
        g = jax.jit(lambda x, w: f(x, w, precision), compiler_options=options)
        return np.asarray(g(jax.device_put(a, device), jax.device_put(b, device)))

    highest = run(lax.Precision.HIGHEST, "highest")
    assert highest.dtype == np.float32
    reference = np.einsum("...ij,jk->...ik", a, b.T if case == "transposed_rhs" else b)
    a_ref = a.reshape(-1, k) if case == "batched" else a
    b_ref = b.T if case == "transposed_rhs" else b
    exact_error = error(highest.reshape(-1, highest.shape[-1]), a_ref, b_ref)
    # FP32 multiply-adds with one rounding each: within k * 2^-24 of the bound, in practice a few 1e-7.
    assert exact_error <= k * 2.0**-24, f"highest precision is off by {exact_error:.3g}"
    assert "ttcore.fp32_exact" in ttnn_ir(tmp_path / "highest"), "the matmul was not marked exact"

    default = run(lax.Precision.DEFAULT, "default")
    default_error = error(default.reshape(-1, default.shape[-1]), a_ref, b_ref)
    assert default_error > 1e-5, f"the default precision was exact too ({default_error:.3g}); did it take the slow path?"
    assert "ttcore.fp32_exact" not in ttnn_ir(tmp_path / "default")
    np.testing.assert_allclose(default, reference, rtol=2e-2, atol=2e-2)


def test_same_operand_twice_then_two_operands():
    """x @ x followed by x @ y of the same shapes: the cached program must read y, not x again.

    Second-order gradients of dot(x, x) hit this: the forward matmul has one operand twice,
    the backward ones have two (tenstorrent/tt-metal#55605, fixed in the tt-metal libtt pins).
    """
    rng = np.random.default_rng(5)
    x = rng.standard_normal((64, 64)).astype(np.float32)
    y = rng.standard_normal((64, 64)).astype(np.float32)
    device = jax.devices("tt")[0]
    f = jax.jit(
        lambda a, b: (
            jnp.dot(a, a, precision=lax.Precision.HIGHEST),
            jnp.dot(a, b, precision=lax.Precision.HIGHEST),
            jnp.dot(b, a, precision=lax.Precision.HIGHEST),
        )
    )
    xx, xy, yx = (np.asarray(r) for r in f(jax.device_put(x, device), jax.device_put(y, device)))
    bound = 64 * 2.0**-24
    assert error(xx, x, x) <= bound
    assert error(xy, x, y) <= bound, f"x @ y after x @ x is off by {error(xy, x, y):.3g}"
    assert error(yx, y, x) <= bound, f"y @ x after x @ x is off by {error(yx, y, x):.3g}"


def test_activation_after_exact_matmul(tmp_path):
    """The exact matmul has no fused activation; the activation runs on its own."""
    rng = np.random.default_rng(3)
    a = rng.standard_normal((64, 96)).astype(np.float32)
    b = rng.standard_normal((96, 128)).astype(np.float32)
    device = jax.devices("tt")[0]
    f = jax.jit(
        lambda x, w: jax.nn.relu(jnp.dot(x, w, precision=lax.Precision.HIGHEST)),
        compiler_options={"export_path": str(tmp_path)},
    )
    got = np.asarray(f(jax.device_put(a, device), jax.device_put(b, device)))
    exact = np.maximum(np.asarray(a, np.float64) @ np.asarray(b, np.float64), 0)
    bound = np.abs(np.asarray(a, np.float64)) @ np.abs(np.asarray(b, np.float64))
    assert np.max(np.abs(got - exact) / np.maximum(bound, 1e-30)) <= 96 * 2.0**-24
    ir = ttnn_ir(tmp_path)
    assert "ttcore.fp32_exact" in ir
    assert '"ttnn.relu"' in ir, "the activation was fused into the exact matmul"
