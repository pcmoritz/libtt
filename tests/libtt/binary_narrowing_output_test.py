"""An FP32 elementwise product, sum, difference or quotient cast to BF16 is one
binary op with a BF16 output: binary_ng computes in FP32 and rounds once as it
packs (exact ties may round away from zero where a typecast rounds to even).
And a sum, difference or product of BF16 tensors cast to FP32 reads the BF16
tensors themselves: binary_ng unpacks them into FP32 DST exactly."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def compile_and_run(fn, tmp_path, *args):
    run = jax.jit(fn, compiler_options={"optimization_level": "O1", "export_path": str(tmp_path)})
    device = jax.devices("tt")[0]
    out = run(*(jax.device_put(a, device) for a in args))
    (ir,) = [path.read_text() for path in (tmp_path / "irs").glob("ttnn_[0-9]*.mlir")]
    return np.asarray(out), ir


@pytest.mark.parametrize("op", ["multiply", "add", "subtract", "divide"])
def test_fp32_binary_cast_to_bf16(tmp_path, op):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4, 2560)).astype(np.float32)
    y = (rng.normal(size=(4, 2560)) + (3.0 if op == "divide" else 0.0)).astype(np.float32)
    fn = getattr(jnp, op)
    actual, ir = compile_and_run(lambda a, b: fn(a, b).astype(jnp.bfloat16), tmp_path, x, y)
    assert actual.dtype == jnp.bfloat16
    assert "ttnn.typecast" not in ir
    exact = getattr(np, op)(x.astype(np.float64), y.astype(np.float64))
    got = actual.astype(np.float64)
    half_ulp = np.ldexp(1.0, np.frexp(np.abs(got))[1] - 9)  # BF16 keeps 8 significant bits
    # FP32 rounding of the operation itself plus the BF16 rounding of the result.
    assert np.all(np.abs(got - exact) <= half_ulp * 1.001 + np.abs(exact) * 2.0**-20)


def test_bf16_residual_sum_in_fp32(tmp_path):
    """A residual update in FP32, bf16(f32(a) + f32(b) + f32(c)) (Inkling 2's), runs as two adds without typecasts."""
    rng = np.random.default_rng(1)
    a, b, c = (rng.normal(size=(4, 2304), scale=s).astype(jnp.bfloat16) for s in (8.0, 1.0, 0.25))
    f32 = lambda v: v.astype(jnp.float32)  # noqa: E731
    actual, ir = compile_and_run(
        lambda a, b, c: (f32(a) + f32(b) + f32(c)).astype(jnp.bfloat16), tmp_path, a, b, c
    )
    assert actual.dtype == jnp.bfloat16
    assert "ttnn.typecast" not in ir
    assert ir.count('"ttnn.add"') == 2
    exact = a.astype(np.float64) + b.astype(np.float64) + c.astype(np.float64)
    got = actual.astype(np.float64)
    half_ulp = np.ldexp(1.0, np.frexp(np.abs(got))[1] - 9)
    assert np.all(np.abs(got - exact) <= half_ulp * 1.001 + np.abs(exact) * 2.0**-20)


@pytest.mark.parametrize("op", ["multiply", "subtract"])
def test_bf16_operands_in_fp32(tmp_path, op):
    """f32(a) op f32(b) with an FP32 result reads a and b in BF16, exactly."""
    rng = np.random.default_rng(2)
    a, b = (rng.normal(size=(4, 2560)).astype(jnp.bfloat16) for _ in range(2))
    fn = getattr(jnp, op)
    actual, ir = compile_and_run(lambda a, b: fn(a.astype(jnp.float32), b.astype(jnp.float32)), tmp_path, a, b)
    assert actual.dtype == jnp.float32
    assert "ttnn.typecast" not in ir
    np.testing.assert_array_equal(actual, getattr(np, op)(a.astype(np.float32), b.astype(np.float32)))


def test_broadcast_bf16_operand_keeps_its_cast(tmp_path):
    """A broadcast BF16 operand is not read in place (binary_ng's broadcasts of BF16 into FP32 DST take other
    paths); the sum stays exact."""
    rng = np.random.default_rng(3)
    a = rng.normal(size=(4, 2560)).astype(jnp.bfloat16)
    b = rng.normal(size=(1, 2560)).astype(jnp.bfloat16)
    actual, _ = compile_and_run(lambda a, b: a.astype(jnp.float32) + b.astype(jnp.float32), tmp_path, a, b)
    np.testing.assert_array_equal(actual, a.astype(np.float32) + b.astype(np.float32))
