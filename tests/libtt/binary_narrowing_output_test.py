"""An FP32 elementwise product, sum, difference or quotient cast to BF16 is one
binary op with a BF16 output: binary_ng computes in FP32 and rounds once as it
packs (exact ties may round away from zero where a typecast rounds to even)."""

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
