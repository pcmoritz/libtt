"""Elementwise ops next to a reshape run on the side with fewer tiles.

TTNN tiles the last two dims, so [1, 10240] takes 320 tiles where its
[1, 4, 2560] view takes 80. A typecast before such a reshape, or a multiply by
a constant and a typecast after one, move to the [1, 4, 2560] side.
"""

import re

import jax
import jax.numpy as jnp
import numpy as np


def compile_and_run(fn, tmp_path, *args):
    run = jax.jit(fn, compiler_options={"optimization_level": "O1", "export_path": str(tmp_path)})
    device = jax.devices("tt")[0]
    out = run(*(jax.device_put(a, device) for a in args))
    (ir,) = [path.read_text() for path in (tmp_path / "irs").glob("ttnn_[0-9]*.mlir")]
    return np.asarray(out).astype(np.float32), ir


def result_shapes(ir, op):
    return re.findall(rf'"ttnn\.{op}".*?-> tensor<([0-9x]+)x[a-z0-9]+[,>]', ir)


def test_typecast_before_a_reshape_runs_after_it(tmp_path):
    x = np.random.default_rng(0).normal(size=(1, 10240)).astype(jnp.bfloat16)
    actual, ir = compile_and_run(lambda v: v.astype(jnp.float32).reshape(1, 4, 2560) * 3.0, tmp_path, x)
    assert result_shapes(ir, "typecast") == ["1x4x2560"]
    np.testing.assert_allclose(actual, x.astype(np.float32).reshape(1, 4, 2560) * 3.0, rtol=1e-6)


def test_multiply_and_typecast_after_a_reshape_run_before_it(tmp_path):
    rng = np.random.default_rng(1)
    v = rng.normal(size=(1, 4, 2560)).astype(np.float32)
    w = rng.normal(size=10240).astype(np.float32)

    def fn(v):
        return (jnp.exp(v).reshape(1, 10240) * jnp.asarray(w)).astype(jnp.bfloat16)

    actual, ir = compile_and_run(fn, tmp_path, v)
    assert result_shapes(ir, "multiply") == ["1x4x2560"]
    # The cast to BF16 folds into the multiply's output.
    assert result_shapes(ir, "typecast") == []
    want = (np.exp(v).reshape(1, 10240) * w).astype(jnp.bfloat16).astype(np.float32)
    np.testing.assert_allclose(actual, want, rtol=1e-2, atol=1e-2)
