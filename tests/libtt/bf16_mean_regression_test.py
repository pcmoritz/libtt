"""jnp.mean of a BF16 tensor reaches the compiler as a cast to FP32, an FP32 mean
and a cast back (Qwen3.8-Flash-Next's hyper-connection stream mean). Over the
second-to-last dimension, of at most 32 elements, it runs as one BF16 mean into
FP32 DST. The FPU sums a tile's rows with a narrower adder than the FP32 mean's
SFPU path: errors stay near 2^-15 of the inputs' magnitude, well under the BF16
inputs' own rounding. Means along rows and over several dimensions lose more and
keep the FP32 path."""

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


def check_mean(actual, x, axis):
    exact = x.astype(np.float64).mean(axis)
    scale = np.abs(x.astype(np.float64)).max(axis)
    got = actual.astype(np.float64)
    ulp = np.ldexp(1.0, np.frexp(np.abs(got))[1] - 8)  # BF16 keeps 8 significant bits
    assert np.all(np.abs(got - exact) <= np.ldexp(scale, -13) + ulp / 2)


@pytest.mark.parametrize("shape,axis", [((1, 4, 2560), 1), ((8, 32, 64), 1), ((8, 16, 64), -2)])
def test_short_mean_runs_in_bf16(tmp_path, shape, axis):
    x = np.random.default_rng(0).normal(size=shape).astype(jnp.bfloat16)
    actual, ir = compile_and_run(lambda v: jnp.mean(v, axis), tmp_path, x)
    assert actual.dtype == jnp.bfloat16
    assert "ttnn.mean" in ir and "ttnn.typecast" not in ir
    check_mean(actual, x, axis)


@pytest.mark.parametrize("shape,axis", [((2, 64, 96), 1), ((2, 4, 96), (0, 1)), ((64, 24), -1)])
def test_other_means_keep_fp32(tmp_path, shape, axis):
    x = np.random.default_rng(1).normal(size=shape).astype(jnp.bfloat16)
    actual, ir = compile_and_run(lambda v: jnp.mean(v, axis), tmp_path, x)
    assert "ttnn.typecast" in ir
    check_mean(actual, x, axis)
