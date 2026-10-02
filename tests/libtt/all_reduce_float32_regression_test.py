"""Cross-device sums of float32 values must keep float32 precision."""

import jax
import numpy as np
import pytest
from jax.sharding import NamedSharding, PartitionSpec as P


@pytest.mark.parametrize("collective", ["psum", "psum_scatter"])
def test_float32_sum_precision(collective):
    # Adding on the FPU keeps only TF32 (10 mantissa bits): relative errors
    # around 1e-4 instead of float32's ~1e-7.
    devices = jax.devices("tt")
    if len(devices) < 2:
        pytest.skip("needs at least two devices")
    n = len(devices)
    mesh = jax.make_mesh((n,), ("x",), devices=devices)
    x = np.random.default_rng(0).normal(size=(n, 64, 256)).astype(np.float32)
    if collective == "psum":
        body, out_specs = (lambda a: jax.lax.psum(a, "x")), P("x")
    else:
        body = lambda a: jax.lax.psum_scatter(a[0], "x", scatter_dimension=0, tiled=True)[None]
        out_specs = P("x")
    run = jax.jit(jax.shard_map(body, mesh=mesh, in_specs=P("x"), out_specs=out_specs))
    out = np.asarray(run(jax.device_put(x, NamedSharding(mesh, P("x")))))

    total = x.astype(np.float64).sum(0)
    expected = total[None].repeat(n, 0) if collective == "psum" else total.reshape(n, 64 // n, 256)
    # A float32 sum of n values is within a few float32 rounding steps of the
    # exact sum.
    np.testing.assert_allclose(out, expected, rtol=0, atol=8 * n * np.finfo(np.float32).eps * np.abs(x).max())
