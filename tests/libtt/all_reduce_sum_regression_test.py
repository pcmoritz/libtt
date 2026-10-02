"""Cross-device sums of 32-bit values must be exact to their data type."""

import jax
import numpy as np
import pytest
from jax.sharding import NamedSharding, PartitionSpec as P


@pytest.mark.parametrize("dtype", [np.float32, np.int32, np.uint32])
@pytest.mark.parametrize("collective", ["psum", "psum_scatter"])
def test_32bit_sum(collective, dtype):
    # Adding on the FPU keeps only TF32 (10 mantissa bits): float32 sums were
    # off by around 1e-4 relative, and integer sums were wrong.
    devices = jax.devices("tt")
    if len(devices) < 2:
        pytest.skip("needs at least two devices")
    n = len(devices)
    mesh = jax.make_mesh((n,), ("x",), devices=devices)
    rng = np.random.default_rng(0)
    if dtype == np.float32:
        x = rng.normal(size=(n, 64, 256)).astype(dtype)
    else:
        info = np.iinfo(dtype)
        x = rng.integers(info.min, info.max, size=(n, 64, 256), dtype=dtype, endpoint=True)
    if collective == "psum":
        body = lambda a: jax.lax.psum(a, "x")
    else:
        body = lambda a: jax.lax.psum_scatter(a[0], "x", scatter_dimension=0, tiled=True)[None]
    run = jax.jit(jax.shard_map(body, mesh=mesh, in_specs=P("x"), out_specs=P("x")))
    out = np.asarray(run(jax.device_put(x, NamedSharding(mesh, P("x")))))

    if dtype == np.float32:
        total = x.astype(np.float64).sum(0)
    else:
        total = x.sum(0, dtype=dtype)  # Wraps around like the device.
    expected = total[None].repeat(n, 0) if collective == "psum" else total.reshape(n, 64 // n, 256)
    if dtype == np.float32:
        # A float32 sum of n values is within a few float32 rounding steps of
        # the exact sum.
        atol = 8 * n * np.finfo(np.float32).eps * np.abs(x).max()
        np.testing.assert_allclose(out, expected, rtol=0, atol=atol)
    else:
        np.testing.assert_array_equal(out, expected)
