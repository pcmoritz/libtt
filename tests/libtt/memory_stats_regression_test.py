"""Device memory stats report the DRAM that device-resident arrays use."""

import jax
import jax.numpy as jnp


def test_bytes_in_use_counts_device_arrays():
    device = jax.devices()[0]
    before = device.memory_stats()["bytes_in_use"]
    # Program outputs live on the device; inputs stay on the host until used.
    arrays = jax.jit(lambda: jnp.ones((1024, 1024), jnp.float32) * 2)()
    arrays.block_until_ready()
    stats = device.memory_stats()
    assert stats["bytes_in_use"] - before >= 1024 * 1024 * 4
    assert stats["bytes_limit"] > stats["bytes_in_use"]
    del arrays
