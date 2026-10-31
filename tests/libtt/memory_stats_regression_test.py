"""Device memory stats report the DRAM that device arrays use, and its peak."""

import jax
import jax.numpy as jnp
import numpy as np

MIB = 1024 * 1024


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


def test_bytes_in_use_counts_transferred_arrays():
    # A transferred array's data stays on the host until a program uses it,
    # but the array holds device memory from the transfer on.
    device = jax.devices()[0]
    before = device.memory_stats()["bytes_in_use"]
    array = jax.device_put(np.ones((1024, 1024), np.float32), device)
    array.block_until_ready()
    assert device.memory_stats()["bytes_in_use"] - before >= 4 * MIB
    del array
    assert device.memory_stats()["bytes_in_use"] - before < 4 * MIB


def test_complex_arrays_count_both_parts():
    device = jax.devices()[0]
    before = device.memory_stats()["bytes_in_use"]
    array = jax.device_put(np.ones(1024 * 1024, np.complex64), device)
    array.block_until_ready()
    assert device.memory_stats()["bytes_in_use"] - before >= 8 * MIB
    del array


def test_peak_bytes_in_use_keeps_the_high_point():
    device = jax.devices()[0]
    array = jax.jit(lambda: jnp.ones((2048, 2048), jnp.float32))()
    array.block_until_ready()
    high = device.memory_stats()["bytes_in_use"]
    del array
    stats = device.memory_stats()
    assert stats["bytes_in_use"] <= high - 16 * MIB
    assert stats["peak_bytes_in_use"] >= high
