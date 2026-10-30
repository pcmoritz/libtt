"""copy_to_host_async enqueues the read right after the program that produced
the array, without blocking; the array stays usable while the read is in
flight: reading it, or passing it to another program, waits for the read.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P


def _slow(m, rounds):
    for _ in range(rounds):
        m = jnp.tanh(m @ m)
    return m


def _wait(x):
    # block_until_ready only waits for dispatch on TT; a dependent read waits
    # for the device.
    np.asarray(x[0, :1])


def test_an_array_with_a_read_in_flight_keeps_its_values():
    device = jax.devices("tt")[0]
    produce = jax.jit(lambda v: v * 2 + 1)
    consume = jax.jit(lambda x, v: x * 3 - v)
    slow = jax.jit(lambda m: _slow(m, 16))
    m = jax.device_put(np.full((1024, 1024), 1 / 1024, np.float32).astype(jnp.bfloat16), device)
    for step in range(4):
        values = np.arange(64, dtype=np.float32).reshape(2, 32) + step
        v = jax.device_put(values, device)
        x = produce(v)
        x.copy_to_host_async()
        y = slow(m)
        # The array goes back to a program while its read may be in flight.
        z = consume(x, v)
        np.testing.assert_array_equal(np.asarray(z), (values * 2 + 1) * 3 - values)
        np.testing.assert_array_equal(np.asarray(x), values * 2 + 1)
        _wait(y)


def test_copy_of_a_sharded_array():
    devices = jax.devices("tt")
    mesh = jax.make_mesh((len(devices),), ("x",), devices=devices)
    sharding = NamedSharding(mesh, P("x"))
    values = np.arange(len(devices) * 64, dtype=np.float32).reshape(len(devices) * 2, 32)
    x = jax.jit(lambda v: v + 5, out_shardings=sharding)(jax.device_put(values, sharding))
    x.copy_to_host_async()
    np.testing.assert_array_equal(np.asarray(x), values + 5)
