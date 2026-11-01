"""An executable that PJRT_Executable_Serialize wrote loads back with
PJRT_Executable_DeserializeAndLoad, which compiles its program only to TTIR, for
its arguments and outputs, and takes the TTNN program and flatbuffer from the
payload. torch.compile's caches store torch_tpu's executables that way, and JAX's
serialize_executable does too.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import serialize_executable


def device_put(x):
    return jax.device_put(x, jax.devices("tt")[0])


def round_trip(compiled):
    return serialize_executable.deserialize_and_load(
        *serialize_executable.serialize(compiled), execution_devices=jax.devices("tt")[:1]
    )


def test_loaded_executable_runs_like_the_compiled_one():
    rng = np.random.default_rng(0)
    x = device_put(rng.standard_normal((64, 128)).astype(np.float32))
    w = device_put(rng.standard_normal((128, 32)).astype(np.float32))
    compiled = jax.jit(lambda x, w: (jnp.tanh(x @ w), jnp.sum(x, axis=0))).lower(x, w).compile()
    loaded = round_trip(compiled)
    for got, expected in zip(loaded(x, w), compiled(x, w)):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(expected))


def test_loaded_executable_keeps_donation():
    x = device_put(np.arange(1024, dtype=np.float32).reshape(32, 32))
    compiled = jax.jit(lambda x: x * 2, donate_argnums=0).lower(x).compile()
    loaded = round_trip(compiled)
    np.testing.assert_array_equal(np.asarray(loaded(x)), np.arange(1024, dtype=np.float32).reshape(32, 32) * 2)
    assert x.is_deleted()

