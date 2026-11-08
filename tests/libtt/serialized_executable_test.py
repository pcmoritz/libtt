"""An executable that PJRT_Executable_Serialize wrote loads back with
PJRT_Executable_DeserializeAndLoad, which compiles its program only to TTIR, for
its arguments and outputs, and takes the TTNN program and flatbuffer from the
payload. torch.compile's caches store torch_tpu's executables that way, and JAX's
serialize_executable does too.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src.lib import xla_client as xc
from jax.experimental import serialize_executable

# The serialized header: an 8 byte magic string, then an offset and a size for
# each of its six sections, the last of which is the plugin's build ID.
HEADER_SIZE = 8 + 6 * 2 * 8
BUILD_ID_OFFSET = HEADER_SIZE - 2 * 8


def device_put(x):
    return jax.device_put(x, jax.devices("tt")[0])


def round_trip(compiled):
    return serialize_executable.deserialize_and_load(
        *serialize_executable.serialize(compiled), execution_devices=jax.devices("tt")[:1]
    )


def serialize(compiled):
    client = jax.devices("tt")[0].client
    return client.serialize_executable(compiled.runtime_executable())


def deserialize(payload, compile_options=None):
    client = jax.devices("tt")[0].client
    return client.deserialize_executable(payload, xc.DeviceList(tuple(jax.devices("tt")[:1])), compile_options)


def compile_options(**overrides):
    options = xc.CompileOptions()
    options.env_option_overrides = list(overrides.items())
    return options


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


def test_executable_from_another_build_does_not_load():
    """Another build may lower the program differently, so loading its executable
    fails, and callers such as JAX's persistent cache compile it again."""
    payload = bytearray(serialize(jax.jit(lambda x: x + 1).lower(device_put(np.ones(32, np.float32))).compile()))
    # JAX wraps the plugin's payload in IFRT's own header.
    start = payload.find(b"TTSERv01")
    build_id_offset, build_id_size = np.frombuffer(payload, np.uint64, 2, start + BUILD_ID_OFFSET)
    assert build_id_size > 0
    payload[start + HEADER_SIZE + int(build_id_offset)] ^= 0xFF
    with pytest.raises(Exception, match="this build of the plugin"):
        deserialize(bytes(payload))


def test_loaded_executable_takes_overriding_compile_options():
    """JAX's persistent cache loads executables with the options it compiles with,
    which replace the serialized ones."""
    x = device_put(np.arange(32, dtype=np.float32))
    payload = serialize(jax.jit(lambda x: x * 3).lower(x).compile())
    loaded = deserialize(payload, compile_options())
    (got,), = loaded.execute_sharded([x]).disassemble_into_single_device_arrays()
    np.testing.assert_array_equal(np.asarray(got), np.arange(32, dtype=np.float32) * 3)
    with pytest.raises(Exception, match="No such compile option"):
        deserialize(payload, compile_options(no_such_option="1"))


def test_overriding_compile_options_keep_the_flatbuffer_backend(tmp_path):
    payload = serialize(jax.jit(lambda x: x - 1).lower(device_put(np.ones(32, np.float32))).compile())
    with pytest.raises(Exception, match="Only flatbuffer executables can be loaded"):
        deserialize(payload, compile_options(backend="codegen_py", export_path=str(tmp_path)))

