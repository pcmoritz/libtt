"""Non-contiguous host arrays are borrowed, not copied.

jax.device_put of a transposed or sliced numpy view, such as a checkpoint's
experts re-laid out as views of the memory-mapped file, made an owned host
tensor: a second copy of the weight in RAM, kept for as long as the array
lives. Such a view is now borrowed with its strides, and the runtime gathers it
when it is read: into the device, or back into numpy.
"""

import ml_dtypes
import numpy as np
import pytest

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P


def weight(w):
    """Marks w as a weight, prepared once rather than per call."""
    return jax.ffi.ffi_call("tt.weight_dtype_override", jax.ShapeDtypeStruct(w.shape, w.dtype))(
        w, **{"ttcore.weight_dtype": "bf16"}
    )


def strided_view(base, kind):
    half = base.shape[1] // 2
    return {
        "transposed": base.transpose(0, 2, 1),
        "sliced": base[:, :half, :],
        "sliced_transposed": base[:, half:, :].transpose(0, 2, 1),
    }[kind]


def anon_mb():
    with open("/proc/self/status") as status:
        line = next(l for l in status if l.startswith("RssAnon:"))
    return int(line.split()[1]) // 1024


@pytest.mark.parametrize("dtype", [np.float32, ml_dtypes.bfloat16])
@pytest.mark.parametrize("kind", ["transposed", "sliced", "sliced_transposed"])
def test_strided_view_is_read_back_and_computed_on(kind, dtype):
    # Small integers, exact in bf16 and when doubled.
    base = (np.arange(3 * 16 * 32) % 101).astype(dtype).reshape(3, 16, 32)
    view = strided_view(base, kind)
    assert not view.flags.c_contiguous
    y = jax.device_put(view, jax.devices("tt")[0])

    np.testing.assert_array_equal(np.asarray(y), view)
    # As a program input, and as a weight (gathered by the weight's to_device).
    np.testing.assert_array_equal(np.asarray(jax.jit(lambda a: a * 2)(y)), view * 2)
    np.testing.assert_array_equal(np.asarray(jax.jit(lambda a: weight(a) * 2)(y)), view * 2)


def test_sharded_transposed_view():
    devices = jax.devices("tt")
    mesh = Mesh(np.array(devices), ("x",))
    base = (np.arange(len(devices) * 2 * 16 * 32) % 101).astype(ml_dtypes.bfloat16)
    view = base.reshape(len(devices) * 2, 16, 32).transpose(0, 2, 1)
    y = jax.device_put(view, NamedSharding(mesh, P("x")))

    np.testing.assert_array_equal(np.asarray(y), view)
    np.testing.assert_array_equal(np.asarray(jax.jit(lambda a: weight(a) * 2)(y)), view * 2)


@pytest.mark.parametrize("trace", [False, True])
def test_column_sharded_input(trace):
    """Each device's columns are a strided view, read as a (traced) program input."""
    devices = jax.devices("tt")
    mesh = Mesh(np.array(devices), ("x",))
    x = (np.arange(2 * 248320) % 101).astype(np.float32).reshape(2, 248320)
    run = jax.jit(lambda v: v * 2, compiler_options={"enable_trace": str(trace).lower()})
    out = run(jax.device_put(x, NamedSharding(mesh, P(None, "x"))))
    np.testing.assert_array_equal(np.asarray(out), x * 2)


@pytest.mark.parametrize("mode, transposed", [("r", True), ("c", True), ("c", False)])
def test_mapped_weight_stays_in_the_file(tmp_path, mode, transposed):
    """A weight read from a file mapping is not copied for as long as it lives.

    Mode "c" is copy-on-write, as numpy.memmap and safetensors' numpy backend
    map files: pinning it for a DMA write would break its copy-on-write, page
    by page (which has also livelocked the kernel), so the write copies it.
    """
    # 64 MiB, above the size from which tt-metal pins host memory for a write.
    shape = (8, 1024, 4096)
    path = tmp_path / "weight.bin"
    values = np.memmap(path, ml_dtypes.bfloat16, mode="w+", shape=shape)
    values[:] = (np.arange(np.prod(shape)) % 101).astype(ml_dtypes.bfloat16).reshape(shape)
    values.flush()
    del values
    mapped = np.memmap(path, ml_dtypes.bfloat16, mode=mode, shape=shape)
    view = mapped.transpose(0, 2, 1) if transposed else mapped
    want = float(np.asarray(view[:, :8, :8], np.float32).sum())

    corner_sum = jax.jit(lambda a: jnp.sum(weight(a)[:, :8, :8].astype(jnp.float32)))
    device = jax.devices("tt")[0]
    assert float(corner_sum(jax.device_put(view, device))) == want  # Compiles.

    before = anon_mb()
    y = jax.device_put(view, device)
    assert float(corner_sum(y)) == want
    # A copy would be 64 MiB; the gather for the transfer is freed after it.
    assert anon_mb() - before < 16
