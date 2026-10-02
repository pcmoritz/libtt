"""Embedding lookups in a table whose rows are sharded across devices."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import NamedSharding, PartitionSpec as P


def _devices(count=None):
    devices = jax.devices("tt")
    if len(devices) < 2 or count not in (None, len(devices)):
        pytest.skip(f"needs {count or 'at least two'} devices")
    return devices


def _check_lookup(tmp_path, mesh, table, ids, table_spec, ids_spec=P()):
    # The table must not be all-gathered onto every device. Gather clamps
    # indices past the end into the table, and JAX wraps negative ones.
    run = jax.jit(
        lambda t, i: t.at[i].get(mode="clip", out_sharding=NamedSharding(mesh, ids_spec)),
        compiler_options={
            "export_path": str(tmp_path),
            "export_model_name": "embedding",
            "export_tensors": "false",
        },
    )
    with jax.set_mesh(mesh):
        out = run(
            jax.device_put(table, NamedSharding(mesh, table_spec)),
            jax.device_put(ids, NamedSharding(mesh, ids_spec)),
        )
    np.testing.assert_array_equal(np.asarray(out), table[np.minimum(ids, len(table) - 1)])
    (ir,) = (tmp_path / "irs").glob("ttnn_runtime_embedding_*.mlir")
    assert '"ttnn.all_gather"' not in ir.read_text()


@pytest.mark.parametrize("dtype", [jnp.bfloat16, np.float16, np.float32, np.int32, np.uint32])
@pytest.mark.parametrize("ids_shape", [(10,), (2, 5)])
def test_row_sharded_embedding(dtype, ids_shape, tmp_path):
    # Like a vocabulary-parallel token embedding: each device holds a range of
    # rows, and the lookup is replicated.
    devices = _devices()
    mesh = jax.make_mesh((1, len(devices)), ("data", "tensor"), devices=devices)
    rows = 64 * len(devices)
    rng = np.random.default_rng(0)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        table = rng.integers(info.min, info.max, size=(rows, 256), dtype=dtype, endpoint=True)
    else:
        table = rng.normal(size=(rows, 256)).astype(dtype)
    # The first and last rows of every device, a negative index, and indices
    # past the end.
    ids = [0, 63, 64, 127, rows - 1, -1, 5, rows - 64, rows, 2**30]
    _check_lookup(tmp_path, mesh, table, np.array(ids, np.int32).reshape(ids_shape), P("tensor", None))


# On a 2x2 mesh the device's shard comes from its mesh coordinates, and the
# indices may be sharded along the other axis.
@pytest.mark.parametrize(
    "table_spec,ids_spec",
    [
        (P("x", None), P()),
        (P("y", None), P()),
        (P(("x", "y"), None), P()),
        (P(("y", "x"), None), P()),
        (P("y", None), P("x")),
    ],
)
def test_row_sharded_embedding_2d_mesh(table_spec, ids_spec, tmp_path):
    mesh = jax.make_mesh((2, 2), ("x", "y"), devices=_devices(4))
    table = np.random.default_rng(0).normal(size=(256, 256)).astype(np.float32)
    ids = np.array([0, 63, 64, 127, 128, 255, 200, 5], np.int32)
    _check_lookup(tmp_path, mesh, table, ids, table_spec, ids_spec)


def test_lookup_in_shard_map():
    # A shard_map body looks up its own shard of the table with indices it made
    # local itself; they must not be shifted again.
    devices = _devices()
    n = len(devices)
    mesh = jax.make_mesh((1, n), ("data", "tensor"), devices=devices)
    table = np.random.default_rng(0).normal(size=(64 * n, 256)).astype(np.float32)
    ids = np.array([0, 63, 64, 127, 5, 70, 1, 2], np.int32) % (64 * n)

    def body(shard, start, i):
        local = i - start[0]
        clamped = jnp.clip(local, 0, shard.shape[0] - 1)
        rows = jnp.where((clamped == local)[:, None], jnp.take(shard, clamped, axis=0), 0)
        return jax.lax.psum(rows, "tensor")

    def lookup(t, i):
        starts = jnp.arange(n, dtype=jnp.int32) * (t.shape[0] // n)
        return jax.shard_map(
            body,
            mesh=mesh,
            in_specs=(P("tensor"), P("tensor"), P()),
            out_specs=P(),
            axis_names={"tensor"},
        )(t * 2, starts, i)

    with jax.set_mesh(mesh):
        out = jax.jit(lookup)(
            jax.device_put(table, NamedSharding(mesh, P("tensor", None))),
            jax.device_put(ids, NamedSharding(mesh, P())),
        )
    np.testing.assert_array_equal(np.asarray(out), table[ids] * 2)
