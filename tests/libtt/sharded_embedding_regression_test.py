"""Embedding lookups in a table whose rows are sharded across devices."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import NamedSharding, PartitionSpec as P


@pytest.mark.parametrize("dtype", [jnp.bfloat16, np.float16, np.float32, np.int32, np.uint32])
@pytest.mark.parametrize("ids_shape", [(8,), (2, 5)])
def test_row_sharded_embedding(dtype, ids_shape, tmp_path):
    # Like a vocabulary-parallel token embedding: each device holds a range of
    # rows, and the lookup is replicated. The table must not be all-gathered
    # onto every device.
    devices = jax.devices("tt")
    if len(devices) < 2:
        pytest.skip("needs at least two devices")
    mesh = jax.make_mesh((1, len(devices)), ("data", "tensor"), devices=devices)
    rows = 64 * len(devices)
    rng = np.random.default_rng(0)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        table = rng.integers(info.min, info.max, size=(rows, 256), dtype=dtype, endpoint=True)
    else:
        table = rng.normal(size=(rows, 256)).astype(dtype)
    # The first and last rows of every device, and a negative index.
    ids = [0, 63, 64, 127, rows - 1, -1, 5, rows - 64]
    ids = np.resize(np.array(ids, np.int32), ids_shape)

    run = jax.jit(
        lambda t, i: t.at[i].get(out_sharding=NamedSharding(mesh, P())),
        compiler_options={
            "export_path": str(tmp_path),
            "export_model_name": "embedding",
            "export_tensors": "false",
        },
    )
    with jax.set_mesh(mesh):
        out = run(
            jax.device_put(table, NamedSharding(mesh, P("tensor", None))),
            jax.device_put(ids, NamedSharding(mesh, P())),
        )
    np.testing.assert_array_equal(np.asarray(out), table[ids])

    (ir,) = (tmp_path / "irs").glob("ttnn_runtime_embedding_*.mlir")
    assert '"ttnn.all_gather"' not in ir.read_text()


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
    devices = jax.devices("tt")
    if len(devices) != 4:
        pytest.skip("needs four devices")
    mesh = jax.make_mesh((2, 2), ("x", "y"), devices=devices)
    table = np.random.default_rng(0).normal(size=(256, 256)).astype(np.float32)
    ids = np.array([0, 63, 64, 127, 128, 255, 200, 5], np.int32)

    run = jax.jit(
        lambda t, i: t.at[i].get(out_sharding=NamedSharding(mesh, ids_spec)),
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
    np.testing.assert_array_equal(np.asarray(out), table[ids])

    (ir,) = (tmp_path / "irs").glob("ttnn_runtime_embedding_*.mlir")
    assert '"ttnn.all_gather"' not in ir.read_text()


def test_lookup_in_shard_map(tmp_path):
    # A shard_map body looks up its own shard of the table with indices it made
    # local itself; they must not be shifted again.
    devices = jax.devices("tt")
    if len(devices) < 2:
        pytest.skip("needs at least two devices")
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


def test_out_of_range_indices_are_clamped():
    # Gather clamps indices into the whole table, so an index past the end
    # reads the last row rather than zeros. (JAX wraps negative indices.)
    devices = jax.devices("tt")
    if len(devices) < 2:
        pytest.skip("needs at least two devices")
    mesh = jax.make_mesh((1, len(devices)), ("data", "tensor"), devices=devices)
    rows = 64 * len(devices)
    table = np.random.default_rng(0).normal(size=(rows, 256)).astype(np.float32)
    ids = np.array([rows, rows + 70, 2**30, rows + 1, 0, rows - 1, 64, 3], np.int32)

    run = jax.jit(lambda t, i: t.at[i].get(mode="clip", out_sharding=NamedSharding(mesh, P())))
    with jax.set_mesh(mesh):
        out = run(
            jax.device_put(table, NamedSharding(mesh, P("tensor", None))),
            jax.device_put(ids, NamedSharding(mesh, P())),
        )
    np.testing.assert_array_equal(np.asarray(out), table[np.minimum(ids, rows - 1)])
