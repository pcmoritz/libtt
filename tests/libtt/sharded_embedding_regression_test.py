"""Embedding lookups in a table whose rows are sharded across devices."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import NamedSharding, PartitionSpec as P


@pytest.mark.parametrize("dtype", [jnp.bfloat16, np.float32])
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
    table = np.random.default_rng(0).normal(size=(rows, 256)).astype(dtype)
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
    text = ir.read_text()
    assert '"ttnn.embedding"' in text
    assert '"ttnn.all_gather"' not in text


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
