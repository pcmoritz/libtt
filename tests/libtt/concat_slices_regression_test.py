"""Concatenations of slices of one tensor, which the compiler may fold into one slice."""

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize(
    "bounds",
    [
        # Adjacent ranges, like [q | k | v] rebuilt from a fused projection.
        [(0, 512), (512, 1024), (1024, 2560)],
        # A gap and a reversed order must stay concatenations.
        [(0, 512), (1024, 2560)],
        [(512, 1024), (0, 512)],
    ],
)
def test_concat_of_slices(bounds):
    def run(x):
        y = x @ jnp.eye(x.shape[1], dtype=x.dtype)
        return jnp.concatenate([y[:, None, a:b] for a, b in bounds], axis=2)

    x = (np.arange(2 * 4096) % 97 / 8).astype(jnp.bfloat16).reshape(2, 4096)
    out = jax.jit(run)(jax.device_put(x, jax.devices("tt")[0]))
    expected = np.concatenate([x[:, None, a:b] for a, b in bounds], axis=2)
    np.testing.assert_array_equal(np.asarray(out), expected)


def test_concat_of_slices_reshaped_to_rows(tmp_path):
    """PyTorch splits a flattened [q | k | v | z] projection into vectors and
    concatenates [q | k | v] as rows; adjacent slices still fold into one."""

    def run(x):
        flat = (x @ jnp.eye(x.shape[1], dtype=x.dtype)).reshape(-1)
        parts = [flat[a:b].reshape(1, -1) for a, b in [(0, 512), (512, 1024), (1024, 2560)]]
        return jnp.concatenate(parts, axis=1)

    x = (np.arange(4096) % 97 / 8).astype(jnp.bfloat16).reshape(1, 4096)
    compiled = jax.jit(run, compiler_options={"export_path": str(tmp_path)})
    out = compiled(jax.device_put(x, jax.devices("tt")[0]))
    np.testing.assert_array_equal(np.asarray(out), x[:, :2560])
    ir = "\n".join(p.read_text() for p in (tmp_path / "irs").glob("ttnn_[0-9]*.mlir"))
    assert ir and '"ttnn.concat"' not in ir, "the slices were concatenated"


def test_slices_of_a_row_through_its_vector(tmp_path):
    """Halves of a [1, N] row taken through its [N] vector, as PyTorch splits a
    one-token gate/up projection, are slices of the row itself."""

    def run(x):
        flat = (x * 2).reshape(-1)
        return flat[:4352].reshape(1, -1) * flat[4352:].reshape(1, -1)

    x = (np.arange(8704) % 97 / 8).astype(jnp.bfloat16).reshape(1, 8704)
    compiled = jax.jit(run, compiler_options={"export_path": str(tmp_path)})
    out = compiled(jax.device_put(x, jax.devices("tt")[0]))
    y = x.astype(np.float32) * 2
    np.testing.assert_array_equal(np.asarray(out), (y[:, :4352] * y[:, 4352:]).astype(jnp.bfloat16))
    ir = "\n".join(p.read_text() for p in (tmp_path / "irs").glob("ttnn_[0-9]*.mlir"))
    assert ir and '"ttnn.reshape"' not in ir, "the row was sliced through its vector"


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
def test_rows_of_a_reshaped_row(tmp_path, dtype):
    """torch_tpu lowers the per-head views of a one-token [q | k | v]
    projection, [1, 1536] -> [12, 128] -> rows, as row slices; the k and v rows
    start inside a tile and are column slices of the [1, 1536] row instead."""

    def run(x, w):
        heads = (x * 2).reshape(12, 128).astype(jnp.bfloat16)
        return heads[:8] * w, heads[8:10] * w, heads[10:] * w

    x = (np.arange(1536) % 17 - 8).astype(dtype).reshape(1, 1536)
    w = (np.arange(128) % 15 - 7).astype(jnp.bfloat16)
    device = jax.devices("tt")[0]
    compiled = jax.jit(run, compiler_options={"export_path": str(tmp_path)})
    out = compiled(jax.device_put(x, device), jax.device_put(w, device))
    heads = (x.astype(np.float32) * 2).reshape(12, 128) * w.astype(np.float32)
    for got, rows in zip(out, [heads[:8], heads[8:10], heads[10:]]):
        np.testing.assert_array_equal(np.asarray(got).astype(np.float32), rows)
    ir = "\n".join(p.read_text() for p in (tmp_path / "irs").glob("ttnn_[0-9]*.mlir"))
    assert ir and not re.search(r'"ttnn\.slice_static".*tensor<12x128x\w+.*-> tensor<2x128x', ir), (
        "rows inside a tile were sliced from the reshaped row"
    )
