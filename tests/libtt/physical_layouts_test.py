"""Tensors inside a program take physical layouts without tile padding.

TTNN pads each slice of a tiled tensor's last two dims to whole 32x32 tiles,
so [T, 1, V] takes 32 times its size. ttir-physical-layouts stores such values
as [T, V] wherever the ops around them run on that shape, whether they are a
program's arguments or made inside it, as torch_tpu's eager ops make [T, 1, V]
from [T, V] with a reshape. Each test checks the values against NumPy and the
shapes of the TTNN program's FP32 tensors.
"""

import re

import jax
import jax.numpy as jnp
import numpy as np

T, V = 256, 4096


def _rows(seed, shape=(T, V)):
    return np.random.default_rng(seed).standard_normal(shape).astype(np.float32)


def _run(f, *args, tmp_path):
    """f's results, and the shapes of the FP32 tensors in its TTNN program."""
    got = jax.jit(f, compiler_options={"export_path": str(tmp_path)})(*args)
    ir = "".join(p.read_text() for p in (tmp_path / "irs").glob("ttnn_[0-9]*.mlir"))
    return got, set(re.findall(r"tensor<([0-9x]+)xf32", ir))


def test_reduction_of_unit_rows_made_by_a_reshape(tmp_path):
    # miles' log-probabilities: logits.unsqueeze(1), then the row maximum.
    x = _rows(0)
    got, shapes = _run(lambda x: jnp.max(x[:, None, :], axis=-1, keepdims=True)[:, 0, 0], x, tmp_path=tmp_path)
    np.testing.assert_array_equal(np.asarray(got), x.max(-1))
    assert "256x1x4096" not in shapes


def test_log_softmax_of_unit_rows(tmp_path):
    def log_softmax(x):
        rows = x[:, None, :]
        shifted = rows - jnp.max(rows, axis=-1, keepdims=True)
        return shifted - jnp.log(jnp.sum(jnp.exp(shifted), axis=-1, keepdims=True))

    x = _rows(1)
    got, shapes = _run(log_softmax, x, tmp_path=tmp_path)
    want = x - x.max(-1, keepdims=True)
    want = want - np.log(np.exp(want).sum(-1, keepdims=True))
    np.testing.assert_allclose(np.asarray(got), want[:, None, :], atol=2e-2)
    assert "256x1x4096" not in shapes


def test_softmax_slice_and_concat_of_unit_rows(tmp_path):
    def f(x, scale):
        rows = jax.nn.softmax(x[:, None, :] * scale[:, None, None], axis=-1)
        return jnp.concatenate([rows[..., V // 2 :], rows[..., : V // 2]], axis=-1)

    x, scale = _rows(2), np.linspace(0.5, 2, T, dtype=np.float32)
    got, shapes = _run(f, x, scale, tmp_path=tmp_path)
    e = np.exp(x * scale[:, None] - (x * scale[:, None]).max(-1, keepdims=True))
    p = e / e.sum(-1, keepdims=True)
    want = np.concatenate([p[:, V // 2 :], p[:, : V // 2]], axis=-1)[:, None, :]
    np.testing.assert_allclose(np.asarray(got), want, atol=1e-3)
    assert not any(shape.startswith("256x1x") for shape in shapes)


def test_shapes_without_padding_are_kept(tmp_path):
    x = _rows(5, (512, 512))
    run = jax.jit(lambda x: jnp.exp(x) + x, compiler_options={"export_path": str(tmp_path)})
    np.testing.assert_allclose(np.asarray(run(x)), np.exp(x) + x, rtol=2e-2, atol=1e-3)
    (ir,) = [p.read_text() for p in (tmp_path / "irs").glob("ttnn_[0-9]*.mlir")]
    assert '"ttnn.reshape"' not in ir


def test_turned_off(tmp_path, monkeypatch):
    monkeypatch.setenv("LIBTT_PHYSICAL_LAYOUTS", "0")
    x = _rows(6)
    got, shapes = _run(lambda x: jnp.max(x[:, None, :] + 1, axis=-1, keepdims=True)[:, 0, 0], x, tmp_path=tmp_path)
    np.testing.assert_array_equal(np.asarray(got), (x + 1).max(-1))
    assert "256x1x4096" in shapes
