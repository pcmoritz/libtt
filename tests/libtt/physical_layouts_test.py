"""Large tensors stored without tile padding (ttir-physical-layouts).

TTNN pads each slice of a tiled tensor's last two dims to whole 32x32 tiles: a
rank-1 array of N elements is one row of N / 32 tiles, each padded to 32 rows,
a matrix of a few rows is padded to 32 rows too, as is each row of a [T, 1, V]
tensor. Rank-1 arrays of at least 2^20 elements and matrices of at most 8 rows
and 2^22 elements, a multiple of 32, are stored as [N / 32, 32] instead, and
[T, 1, V] tensors of at least 2^20 elements as [T, V]: as device buffers, at
program boundaries, and through the ops that have a rule for them. Other ops,
and arrays that aren't multiples of 32, still see their own shape. Each test
checks the values against NumPy, and some the shapes of the exported TTNN
program's FP32 tensors.
"""

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

N = 1 << 22
T, V = 256, 4096


def _random(shape, seed=0):
    return np.random.default_rng(seed).standard_normal(shape).astype(np.float32)


def _run(f, *args, tmp_path):
    """f's results, and the shapes of the FP32 tensors in its TTNN program."""
    got = jax.jit(f, compiler_options={"export_path": str(tmp_path)})(*args)
    ir = "".join(p.read_text() for p in (tmp_path / "irs").glob("ttnn_[0-9]*.mlir"))
    return got, set(re.findall(r"tensor<([0-9x]+)xf32", ir))


# Flat buffers and few-row matrices, folded to [N / 32, 32].


def test_elementwise_and_cast():
    x, y = _random(N), _random(N, seed=1)
    got = jax.jit(lambda x, y: (x * 2 + y).astype(jnp.bfloat16))(x, y)
    np.testing.assert_array_equal(np.asarray(got), (x * 2 + y).astype(jnp.bfloat16))


def test_slices_and_concatenation():
    x, y = _random(N), _random(N // 2, seed=1)

    def f(x, y):
        # FSDP2's flat buffers: shards concatenated, parameters sliced back out.
        flat = jnp.concatenate([x[: N // 4], y, x[N // 4 :]])
        return flat[N // 4 : N // 4 + N // 2] + 1, flat[96 : 96 + (1 << 20)]

    aligned, offset = jax.jit(f)(x, y)
    flat = np.concatenate([x[: N // 4], y, x[N // 4 :]])
    np.testing.assert_array_equal(np.asarray(aligned), y + 1)
    np.testing.assert_array_equal(np.asarray(offset), flat[96 : 96 + (1 << 20)])


@pytest.mark.parametrize(
    "start,size",
    [(0, 1 << 20), (1 << 20, 1 << 20), (N, 1 << 20), (2048 + 7, 1 << 20), (64, 1000)],
    ids=["front", "middle", "clamped", "unaligned", "small"],
)
def test_update_at_constant_start(start, size):
    # FSDP2 copies each shard into its flat all-gather buffer at a constant
    # offset. Updates off 32-element boundaries scatter rows instead.
    buffer, update = _random(N), _random(size, seed=1)
    got = jax.jit(lambda b, u: jax.lax.dynamic_update_slice(b, u, (start,)))(buffer, update)
    want = buffer.copy()
    begin = min(start, N - size)
    want[begin : begin + size] = update
    np.testing.assert_array_equal(np.asarray(got), want)


def test_views_in_other_shapes():
    # A flat parameter viewed in its own shape, and a gradient flattened back.
    x = _random(N)
    matrix, flat = jax.jit(lambda x: (x.reshape(2048, -1) * 3, (x.reshape(2048, -1) + 1).reshape(-1)))(x)
    np.testing.assert_array_equal(np.asarray(matrix), x.reshape(2048, -1) * 3)
    np.testing.assert_array_equal(np.asarray(flat), x + 1)


def test_results_feed_programs_and_host():
    # A program's folded result is another's argument, and reads back as rank 1.
    x = _random(N)
    doubled = jax.jit(lambda x: x * 2)(x)
    assert doubled.shape == (N,)
    np.testing.assert_array_equal(np.asarray(jax.jit(lambda d: d - 1)(doubled)), x * 2 - 1)
    np.testing.assert_array_equal(np.asarray(doubled), x * 2)
    # Other ops still take the rank-1 shape.
    np.testing.assert_allclose(float(jax.jit(jnp.sum)(doubled)), float(np.sum(x * 2, dtype=np.float64)), rtol=1e-4)


def test_few_row_matrices():
    # FSDP2's gradients for a reduce-scatter between 2 ranks: each gradient as [2, N / 2] in FP32, the
    # parameters' blocks side by side, then flattened for the collective.
    g1, g2 = _random(1 << 23).reshape(4096, -1), _random(1024, seed=1)

    def copy_in(g1, g2):
        pieces = [g.reshape(2, -1).astype(jnp.float32) for g in (g1, g2)]
        return jnp.concatenate(pieces, axis=1)

    stacked = jax.jit(copy_in)(g1, g2)
    want = np.concatenate([g1.reshape(2, -1), g2.reshape(2, -1)], axis=1)
    np.testing.assert_array_equal(np.asarray(stacked), want)
    flat, halves = jax.jit(lambda s: (s.reshape(-1) * 2, s[:, : 1 << 22] + s[:, 32 : 32 + (1 << 22)]))(stacked)
    np.testing.assert_array_equal(np.asarray(flat), want.reshape(-1) * 2)
    np.testing.assert_array_equal(np.asarray(halves), want[:, : 1 << 22] + want[:, 32 : 32 + (1 << 22)])


def test_unpadded_size():
    # 1 GB of FP32 per array: padded to 32 rows, each would take 32 GB, more than a chip's DRAM.
    n = 1 << 28
    x = jax.device_put(jnp.ones(n, jnp.float32))
    y = jax.jit(lambda x: x * 2 + 1)(x)
    z = jax.jit(lambda x, y: x + y)(x, y)
    # Read back whole: an op on one element would see the rank-1 shape.
    assert np.all(np.asarray(z) == 4.0)


def test_not_a_multiple_of_32():
    x = _random(N + 1)
    np.testing.assert_array_equal(np.asarray(jax.jit(lambda x: x * 2)(x)), x * 2)


# [T, 1, V] as [T, V].


def test_unit_rows():
    # Megatron's [sequence, batch, vocabulary] logits with a batch of one, through a cross entropy one op per
    # program, as miles computes log-probs eagerly: in tiles padded to 32 rows, each FP32 [T, 1, V] result would
    # take 40 GB; as [T, V], 1.2 GB.
    t, v = 2048, 151936
    rng = np.random.default_rng(0)
    host = rng.standard_normal((t, v), np.float32)
    tokens = rng.integers(0, v, t).astype(np.int32)
    logits = jax.device_put(jnp.asarray(host))
    x = jax.jit(lambda l: l[:, None, :])(logits)
    peak = jax.jit(lambda x: x.max(-1))(x)
    shifted = jax.jit(lambda x, m: x - m[..., None])(x, peak)
    total = jax.jit(lambda s: jnp.exp(s).sum(-1))(shifted)
    picked = jax.jit(lambda s, i: jnp.take_along_axis(s, i[:, None, None], axis=-1)[..., 0])(shifted, tokens)
    log_probs = np.asarray(jax.jit(lambda p, z: p - jnp.log(z))(picked, total))[:, 0]
    want = host[np.arange(t), tokens] - host.max(-1) - np.log(np.exp(host - host.max(-1, keepdims=True)).sum(-1))
    np.testing.assert_allclose(log_probs, want, rtol=1e-3, atol=1e-3)


def test_reduction_of_unit_rows_made_by_a_reshape(tmp_path):
    # miles' log-probabilities: logits.unsqueeze(1), then the row maximum.
    x = _random((T, V))
    got, shapes = _run(lambda x: jnp.max(x[:, None, :], axis=-1, keepdims=True)[:, 0, 0], x, tmp_path=tmp_path)
    np.testing.assert_array_equal(np.asarray(got), x.max(-1))
    assert "256x1x4096" not in shapes


def test_log_softmax_of_unit_rows(tmp_path):
    def log_softmax(x):
        rows = x[:, None, :]
        shifted = rows - jnp.max(rows, axis=-1, keepdims=True)
        return shifted - jnp.log(jnp.sum(jnp.exp(shifted), axis=-1, keepdims=True))

    x = _random((T, V), seed=1)
    got, shapes = _run(log_softmax, x, tmp_path=tmp_path)
    want = x - x.max(-1, keepdims=True)
    want = want - np.log(np.exp(want).sum(-1, keepdims=True))
    np.testing.assert_allclose(np.asarray(got), want[:, None, :], atol=2e-2)
    assert "256x1x4096" not in shapes


def test_softmax_slice_and_concat_of_unit_rows(tmp_path):
    def f(x, scale):
        rows = jax.nn.softmax(x[:, None, :] * scale[:, None, None], axis=-1)
        return jnp.concatenate([rows[..., V // 2 :], rows[..., : V // 2]], axis=-1)

    x, scale = _random((T, V), seed=2), np.linspace(0.5, 2, T, dtype=np.float32)
    got, shapes = _run(f, x, scale, tmp_path=tmp_path)
    e = np.exp(x * scale[:, None] - (x * scale[:, None]).max(-1, keepdims=True))
    p = e / e.sum(-1, keepdims=True)
    want = np.concatenate([p[:, V // 2 :], p[:, : V // 2]], axis=-1)[:, None, :]
    np.testing.assert_allclose(np.asarray(got), want, atol=1e-3)
    assert not any(shape.startswith("256x1x") for shape in shapes)


# Neither.


def test_shapes_without_padding_are_kept(tmp_path):
    x = _random((512, 512), seed=5)
    run = jax.jit(lambda x: jnp.exp(x) + x, compiler_options={"export_path": str(tmp_path)})
    np.testing.assert_allclose(np.asarray(run(x)), np.exp(x) + x, rtol=2e-2, atol=1e-3)
    (ir,) = [p.read_text() for p in (tmp_path / "irs").glob("ttnn_[0-9]*.mlir")]
    assert '"ttnn.reshape"' not in ir


def test_turned_off(tmp_path, monkeypatch):
    monkeypatch.setenv("LIBTT_PHYSICAL_LAYOUTS", "0")
    x = _random((T, V), seed=6)
    got, shapes = _run(lambda x: jnp.max(x[:, None, :] + 1, axis=-1, keepdims=True)[:, 0, 0], x, tmp_path=tmp_path)
    np.testing.assert_array_equal(np.asarray(got), (x + 1).max(-1))
    assert "256x1x4096" in shapes
