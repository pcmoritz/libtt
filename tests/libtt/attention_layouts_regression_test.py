"""Attention's dot_generals and transposes lower without tile-padded layouts.

TTNN pads the last two dims of a tiled tensor to whole 32x32 tiles, so a
tensor with a small or size-1 dim among its last two takes many times its
size, and the ops that make and read it move all of that. Two such layouts
dominated torchtitan's Qwen3-1.7B backward on one Blackhole:

- The gradient dot_general of grouped-query attention contracts over (group,
  query): dS[b, n, g, t, s] with q[b, n, g, t, h]. Permuting dS to
  [b, n, s, g, t] put the group, of size 2, second to last. dS is already
  ordered (batch, contract, result), so it goes into the matmul as it is, with
  transpose_a.
- torch_tpu transposes [B, S, N, H] to [B, N, S, H] with B = 1 as a reshape to
  [S, N, H, 1] and a permute, a size-1 dim last; the permute now moves only
  the other dims, between reshapes that fold into their neighbours.
"""

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from jax import lax


def _small_integers(rng, shape):
    # Products and sums of these are exact in bfloat16 inputs and f32 sums.
    return rng.integers(-2, 3, shape).astype(ml_dtypes.bfloat16)


def _ttnn_irs(tmp_path):
    irs = [path.read_text() for path in (tmp_path / "irs").glob("ttnn_[0-9]*.mlir")]
    assert irs
    return irs


def test_grouped_query_attention_gradient_contracts_without_a_permute(tmp_path):
    rng = np.random.default_rng(0)
    b, n, g, t, s, h = 1, 2, 2, 256, 256, 128
    ds = _small_integers(rng, (b, n, g, t, s))
    q = _small_integers(rng, (b, n, g, t, h))
    dimension_numbers = (((2, 3), (2, 3)), ((0, 1), (0, 1)))
    run = jax.jit(
        lambda ds, q: lax.dot_general(ds, q, dimension_numbers, preferred_element_type=jnp.float32),
        compiler_options={"export_path": str(tmp_path)},
    )
    device = jax.devices("tt")[0]
    got = np.asarray(run(jax.device_put(ds, device), jax.device_put(q, device)))
    want = np.einsum("bngts,bngth->bnsh", ds.astype(np.float32), q.astype(np.float32))
    np.testing.assert_array_equal(got, want)
    irs = _ttnn_irs(tmp_path)
    assert all('"ttnn.permute"' not in ir for ir in irs)
    assert any("transpose_a = true" in ir for ir in irs)


def test_transpose_moving_a_size_one_dim_keeps_it_out_of_tiles(tmp_path):
    rng = np.random.default_rng(1)
    s, n, h = 256, 16, 128
    x = _small_integers(rng, (1, s, n, h))

    def heads(x):
        # [B, S, N, H] to [B, N, S, H] and back to [S, N * H], as torch_tpu
        # writes them for B = 1.
        y = jnp.transpose(x.reshape(s, n, h, 1), (3, 1, 0, 2)) * 2
        return jnp.transpose(y, (2, 1, 3, 0)).reshape(s, n * h), y

    run = jax.jit(heads, compiler_options={"export_path": str(tmp_path)})
    merged, split = run(jax.device_put(x, jax.devices("tt")[0]))
    xf = x.astype(np.float32)
    np.testing.assert_array_equal(np.asarray(split).astype(np.float32), 2 * xf.transpose(0, 2, 1, 3))
    np.testing.assert_array_equal(np.asarray(merged).astype(np.float32), 2 * xf.reshape(s, n * h))
    irs = _ttnn_irs(tmp_path)
    assert all("x128x1x" not in ir for ir in irs), "a size-1 dim is last but one tile dim"
