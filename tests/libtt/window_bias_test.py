"""tt.window_bias: the decode attention's relative-position window bias (sglang_tpu_kernels._paged_decode's
_window_bias_call on _window_bias_table) runs as one ttnn.window_bias and matches _decode_window_bias's definition in
float64 (column j of the window of W keys ending at the position's key tile gets rel_r . rel_proj[:, d] / sm_scale for
its distance d in [0, extent), else 0) within BF16 rounding; the sink tiles hold sink / sm_scale in column 0."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_plugins.libtt import sglang_tpu_kernels as kernels


def reference(rel_r, rel_proj, positions, sm_scale):
    rel_dim, extent = rel_proj.shape
    window_tiles = -(-extent // 32) + 1
    width = 32 * window_tiles
    tokens, heads, _ = rel_r.shape
    bias = np.zeros((tokens, 1, heads, width))
    for t, p in enumerate(positions):
        first = 32 * max(0, p // 32 + 1 - window_tiles)
        d = p - first - np.arange(width)
        valid = (d >= 0) & (d < extent)
        columns = np.where(valid[None], rel_proj[:, np.clip(d, 0, extent - 1)], 0.0)  # [rel_dim, width]
        bias[t, 0] = rel_r[t].astype(np.float64) @ columns / sm_scale
    return bias


@pytest.mark.parametrize("trace", [False, True])
@pytest.mark.parametrize(
    "positions,sink", [([0], True), ([7], False), ([1055], True), ([31, 32, 1056, 5000], True), (list(range(5, 1700, 53)), True)]
)
def test_window_bias(trace, positions, sink, tmp_path):
    heads, extent, sm_scale = 6, 1024, 128**-0.5
    tokens = len(positions)

    def run(rel_r, rel_proj, positions, sink):
        table = kernels._window_bias_table(rel_proj, sm_scale)
        return kernels._window_bias_call(rel_r, table, positions, sink, sm_scale)

    run = jax.jit(run, compiler_options={"enable_trace": str(trace).lower(), "export_path": str(tmp_path)})
    device = jax.devices("tt")[0]
    rng = np.random.default_rng(tokens * 7 + positions[0])
    bf16 = lambda x: np.asarray(jnp.asarray(x, jnp.bfloat16), np.float64)  # noqa: E731
    rel_r = bf16(rng.normal(0, 1, (tokens, heads, 16)))
    rel_proj = bf16(rng.normal(0, 1, (16, extent)))
    sinks = bf16(rng.normal(0, 2, (tokens, heads)))
    args = [
        jnp.asarray(rel_r, jnp.bfloat16),
        jnp.asarray(rel_proj, jnp.bfloat16),
        jnp.asarray(positions, jnp.int32),
        jnp.asarray(sinks, jnp.bfloat16) if sink else None,
    ]
    bias, sink_tiles = run(*[None if a is None else jax.device_put(a, device) for a in args])
    want = reference(rel_r, rel_proj, positions, sm_scale)
    got = np.asarray(bias, np.float64)
    assert got.shape == want.shape
    # Within BF16 rounding of each value plus 2e-3 of the row's largest sum of |products| (a sum that cancels gets no
    # tighter bound than its terms).
    scale = reference(np.abs(rel_r), np.abs(rel_proj), positions, sm_scale).max(-1, keepdims=True)
    np.testing.assert_array_less(np.abs(got - want), np.abs(want) * 2.0**-7 + 2e-3 * scale + 1e-30)
    if sink:
        tiles = np.asarray(sink_tiles, np.float64)
        assert tiles.shape == (tokens, 1, heads, 32)
        assert (tiles[..., 1:] == 0).all()
        np.testing.assert_allclose(tiles[:, 0, :, 0], sinks / sm_scale, rtol=2.0**-8)
    else:
        assert sink_tiles is None
    irs = [path.read_text() for path in (tmp_path / "irs").glob("ttnn_*.mlir")]
    assert irs and any('"ttnn.window_bias"' in ir for ir in irs)
