"""DFlash's draft KV write (sglang_tpu.kv_fused_rope_write) on the TT KV pool layout.

`_fused_rope_write_kv` rotates K (neox RoPE) and writes each draft layer's K and
V rows to their slots in `[pages, kv_heads, page_size, 2 * head_dim]` buffers,
spans of contiguous slots at a time, as SGLang-TorchTPU's DFlash draft does
after each extend and verify. Checked against NumPy, including padding spans,
spans that end a page, and heads and head dims smaller than the pool's.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax_plugins.libtt import sglang_tpu_kernels

PAGES, PAGE = 8, 64


def reference(bufs, k, v, cos, sin, spans):
    bufs = [b.astype(np.float32).copy() for b in bufs]
    half = k.shape[-1] // 2
    x1, x2 = k[..., :half], k[..., half:]
    rotated = np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)
    heads, head_dim = k.shape[2], k.shape[3]
    for dst, src, n in spans:
        for j in range(n):
            page, offset = divmod(dst + j, PAGE)
            for layer, buf in enumerate(bufs):
                # A written row is the token's K and V, zero-padded to the pool's.
                buf[page, :, offset, :] = 0
                buf[page, :heads, offset, :head_dim] = rotated[layer, src + j]
                row = buf.shape[-1] // 2
                buf[page, :heads, offset, row : row + head_dim] = v[layer, src + j]
    return bufs


@pytest.mark.parametrize(
    "heads, head_dim, cache_heads, cache_head_dim, spans",
    [
        # An extend: two requests' tokens, the second's crossing a page.
        (2, 128, 2, 128, [(70, 0, 10), (120, 10, 8), (128, 18, 6), (0, 0, 0)]),
        # A verify block per request, at the end of a page, and padding spans.
        (8, 128, 8, 128, [(60, 0, 4), (200, 4, 4), (0, 0, 0), (0, 0, 0)]),
        # Fewer heads and a narrower head than the pool.
        (1, 64, 2, 128, [(5, 0, 3), (300, 3, 5)]),
    ],
)
def test_draft_kv_write(heads, head_dim, cache_heads, cache_head_dim, spans):
    layers = 2
    tokens = max(src + n for _, src, n in spans) + 2  # trailing tokens no span writes
    rng = np.random.default_rng(0)
    k = rng.normal(0, 1, (layers, tokens, heads, head_dim)).astype(np.float32)
    v = rng.normal(0, 1, (layers, tokens, heads, head_dim)).astype(np.float32)
    angle = rng.uniform(0, 6.3, (tokens, 1, head_dim // 2)).astype(np.float32)
    cos, sin = np.cos(angle), np.sin(angle)
    bufs = [
        np.asarray(rng.normal(0, 1, (PAGES, cache_heads, PAGE, 2 * cache_head_dim)), jnp.bfloat16)
        for _ in range(layers)
    ]
    dst, src, lens = (np.array(x, np.int32) for x in zip(*spans))

    def write(dst, src, lens, k, v, cos, sin, *bufs):
        return tuple(sglang_tpu_kernels._fused_rope_write_kv(dst, src, lens, k, v, cos, sin, *bufs, max_span_len=16))

    device = jax.devices("tt")[0]
    got = jax.jit(write, donate_argnums=tuple(range(7, 7 + layers)))(
        *[jax.device_put(x, device) for x in (dst, src, lens, k, v, cos, sin, *bufs)]
    )
    want = reference([np.asarray(b) for b in bufs], k, v, cos, sin, spans)
    for g, w in zip(got, want):
        np.testing.assert_allclose(np.asarray(g).astype(np.float32), w, atol=0.05, rtol=0.02)
