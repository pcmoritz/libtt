"""Paged decode attention must match a reference for any split of the KV cache.

The compiler splits each KV head of each user across several cores, and the
kernel reads the cache in chunks sized to its length (32 to 256 tokens), so
cover positions inside, at and across chunk and page boundaries, more than one
user, and grouped query heads. The cache is separate K and V, or one cache of
[K | V] rows passed as both, as SGLang-TorchTPU's decode reads it.

A sliding window of W attends to the last W positions, the user's own
included, and an attention sink adds one logit per query head to the softmax's
denominator, unscaled (the kernel takes it divided by the scale), as GPT-OSS's
sliding-window layers do.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

HEAD_DIM = 128
BLOCK = 32
PAGES_PER_USER = 24


def reference(query, k_cache, v_cache, page_table, positions, window=None, sinks=None):
    _, users, q_heads, dim = query.shape
    kv_heads = k_cache.shape[1]
    out = np.zeros(query.shape, np.float32)
    for user in range(users):
        length = positions[user] + 1
        blocks = page_table[user]
        first = 0 if window is None else max(0, length - window)
        keys = k_cache[blocks].transpose(1, 0, 2, 3).reshape(kv_heads, -1, dim)[:, first:length]
        values = v_cache[blocks].transpose(1, 0, 2, 3).reshape(kv_heads, -1, dim)[:, first:length]
        for head in range(q_heads):
            kv = head // (q_heads // kv_heads)
            scores = keys[kv].astype(np.float32) @ query[0, user, head].astype(np.float32)
            scores /= np.sqrt(dim)
            top = scores.max() if sinks is None else max(scores.max(), sinks[head])
            weights = np.exp(scores - top)
            total = weights.sum() + (0 if sinks is None else np.exp(sinks[head] - top))
            out[0, user, head] = weights @ values[kv].astype(np.float32) / total
    return out


@pytest.mark.parametrize("kv_rows", [False, True], ids=["separate", "kv_rows"])
@pytest.mark.parametrize("trace", [False, True])
@pytest.mark.parametrize(
    "positions",
    [[4], [63], [64], [200], [255], [600], [5, 63, 130, 255], [40, 700]],
    ids=lambda p: "-".join(map(str, p)),
)
def test_paged_decode_attention(positions, trace, kv_rows):
    device = jax.devices("tt")[0]
    rng = np.random.default_rng(0)
    users = len(positions)
    q_heads, kv_heads = 8, 2
    blocks = users * PAGES_PER_USER + 3
    query = rng.normal(size=(1, users, q_heads, HEAD_DIM)).astype(jnp.bfloat16)
    k_cache = rng.normal(size=(blocks, kv_heads, BLOCK, HEAD_DIM)).astype(jnp.bfloat16)
    v_cache = rng.normal(size=(blocks, kv_heads, BLOCK, HEAD_DIM)).astype(jnp.bfloat16)
    # Scatter each user's pages over the cache.
    page_table = rng.permutation(blocks)[: users * PAGES_PER_USER].reshape(users, -1).astype(np.int32)
    positions = np.asarray(positions, np.int32)

    def attend(query, k_cache, v_cache, page_table, positions):
        return jax.ffi.ffi_call(
            "tt.paged_scaled_dot_product_attention_decode",
            jax.ShapeDtypeStruct(query.shape, query.dtype),
        )(
            query,
            k_cache,
            v_cache,
            page_table,
            positions,
            is_causal=True,
            has_attention_mask=False,
            has_cur_pos_tensor=True,
            has_attention_sink=False,
        )

    run = jax.jit(
        attend,
        compiler_options={"optimization_level": "O1", "enable_trace": str(trace).lower()},
    )
    if kv_rows:
        cache = np.concatenate([k_cache, v_cache], axis=-1)
        args = [jax.device_put(x, device) for x in (query, cache, cache, page_table, positions)]
    else:
        args = [jax.device_put(x, device) for x in (query, k_cache, v_cache, page_table, positions)]
    expected = reference(
        np.asarray(query, np.float32),
        np.asarray(k_cache, np.float32),
        np.asarray(v_cache, np.float32),
        page_table,
        positions,
    )
    for _ in range(2):
        out = np.asarray(run(*args), np.float32)
        # The kernel computes softmax and accumulates in BF16; with one core
        # per head and 32-token chunks it deviates by up to 0.075 here too.
        np.testing.assert_allclose(out, expected, rtol=0, atol=0.1)


@pytest.mark.parametrize("sinks", [False, True], ids=["no_sinks", "sinks"])
@pytest.mark.parametrize(
    "positions",
    [[100], [127], [128], [129], [140], [200], [255], [256], [600], [5, 130, 300, 701]],
    ids=lambda p: "-".join(map(str, p)),
)
def test_sliding_window(positions, sinks):
    """GPT-OSS's sliding-window layers: a window of 128 over a [K | V] row cache."""
    window = 128
    device = jax.devices("tt")[0]
    rng = np.random.default_rng(1)
    users = len(positions)
    q_heads, kv_heads = 8, 2
    blocks = users * PAGES_PER_USER + 3
    query = rng.normal(size=(1, users, q_heads, HEAD_DIM)).astype(jnp.bfloat16)
    cache = rng.normal(size=(blocks, kv_heads, BLOCK, 2 * HEAD_DIM)).astype(jnp.bfloat16)
    page_table = rng.permutation(blocks)[: users * PAGES_PER_USER].reshape(users, -1).astype(np.int32)
    positions = np.asarray(positions, np.int32)
    scale = HEAD_DIM**-0.5
    sink_logits = rng.normal(size=q_heads).astype(np.float32) * 2 if sinks else None

    def attend(query, cache, page_table, positions, *sink):
        return jax.ffi.ffi_call(
            "tt.paged_scaled_dot_product_attention_decode",
            jax.ShapeDtypeStruct(query.shape, query.dtype),
        )(
            query,
            cache,
            cache,
            page_table,
            positions,
            *sink,
            is_causal=True,
            has_attention_mask=False,
            has_cur_pos_tensor=True,
            has_attention_sink=bool(sink),
            scale=np.float32(scale),
            sliding_window_size=np.int32(window),
        )

    args = [jax.device_put(x, device) for x in (query, cache, page_table, positions)]
    if sinks:
        # The kernel's sink tile: each head's sink divided by the scale, in the first column.
        tile = np.zeros((q_heads, 32), np.float32)
        tile[:, 0] = sink_logits / scale
        args.append(jax.device_put(tile.astype(jnp.bfloat16), device))
    out = np.asarray(jax.jit(attend)(*args), np.float32)[..., :HEAD_DIM]
    k_cache = np.asarray(cache[..., :HEAD_DIM], np.float32)
    v_cache = np.asarray(cache[..., HEAD_DIM:], np.float32)
    # The sinks as the kernel reads them: BF16.
    sink_ref = None if sink_logits is None else np.asarray(
        (sink_logits / scale).astype(jnp.bfloat16), np.float32) * scale
    expected = reference(np.asarray(query, np.float32), k_cache, v_cache, page_table, positions, window, sink_ref)
    np.testing.assert_allclose(out, expected, rtol=0, atol=0.1)
