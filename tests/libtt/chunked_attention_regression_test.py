"""Chunked prefill attention over a paged KV cache.

TTNN's chunked attention required page-table rows of a multiple of 32 bytes.
An SGLang-JAX KV cache whose pages per request are not a multiple of 8
(Qwen3.8-27B with 200,000 tokens) failed to compile.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

PAGE = 32


def attend(query, key_cache, value_cache, page_table, chunk_start, scale):
    return jax.ffi.ffi_call(
        "tt.chunked_scaled_dot_product_attention",
        jax.ShapeDtypeStruct(query.shape, query.dtype),
        vmap_method="sequential",
    )(query, key_cache, value_cache, page_table, chunk_start, scale=np.float32(scale))


def reference(query, key_cache, value_cache, page_table, chunk_start, scale):
    users, heads, tokens, _ = query.shape
    kv_heads = key_cache.shape[1]
    out = np.zeros(query.shape, np.float32)
    for user in range(users):
        start = int(chunk_start[user if len(chunk_start) > 1 else 0])
        length = start + tokens
        pages = page_table[user, : -(-length // PAGE)]
        keys = key_cache[pages].transpose(1, 0, 2, 3).reshape(kv_heads, -1, query.shape[3])
        values = value_cache[pages].transpose(1, 0, 2, 3).reshape(kv_heads, -1, query.shape[3])
        for head in range(heads):
            kv = head // (heads // kv_heads)
            scores = query[user, head] @ keys[kv, :length].T * scale
            positions = np.arange(length)
            scores[positions[None, :] > start + np.arange(tokens)[:, None]] = -np.inf
            weights = np.exp(scores - scores.max(-1, keepdims=True))
            out[user, head] = weights / weights.sum(-1, keepdims=True) @ values[kv, :length]
    return out


@pytest.mark.parametrize("users", [1, 2])
def test_page_table_rows_of_unaligned_width(users):
    heads, kv_heads, dim, tokens, width = 4, 2, 64, 64, 13
    rng = np.random.default_rng(users)
    query = rng.standard_normal((users, heads, tokens, dim)).astype(np.float32)
    key_cache = rng.standard_normal((users * width + 1, kv_heads, PAGE, dim)).astype(np.float32)
    value_cache = rng.standard_normal(key_cache.shape).astype(np.float32)
    page_table = (1 + rng.permutation(users * width)).reshape(users, width).astype(np.int32)
    chunk_start = np.arange(users, dtype=np.int32) * PAGE
    scale = dim**-0.5

    as_bf16 = lambda x: jnp.asarray(x, jnp.bfloat16)  # noqa: E731
    out = jax.jit(attend, static_argnums=5)(
        as_bf16(query), as_bf16(key_cache), as_bf16(value_cache), page_table, chunk_start, scale
    )
    bf16 = lambda x: np.asarray(as_bf16(x), np.float32)  # noqa: E731
    expected = reference(
        bf16(query), bf16(key_cache), bf16(value_cache), page_table, chunk_start, scale
    )
    np.testing.assert_allclose(np.asarray(out, np.float32), expected, atol=3e-2, rtol=3e-2)
