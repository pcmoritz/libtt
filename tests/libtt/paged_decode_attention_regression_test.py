"""Paged decode attention must match a reference for any split of the KV cache.

The compiler splits each KV head of each user across several cores and reads
the cache in 64-token chunks, so cover positions inside, at and across chunk
and page boundaries, more than one user, and grouped query heads.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

HEAD_DIM = 128
BLOCK = 32
PAGES_PER_USER = 8


def reference(query, k_cache, v_cache, page_table, positions):
    _, users, q_heads, dim = query.shape
    kv_heads = k_cache.shape[1]
    out = np.zeros(query.shape, np.float32)
    for user in range(users):
        length = positions[user] + 1
        blocks = page_table[user]
        keys = k_cache[blocks].transpose(1, 0, 2, 3).reshape(kv_heads, -1, dim)[:, :length]
        values = v_cache[blocks].transpose(1, 0, 2, 3).reshape(kv_heads, -1, dim)[:, :length]
        for head in range(q_heads):
            kv = head // (q_heads // kv_heads)
            scores = keys[kv].astype(np.float32) @ query[0, user, head].astype(np.float32)
            scores /= np.sqrt(dim)
            weights = np.exp(scores - scores.max())
            weights /= weights.sum()
            out[0, user, head] = weights @ values[kv].astype(np.float32)
    return out


@pytest.mark.parametrize("trace", [False, True])
@pytest.mark.parametrize(
    "positions",
    [[4], [63], [64], [200], [255], [5, 63, 130, 255]],
    ids=lambda p: "-".join(map(str, p)),
)
def test_paged_decode_attention(positions, trace):
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
