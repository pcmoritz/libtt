"""SGLang-TorchTPU's attention hook writes every token's keys and values to its
page slot in the fused KV cache (sglang_tpu_kernels).

The hook updates the cache with TTNN shared paged updates of up to 32 tokens
each. Cover a decode batch (one token per sequence) and a chunk of one
sequence that spans two update calls and crosses a tile, so that tokens of one
sequence share tiles.
"""

import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax_plugins.libtt import sglang_tpu_kernels

KV_HEADS = 2
HEAD_DIM = 128
PAGE_SIZE = 64
PAGES = 33
PAGES_PER_SEQ = 8


@pytest.mark.parametrize(
    "q_lens,kv_lens",
    [
        ([1, 1, 1, 1], [5, 70, 130, 1]),  # decode batch
        ([40, 0], [50, 0]),  # 40 tokens of one sequence, positions 10 .. 49
    ],
)
def test_cache_update(q_lens, kv_lens):
    rng = np.random.default_rng(0)
    seqs, tokens = len(q_lens), sum(q_lens)
    dtype = jnp.bfloat16
    cache = np.asarray(rng.normal(size=(PAGES, KV_HEADS, PAGE_SIZE, 2 * HEAD_DIM)), dtype)
    queries = np.asarray(rng.normal(size=(tokens, KV_HEADS, HEAD_DIM)), dtype)
    keys = np.asarray(rng.normal(size=(tokens, KV_HEADS, HEAD_DIM)), dtype)
    values = np.asarray(rng.normal(size=(tokens, KV_HEADS, HEAD_DIM)), dtype)
    table = 1 + np.arange(seqs * PAGES_PER_SEQ, dtype=np.int32).reshape(seqs, PAGES_PER_SEQ)
    cu_q_lens = np.concatenate([[0], np.cumsum(q_lens)]).astype(np.int32)
    kv_lens = np.asarray(kv_lens, np.int32)
    num_seqs = sum(1 for q in q_lens if q)

    attention = sglang_tpu_kernels._ragged_paged_attention(types.SimpleNamespace(DEFAULT_MASK_VALUE=-1e30))

    def run(cache, queries, keys, values, kv_lens, page_indices, cu_q_lens):
        _, cache = attention(
            queries, keys, values, cache, kv_lens, page_indices, cu_q_lens, cu_q_lens,
            jnp.asarray([num_seqs, num_seqs, num_seqs], jnp.int32), None, sm_scale=HEAD_DIM ** -0.5,
        )
        return cache

    device = jax.devices("tt")[0]
    out = jax.jit(run, donate_argnums=0, compiler_options={"optimization_level": "O0"})(
        *[jax.device_put(x, device) for x in (cache, queries, keys, values, kv_lens, table.reshape(-1), cu_q_lens)]
    )

    expected = cache.copy()
    for s in range(seqs):
        for t in range(cu_q_lens[s], cu_q_lens[s + 1]):
            position = kv_lens[s] + t - cu_q_lens[s + 1]
            page, row = table[s, position // PAGE_SIZE], position % PAGE_SIZE
            expected[page, :, row, :HEAD_DIM] = keys[t]
            expected[page, :, row, HEAD_DIM:] = values[t]
    np.testing.assert_array_equal(np.asarray(out), expected)
