"""SGLang-TorchTPU's attention hook runs a prefill on TTNN's chunked paged
attention, one sequence per user (sglang_tpu_kernels).

Each sequence's queries attend to the keys of its own pages up to their
position, also when the prefix before them doesn't start a chunk. Cover one
sequence after an unaligned prefix, two sequences after page-aligned prefixes,
grouped query heads, and tokens past the last sequence (padding).
"""

import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax_plugins.libtt import sglang_tpu_kernels

Q_HEADS = 4
KV_HEADS = 2
HEAD_DIM = 128
PAGE_SIZE = 64
PAGES = 33
PAGES_PER_SEQ = 8


@pytest.mark.parametrize(
    "q_lens,kv_lens,tokens,pages",
    [
        ([40, 0], [50, 0], 64, PAGES),  # positions 10 .. 49 of one sequence
        ([100, 30], [164, 30], 160, PAGES),  # after a page of prefix, and from 0
        ([64, 0, 0, 0], [320, 0, 0, 0], 64, PAGES),  # a chunk after four pages
        # A pool of only the sequence's pages: the padded page table is wider
        # than the pool, its padding entries all page 0.
        ([200], [200], 256, 1 + PAGES_PER_SEQ),
    ],
)
def test_prefill_attention(q_lens, kv_lens, tokens, pages, tmp_path):
    rng = np.random.default_rng(0)
    seqs = len(q_lens)
    scale = HEAD_DIM**-0.5
    dtype = jnp.bfloat16
    cache = np.asarray(rng.normal(size=(pages, KV_HEADS, PAGE_SIZE, 2 * HEAD_DIM)), dtype)
    queries = np.asarray(rng.normal(size=(tokens, Q_HEADS, HEAD_DIM)), dtype)
    keys = np.asarray(rng.normal(size=(tokens, KV_HEADS, HEAD_DIM)), dtype)
    values = np.asarray(rng.normal(size=(tokens, KV_HEADS, HEAD_DIM)), dtype)
    table = 1 + np.arange(seqs * PAGES_PER_SEQ, dtype=np.int32).reshape(seqs, PAGES_PER_SEQ)
    cu_q_lens = np.concatenate([[0], np.cumsum(q_lens)]).astype(np.int32)
    kv_lens = np.asarray(kv_lens, np.int32)
    num_seqs = sum(1 for q in q_lens if q)

    attention = sglang_tpu_kernels._ragged_paged_attention(types.SimpleNamespace(DEFAULT_MASK_VALUE=-1e30))

    def run(cache, queries, keys, values, kv_lens, page_indices, cu_q_lens):
        return attention(
            queries, keys, values, cache, kv_lens, page_indices, cu_q_lens, cu_q_lens,
            jnp.asarray([0, num_seqs, num_seqs], jnp.int32), None, sm_scale=scale,
        )

    device = jax.devices("tt")[0]
    options = {"optimization_level": "O0", "export_path": str(tmp_path)}
    out, _ = jax.jit(run, donate_argnums=0, compiler_options=options)(
        *[jax.device_put(x, device) for x in (cache, queries, keys, values, kv_lens, table.reshape(-1), cu_q_lens)]
    )
    out = np.asarray(out).astype(np.float32)
    ir = "".join(path.read_text() for path in (tmp_path / "irs").glob("ttnn_[0-9]*.mlir"))
    assert "ttnn.chunked_scaled_dot_product_attention" in ir

    # The cache after the update, as float32 keys and values per position.
    updated = cache.astype(np.float32)
    for s in range(seqs):
        for t in range(cu_q_lens[s], cu_q_lens[s + 1]):
            position = kv_lens[s] + t - cu_q_lens[s + 1]
            page, row = table[s, position // PAGE_SIZE], position % PAGE_SIZE
            updated[page, :, row, :HEAD_DIM] = keys[t].astype(np.float32)
            updated[page, :, row, HEAD_DIM:] = values[t].astype(np.float32)
    group = Q_HEADS // KV_HEADS
    for s in range(seqs):
        for t in range(cu_q_lens[s], cu_q_lens[s + 1]):
            position = kv_lens[s] + t - cu_q_lens[s + 1]
            rows = [updated[table[s, p // PAGE_SIZE], :, p % PAGE_SIZE] for p in range(position + 1)]
            k = np.stack(rows)[:, :, :HEAD_DIM]  # [keys, kv_heads, head_dim]
            v = np.stack(rows)[:, :, HEAD_DIM:]
            for h in range(Q_HEADS):
                scores = k[:, h // group] @ queries[t, h].astype(np.float32) * scale
                p = np.exp(scores - scores.max())
                expected = (p / p.sum()) @ v[:, h // group]
                np.testing.assert_allclose(out[t, h], expected, atol=3e-2, rtol=3e-2)
