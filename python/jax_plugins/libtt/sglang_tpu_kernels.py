"""SGLang-TorchTPU's Pallas attention kernel and KV cache layout for Tenstorrent.

SGLang-TorchTPU (`sglang_tpu`) exports JAX functions through torch_tpu, and
its attention kernel is a Pallas TPU kernel that only TPUs run. In TPU
compatibility mode, `tpu_compat` replaces it with the JAX function below when
`sglang_tpu` imports it. It takes the kernel's arguments and returns its
results, with static shapes, so a program compiles once per shape bucket.

The kernel's KV cache layout, `[pages, page_size, kv_heads * 2 / packing,
packing, head_dim]`, puts K and V of a head in the last two dimensions, which
Tenstorrent tiles pad from 2 rows to 32. The fused pool therefore allocates
`[pages, kv_heads, page_size, 2 * head_dim]` per layer instead, a head's key
and value side by side in one row, which is the layout of TTNN's paged cache
operations, and the only one the kernel reads. Each token's row is
written by TTNN's paged cache update, and a batch of no more
tokens than sequence slots, such as a decode batch, runs TTNN's paged decode
attention, which reads a cache twice as wide as the queries as [K | V] rows. A
prefill runs TTNN's chunked paged attention, one sequence per user, so it reads
only each sequence's pages: given the cache as both keys and values and the
queries padded with zeros, it computes softmax(q k) [k v], whose second half is
the output.

Hybrid models' gated DeltaNet kernel runs on libtt's recurrent kernels, with
the recurrent-state pool in their layout (`sglang_tpu_gdn`). Mixture-of-experts
models' fused MoE op runs on TTNN's fused MoE kernel, with the experts in BF16
or FP8, as loaded.
"""

# Users per TTNN paged decode attention call.
MAX_DECODE_USERS = 32
# Tokens that run as decode users when the decode kernel is the only TTNN kernel for the attention (a learned
# relative bias or per-token sinks): each reads its sequence up to its position, where gathering every sequence's
# pages for every token took ~5 ms per layer for a 64-token prefill.
MAX_DECODE_ROUTE_TOKENS = 256
# Tokens per TTNN paged cache update, one core each.
MAX_UPDATE_USERS = 32

RPA_V3_MODULE = "sglang_tpu.attention.pallas.ragged_paged_attention_v3"
KV_POOL_MODULE = "sglang_tpu.kv_pool"
FP8_LINEAR_SMALL_M_MODULE = "sglang_tpu.kernels.fp8_linear_small_m"
GDN_MODULE = "sglang_tpu.attention.pallas.gdn"
MAMBA_POOL_MODULE = "sglang_tpu.mamba_pool"
MAMBA_STATE_ROWS_MODULE = "sglang_tpu.mamba_state_rows"
INKLING_SCONV_MODULE = "sglang_tpu.attention_inkling_sconv_pallas"
KV_FUSED_ROPE_WRITE_MODULE = "sglang_tpu.kv_fused_rope_write"
BLOCK_DRAFT_WORKER_MODULE = "sglang_tpu.experimental.speculative.block_draft_worker"
ALL_GATHER_VIA_ALL_REDUCE_MODULE = "sglang_tpu.all_gather_via_all_reduce"
FUSED_MOE_MODULE = "sglang_tpu.moe.pallas.fused_moe_op"
FP8_MOE_FUSED_RS_MODULE = "sglang_tpu.fp8_moe_fused_rs"
FP4_MOE_MODULE = "sglang_tpu.fp4_moe"
INKLING2_ATTN_RES_MODULE = "sglang_tpu.inkling2_attn_res"
INKLING2_L2_CLIP_MODULE = "sglang_tpu.inkling2_l2_clip"
INKLING2_ROUTE_MODULE = "sglang_tpu.inkling2_route"


def _ragged_paged_attention(module):
    """`ragged_paged_attention` of the RPA v3 module, without Pallas."""
    import jax
    import jax.numpy as jnp

    def ragged_paged_attention(
        queries,  # [tokens, q_heads, head_dim]
        keys,  # [tokens, kv_heads, head_dim]
        values,  # [tokens, kv_heads, head_dim]
        kv_cache_fused,  # see the module docstring
        kv_lens,  # i32[seqs]
        page_indices,  # i32[seqs * pages_per_seq]
        cu_q_lens,  # i32[seqs + 1]
        cu_kv_lens,  # i32[seqs + 1]
        distribution,  # i32[3]: decode, prefill and mixed sequence ranges
        custom_mask,
        attention_sink=None,
        rel_r=None,  # [tokens, q_heads, rel_dim]
        rel_proj=None,  # [rel_dim, extent]
        token_sink=None,  # [tokens, q_heads]
        *,
        causal=1,
        sm_scale=1.0,
        sliding_window=None,
        soft_cap=None,
        mask_value=None,
        q_scale=None,
        k_scale=None,
        v_scale=None,
        xai_temperature_len=None,
        out_dtype=None,
        **_,
    ):
        if custom_mask is not None or xai_temperature_len is not None:
            raise NotImplementedError(
                "custom_mask and xai_temperature_len need the Pallas kernel"
            )
        if mask_value is None:
            mask_value = module.DEFAULT_MASK_VALUE
        tokens, q_heads, head_dim = queries.shape
        kv_heads = keys.shape[1]
        group = q_heads // kv_heads
        cache_shape = kv_cache_fused.shape
        if len(cache_shape) != 4:
            raise NotImplementedError(
                f"KV cache of shape {cache_shape}: only the fused pool's "
                "[pages, kv_heads, page_size, 2 * head_dim] layout (patch_kv_pool)"
            )
        pages, cache_heads, page_size, row = cache_shape
        cache_head_dim = row // 2
        seqs = kv_lens.shape[0]
        pages_per_seq = page_indices.shape[0] // seqs
        num_seqs = distribution[2]

        # Each token's sequence and position. Sequences past num_seqs are
        # padding; so are tokens past the last sequence's queries.
        token = jnp.arange(tokens, dtype=jnp.int32)
        live_seq = jnp.arange(seqs, dtype=jnp.int32) < num_seqs
        ends = (token[:, None] >= cu_q_lens[None, 1:]) & live_seq[None, :]
        seq = jnp.minimum(jnp.sum(ends.astype(jnp.int32), axis=1), seqs - 1)
        # A gather rather than a dynamic slice: TT reads slice bounds on the
        # host, which a traced program can't.
        q_end = cu_q_lens[seq + 1]
        position = kv_lens[seq] + token - q_end
        live = (seq < num_seqs) & (token < q_end) & (position >= 0)
        table = page_indices.reshape(seqs, pages_per_seq)

        # Write each token's keys and values to its page slot, then read every
        # sequence's pages as [seqs, pages_per_seq, kv_heads, page_size, head_dim].
        pad = ((0, 0), (0, cache_heads - kv_heads), (0, cache_head_dim - head_dim))
        new_kv = jnp.concatenate([jnp.pad(keys, pad), jnp.pad(values, pad)], axis=-1)
        new_kv = new_kv.astype(kv_cache_fused.dtype)
        # Page 0 is SGLang's padding page, so a sequence holds at most pages - 1
        # pages, and TTNN's paged operations take a page table narrower than the
        # pool.
        user_pages = table[seq, : min(pages_per_seq, pages - 1)]
        user_positions = jnp.where(live, position, -1)
        # TTNN shared paged updates of up to MAX_UPDATE_USERS tokens, in
        # place, which write each tile once with the rows of every token in
        # it, as if the tokens were written in order, so tokens of one
        # sequence may share a tile. A decode batch's keys and values go to
        # one update, which the compiler's fused attention prologue writes
        # directly. A scatter instead tilized and copied the layer's whole
        # cache on every call.
        cache = kv_cache_fused
        for start in range(0, tokens, MAX_UPDATE_USERS):
            end = min(start + MAX_UPDATE_USERS, tokens)
            cache = jax.ffi.ffi_call(
                "tt.paged_update_cache",
                jax.ShapeDtypeStruct(cache_shape, kv_cache_fused.dtype),
                input_output_aliases={0: 0},
            )(
                cache,
                new_kv[None, start:end],
                user_positions[start:end],
                user_pages[start:end],
                share_cache=True,
            )
        # Each token as a decode user reads its sequence's pages once, which
        # beats gathering them below when there are few tokens per sequence,
        # and beats the chunked kernel below for a few tokens in all, as in a
        # speculative verify: each token attends up to its own position, so
        # a sequence's window stays causal, where the chunked kernel pads
        # every sequence's queries to a chunk of 128 rows.
        decode_only = rel_proj is not None or token_sink is not None
        if (
            tokens <= max(seqs, MAX_DECODE_ROUTE_TOKENS if decode_only else MAX_DECODE_USERS)
            and cache_heads == kv_heads
            and causal
            and soft_cap is None
            and attention_sink is None
            and (rel_proj is None) == (rel_r is None)
            and q_scale is None
            and k_scale is None
            and v_scale is None
        ):
            out = _paged_decode(
                queries,
                cache,
                user_pages,
                user_positions,
                sm_scale,
                sliding_window,
                rel_r=rel_r,
                rel_proj=rel_proj,
                token_sink=token_sink,
            )
            return out.astype(out_dtype or queries.dtype), cache
        # Otherwise each sequence is a user of TTNN's chunked paged attention,
        # which reads only its pages, where gathering every sequence's pages
        # below computes scores against all of them.
        q_lens = cu_q_lens[1:] - cu_q_lens[:-1]
        if (
            cache_heads == kv_heads
            and causal
            and soft_cap is None
            and attention_sink is None
            and rel_proj is None
            and token_sink is None
            and sliding_window is None
            and q_scale is None
            and k_scale is None
            and v_scale is None
        ):
            out = _paged_prefill(
                queries,
                cache,
                table,
                cu_q_lens[:-1],
                jnp.where(live_seq, kv_lens - q_lens, 0),
                seq,
                position,
                live,
                sm_scale,
            )
            return out.astype(out_dtype or queries.dtype), cache
        context = cache[table]
        k = context[:, :, :kv_heads, :, :head_dim]
        v = context[:, :, :kv_heads, :, cache_head_dim : cache_head_dim + head_dim]
        # Attention as batched matmuls over [kv_heads, group * tokens] queries
        # and [kv_heads, keys] keys, group-major, so that every reshape keeps
        # the tiled token and key dimensions whole.
        keys_per_seq = pages_per_seq * page_size
        k = k.transpose(2, 0, 1, 3, 4).reshape(kv_heads, seqs * keys_per_seq, head_dim)
        v = v.transpose(2, 0, 1, 3, 4).reshape(kv_heads, seqs * keys_per_seq, head_dim)
        if k_scale is not None:
            k = k.astype(jnp.float32) * k_scale
        if v_scale is not None:
            v = v.astype(jnp.float32) * v_scale
        k, v = k.astype(queries.dtype), v.astype(queries.dtype)
        q = queries.reshape(tokens, kv_heads, group, head_dim).transpose(1, 2, 0, 3)
        q = q.reshape(kv_heads, group * tokens, head_dim)
        if q_scale is not None:
            q = (q.astype(jnp.float32) * q_scale).astype(queries.dtype)
        scores = jnp.einsum("kqd,knd->kqn", q, k, preferred_element_type=jnp.float32)
        scores = scores * sm_scale
        if soft_cap is not None:
            scores = soft_cap * jnp.tanh(scores / soft_cap)

        # A token attends to its own sequence: the keys up to its position, or
        # all of the sequence's keys without the causal mask.
        key = jnp.arange(seqs * keys_per_seq, dtype=jnp.int32)
        key_seq, key_pos = key // keys_per_seq, key % keys_per_seq
        allowed = (key_seq[None, :] == seq[:, None]) & live[:, None]
        if causal:
            allowed = allowed & (key_pos[None, :] <= position[:, None])
        else:
            allowed = allowed & (key_pos[None, :] < kv_lens[key_seq][None, :])
        if sliding_window is not None:
            allowed = allowed & (position[:, None] - key_pos[None, :] < sliding_window)
        scores = scores.reshape(kv_heads, group, tokens, -1)
        if rel_proj is not None:
            scores = scores + _relative_bias(rel_r, rel_proj, position, key_pos).reshape(
                kv_heads, group, tokens, -1
            )
        scores = jnp.where(allowed[None, None], scores, mask_value)
        if token_sink is not None:
            sink = token_sink.astype(jnp.float32).T.reshape(kv_heads, group, tokens, 1)
            probs = jax.nn.softmax(jnp.concatenate([sink, scores], axis=-1), axis=-1)[..., 1:]
        elif attention_sink is not None:
            sink = jnp.broadcast_to(
                jnp.asarray(attention_sink, jnp.float32).reshape(kv_heads, group, 1, 1),
                (kv_heads, group, tokens, 1),
            )
            probs = jax.nn.softmax(jnp.concatenate([sink, scores], axis=-1), axis=-1)[..., 1:]
        else:
            probs = jax.nn.softmax(scores, axis=-1)
        probs = probs.reshape(kv_heads, group * tokens, -1)
        out = jnp.einsum("kqn,knd->kqd", probs.astype(v.dtype), v, preferred_element_type=jnp.float32)
        out = out.reshape(kv_heads, group, tokens, head_dim).transpose(2, 0, 1, 3)
        out = out.reshape(tokens, q_heads, head_dim).astype(out_dtype or queries.dtype)
        return out, cache

    return ragged_paged_attention


def _relative_bias(rel_r, rel_proj, positions, key_positions):
    """[q_heads, tokens, keys] learned relative-position logits.

    A token at position p and a key at position j get rel_r[token] .
    rel_proj[:, p - j] while 0 <= p - j < extent, and nothing beyond it. Each
    (token, key) pair gathers its column of rel_proj as a row of rel_proj.T, an
    embedding lookup, and a batched matmul over the tokens takes the products.
    """
    import jax.numpy as jnp

    extent = rel_proj.shape[1]
    distance = positions[:, None] - key_positions[None, :]
    columns = jnp.take(
        rel_proj.astype(jnp.float32).T, jnp.clip(distance, 0, extent - 1), axis=0
    )
    bias = jnp.einsum(
        "thf,tkf->htk",
        rel_r.astype(jnp.float32),
        columns,
        preferred_element_type=jnp.float32,
    )
    return jnp.where(((distance >= 0) & (distance < extent))[None], bias, 0.0)


def _prefill_chunk(page_size):
    """TTNN's chunked attention chunk, the multiple its starts must be."""
    return min(page_size, 128)


def _prefill_rows(tokens, page_size):
    """Query rows per sequence: its tokens at any offset below one chunk."""
    chunk = _prefill_chunk(page_size)
    return -(-tokens // chunk) * chunk + chunk


def _prefill_pad_pages(tokens, page_size):
    """Padding pages after a page table row, for keys up to start + rows."""
    return -(-_prefill_rows(tokens, page_size) // page_size)


def _paged_prefill(queries, cache, table, q_starts, prefixes, seq, positions, live, sm_scale):
    """Each sequence's queries as one user of TTNN's chunked paged attention.

    The kernel's queries start at a chunk-aligned position, so a sequence's
    rows start at its prefix rounded down to a chunk, with its queries at their
    offset from there: each attends to the keys up to its own position. The
    other rows are padding, whose outputs are dropped.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np

    tokens, q_heads, head_dim = queries.shape
    page_size, row = cache.shape[2], cache.shape[3]
    seqs = table.shape[0]
    rows = _prefill_rows(tokens, page_size)
    starts = prefixes - prefixes % _prefill_chunk(page_size)
    offsets = prefixes - starts

    q = jnp.pad(queries, ((0, 0), (0, 0), (0, row - head_dim))).astype(cache.dtype)
    sources = q_starts[:, None] + jnp.arange(rows, dtype=jnp.int32)[None, :] - offsets[:, None]
    q = q[jnp.clip(sources, 0, tokens - 1).reshape(-1)]
    q = q.reshape(seqs, rows, q_heads, row).transpose(0, 2, 1, 3)
    # Keys up to start + rows: the sequence's pages, then padding page 0.
    page_table = jnp.pad(table, ((0, 0), (0, _prefill_pad_pages(tokens, page_size))))
    out = jax.ffi.ffi_call(
        "tt.chunked_scaled_dot_product_attention",
        jax.ShapeDtypeStruct(q.shape, q.dtype),
    )(q, cache, cache, page_table, starts, scale=np.float32(sm_scale))
    out = out.transpose(0, 2, 1, 3).reshape(seqs * rows, q_heads, row)
    out = out[jnp.where(live, seq * rows + positions - starts[seq], 0)]
    cache_head_dim = row // 2
    return out[:, :, cache_head_dim : cache_head_dim + head_dim]


def _paged_decode(
    queries,
    cache,
    page_table,
    positions,
    sm_scale,
    sliding_window,
    rel_r=None,
    rel_proj=None,
    token_sink=None,
):
    """Each token as a TTNN decode user, attending to its sequence up to its position.

    A learned relative-position bias (rel_r . rel_proj[:, distance]) goes to the kernel as its causal window bias,
    and per-token sinks as its sink per user and head. The kernel scales scores inside its exponent, the bias and
    sinks with them, where these are added to scaled scores: both are divided by sm_scale.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np

    tokens, q_heads, head_dim = queries.shape
    # The cache's rows are [K | V], each cache_head_dim wide: TTNN reads them as keys and values for queries that
    # wide.
    cache_head_dim = cache.shape[-1] // 2
    q = jnp.pad(queries, ((0, 0), (0, 0), (0, cache_head_dim - head_dim))).astype(cache.dtype)
    window = {} if sliding_window is None else {"sliding_window_size": np.int32(sliding_window)}
    # tt.window_bias forms each chunk's bias (and sink tiles) in one program from a table computed once.
    fused_bias = rel_proj is not None and rel_proj.shape[0] == 16 and q_heads <= 32 and MAX_DECODE_USERS <= 32
    table = _window_bias_table(rel_proj, sm_scale) if fused_bias else None
    bias = None if rel_proj is None or fused_bias else _decode_window_bias(rel_r, rel_proj, positions, sm_scale)
    sink = None
    if token_sink is not None and not fused_bias:
        # [users, 1, heads, 32] with each sink in column 0, as TTNN's per-head sink tile.
        sink = token_sink.astype(jnp.bfloat16) * jnp.bfloat16(1 / sm_scale)
        sink = jnp.pad(sink[:, None, :, None], ((0, 0), (0, 0), (0, 0), (0, 31)))
    outputs = []
    for start in range(0, tokens, MAX_DECODE_USERS):
        chunk = slice(start, start + MAX_DECODE_USERS)
        users = q[None, chunk]
        operands = [users, cache, cache, page_table[chunk]]
        chunk_bias = None if bias is None else bias[chunk]
        chunk_sink = None if sink is None else sink[chunk]
        if fused_bias:
            chunk_bias, chunk_sink = _window_bias_call(
                rel_r[chunk], table, positions[chunk], None if token_sink is None else token_sink[chunk], sm_scale
            )
        if chunk_bias is not None:
            operands.append(chunk_bias)
        operands.append(positions[chunk])
        if chunk_sink is not None:
            operands.append(chunk_sink)
        outputs.append(
            jax.ffi.ffi_call(
                "tt.paged_scaled_dot_product_attention_decode",
                jax.ShapeDtypeStruct(users.shape, users.dtype),
            )(
                *operands,
                is_causal=True,
                has_attention_mask=chunk_bias is not None,
                has_cur_pos_tensor=True,
                has_attention_sink=chunk_sink is not None,
                scale=np.float32(sm_scale),
                **window,
            )
        )
    return jnp.concatenate(outputs, axis=1)[0, :, :, :head_dim]


def _decode_window_bias(rel_r, rel_proj, positions, sm_scale):
    """[tokens, 1, q_heads, width] BF16 window bias of TTNN's causal decode.

    The kernel's window is width / 32 key tiles ending at the tile of each user's position p: column j is the key at
    first + j, first = 32 * max(0, p // 32 + 1 - width / 32), whose bias is rel_r . rel_proj[:, p - first - j] for
    distances in [0, extent). width covers the extent behind any position in a tile. Each (token, column) gathers its
    column of rel_proj as a row of rel_proj.T, which has a row of zeros appended for the other columns, and a matmul
    per token takes the products. The kernel scales the bias with the scores, so it is divided by sm_scale.
    """
    import jax.numpy as jnp

    # rel_proj is a weight: what follows from it alone (the scaled, transposed and padded table) runs once.
    rel_proj = _mark_weight(rel_proj)
    extent = rel_proj.shape[1]
    window_tiles = -(-extent // 32) + 1
    width = 32 * window_tiles
    first = 32 * jnp.maximum(positions // 32 + 1 - window_tiles, 0)
    distance = (positions - first)[:, None] - jnp.arange(width, dtype=jnp.int32)[None, :]
    rows = jnp.where((distance >= 0) & (distance < extent), distance, extent)
    table = jnp.pad(rel_proj.astype(jnp.float32).T / sm_scale, ((0, 1), (0, 0)))
    columns = table.at[rows].get(mode="promise_in_bounds")
    bias = jnp.einsum(
        "thf,twf->thw", rel_r.astype(jnp.float32), columns, preferred_element_type=jnp.float32
    )
    return bias.astype(jnp.bfloat16)[:, None]


def _window_bias_table(rel_proj, sm_scale):
    """[2W, rel_dim] FP32 for tt.window_bias: row W - 1 - d holds distance d's column of rel_proj / sm_scale, zero
    where d is outside [0, extent). A key tile's 32 window columns are then 32 consecutive rows. W is
    `_decode_window_bias`'s width; the table depends on the weight alone, so it is computed once."""
    import jax.numpy as jnp

    rel_proj = _mark_weight(rel_proj)
    extent = rel_proj.shape[1]
    width = 32 * (-(-extent // 32) + 1)
    distance = width - 1 - jnp.arange(2 * width, dtype=jnp.int32)
    valid = (distance >= 0) & (distance < extent)
    rows = (rel_proj.astype(jnp.float32).T / sm_scale)[jnp.clip(distance, 0, extent - 1)]
    return jnp.where(valid[:, None], rows, jnp.float32(0))


def _window_bias_call(rel_r, table, positions, sink, sm_scale):
    """tt.window_bias for up to 32 tokens: `_decode_window_bias`'s bias [tokens, 1, q_heads, W] from rel_r
    [tokens, q_heads, 16] and `_window_bias_table`'s table and, with sink [tokens, q_heads], the kernel's sink tiles
    [tokens, 1, q_heads, 32] (sink / sm_scale in column 0), in one program."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    tokens, heads, rel_dim = rel_r.shape
    width = table.shape[0] // 2
    results = [jax.ShapeDtypeStruct((tokens, 1, heads, width), jnp.bfloat16)]
    operands = [rel_r.reshape(tokens, heads * rel_dim).astype(jnp.bfloat16), table, positions.astype(jnp.int32)]
    if sink is not None:
        operands.append(sink.astype(jnp.bfloat16))
        results.append(jax.ShapeDtypeStruct((tokens, 1, heads, 32), jnp.bfloat16))
    outs = jax.ffi.ffi_call("tt.window_bias", tuple(results))(
        *operands, heads=np.int64(heads), sink_scale=np.float32(1 / sm_scale), has_sink=sink is not None
    )
    return outs[0], (outs[1] if sink is not None else None)


def patch_rpa_v3(module):
    module.ragged_paged_attention = _ragged_paged_attention(module)


def patch_kv_pool(module):
    """Gives the fused MHA pool the `[pages, kv_heads, page_size, 2 * head_dim]` layout."""
    pool = module.TpuFusedMHATokenToKVPool

    def _create_buffers(self):
        shape = (
            self.num_blocks,
            self.num_kv_heads_x2 // 2,
            self.page_size,
            2 * self.padded_head_dim,
        )
        module.logger.info(
            "Allocating TT KV buffers: layers=%d, shape per layer=%s, dtype=%s",
            self.layer_num,
            shape,
            self.store_dtype,
        )
        self.kv_buffer = [
            module._zeros_kv_buffer(shape, self.store_dtype, self.device)
            for _ in range(self.layer_num)
        ]
        self.data_ptrs = module.torch.tensor(
            [x.data_ptr() for x in self.kv_buffer], dtype=module.torch.uint64, device=self.device
        )
        self.data_strides = module.torch.tensor(
            [x[0].nbytes for x in self.kv_buffer], device=self.device
        )

    def _logical_kv_view(self, layer_id):
        """[tokens, 2 * heads, head_dim], K and V of a head interleaved, as the pool's
        own getters read it; a copy, since tokens aren't contiguous here."""
        buf = self.kv_buffer[layer_id - self.start_layer]
        n_blocks, heads, page_size, row = buf.shape
        return buf.transpose(1, 2).reshape(n_blocks * page_size, 2 * heads, row // 2)

    pool._create_buffers = _create_buffers
    pool._logical_kv_view = _logical_kv_view
    pool.KV_LAYOUT_NAME = "tt-head-major"


def _quantize_rows(x, dtype, min_scale):
    """Per row of x ([M, K]): scale = max(amax(|x|) / fp8_max, min_scale) and
    x / scale cast to float8 `dtype` (not saturated). Up to one tile row of
    whole tiles runs as one TTNN program (libtt's quantize_rows); otherwise the
    same math as separate ops."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    tokens, k = x.shape
    if tokens <= 32 and k % 32 == 0 and x.dtype in (jnp.bfloat16, jnp.float32):
        return jax.ffi.ffi_call(
            "tt.quantize_rows",
            (
                jax.ShapeDtypeStruct(x.shape, dtype),
                jax.ShapeDtypeStruct((tokens, 1), jnp.float32),
            ),
        )(x, min_scale=np.float32(min_scale), e5m2=np.bool_(dtype == jnp.float8_e5m2))
    fp8_max = float(jnp.finfo(dtype).max)
    scale = jnp.maximum(
        jnp.max(jnp.abs(x), axis=-1, keepdims=True).astype(jnp.float32) / fp8_max, min_scale
    )
    return (x.astype(jnp.float32) / scale).astype(dtype), scale


def _fp8_linear_small_m(x, weight, weight_scale):
    """`fp8_linear_small_m` without Pallas, with the kernel's math: a per-row
    scale amax / fp8_max floored at 1e-12, an unsaturated FP8 cast, an
    f32-accumulated FP8 dot, then (acc * x_scale) * w_scale."""
    import jax
    import jax.numpy as jnp

    x_q, x_scale = _quantize_rows(x, weight.dtype, 1e-12)
    acc = jax.lax.dot_general(
        x_q, weight, (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32
    )
    # A per-tensor scale stays a scalar: broadcast to [1, n], it costs a
    # repeat program and a full-width multiply.
    w_scale = weight_scale.astype(jnp.float32).reshape(1, -1)
    return (acc * x_scale * w_scale).astype(x.dtype)


def patch_fp8_linear_small_m(module):
    module.fp8_linear_small_m = _fp8_linear_small_m


def patch_gdn(module):
    from jax_plugins.libtt import sglang_tpu_gdn

    module.fused_conv1d_gdn = sglang_tpu_gdn.fused_conv1d_gdn


def patch_mamba_state_rows(module):
    """Row reads and writes of a state pool (Inkling's short-convolution windows)
    without Pallas: a gather of the rows, and TTNN's in-place state-pool update,
    which writes whole slots and skips slot 0, SGLang's padding slot, where the
    Pallas kernel skips the rows that aren't live."""
    import functools

    @functools.cache
    def state_row_read_op():
        import jax
        import jax.numpy as jnp
        import torch
        from torch_tpu._internal import pallas

        # torch_tpu's jax_op reads the signature's annotations.
        def read_rows(pool: jax.Array, rows: jax.Array) -> tuple[jax.Array, jax.Array]:
            # The callers clamp the rows to the pool.
            flat = pool.reshape(pool.shape[0], -1)
            out = flat.at[rows].get(mode="promise_in_bounds")
            return pool, out.reshape(rows.shape[0], *pool.shape[1:])

        op = pallas.jax_op("sglang::tpu_state_row_read", read_rows, donate_argnums=(0,))

        @op.register_fake
        def fake(pool, rows):
            return torch.empty_like(pool), pool.new_empty((rows.shape[0], *pool.shape[1:]))

        return op

    @functools.cache
    def state_row_write_op():
        import jax
        import jax.numpy as jnp
        import torch
        from torch_tpu._internal import pallas

        from jax_plugins.libtt import sglang_tpu_gdn

        def write_rows(pool: jax.Array, rows: jax.Array, live: jax.Array, new: jax.Array) -> jax.Array:
            return sglang_tpu_gdn._write_rows(pool, jnp.where(live != 0, rows, 0), new)

        op = pallas.jax_op("sglang::tpu_state_row_write", write_rows, donate_argnums=(0,))

        @op.register_fake
        def fake(pool, rows, live, new):
            return torch.empty_like(pool)

        return op

    module.state_row_read_op = state_row_read_op
    module.state_row_write_op = state_row_write_op


def patch_inkling_sconv(module):
    """Inkling's short convolutions: a decode step runs on TTNN's causal conv
    update, without its SiLU, which reads and updates each request's window in
    the pool in place (`[slots, channels, 3]`, see
    `sglang_tpu_gdn.patch_mamba_pool`). With the model's decode weight the
    residual and the scale are folded in; global layers' K and V add each row's
    dynamic taps (per head) in the kernel, which scales the result. The rest
    (extends and verifies) convolve in torch, as in SGLang-TorchTPU."""
    import functools

    import torch

    backend = module.TpuInklingSconvBackend
    torch_decode = backend._xla_sconv_decode

    @functools.cache
    def conv_update_op(out_dtype="float32"):
        import jax
        import jax.numpy as jnp
        from torch_tpu._internal import pallas

        from jax_plugins.libtt import sglang_tpu_gdn

        def conv_update(
            state: jax.Array, x: jax.Array, weight: jax.Array, slots: jax.Array, initial: jax.Array
        ) -> tuple[jax.Array, jax.Array]:
            outputs = []
            for start in range(0, x.shape[0], sglang_tpu_gdn.MAX_ROWS):
                rows = slice(start, start + sglang_tpu_gdn.MAX_ROWS)
                state, out = jax.ffi.ffi_call(
                    "tt.causal_conv1d_update",
                    (
                        jax.ShapeDtypeStruct(state.shape, state.dtype),
                        jax.ShapeDtypeStruct(x[rows].shape, jnp.dtype(out_dtype)),
                    ),
                    input_output_aliases={0: 0},
                    vmap_method="sequential",
                )(state, x[rows], weight, slots[rows], initial[rows], apply_silu=False)
                outputs.append(out)
            return state, jnp.concatenate(outputs)

        op = pallas.jax_op(f"libtt::inkling_conv_update_{out_dtype}", conv_update, donate_argnums=(0,))

        @op.register_fake
        def fake(state, x, weight, slots, initial):
            return torch.empty_like(state), x.new_empty(x.shape, dtype=getattr(torch, out_dtype))

        return op

    @functools.cache
    def dynamic_conv_update_op(head_dim, kernel_size):
        """conv_update_op's BF16 form with each row's dynamic taps ([rows, heads * kernel_size], lag order) added to
        the weight's and the result scaled by (2 * kernel_size + 1)^-0.5, the model's scale with dynamic weights."""
        import jax
        import jax.numpy as jnp
        import numpy as np
        from torch_tpu._internal import pallas

        from jax_plugins.libtt import sglang_tpu_gdn

        output_scale = (2 * kernel_size + 1) ** -0.5

        def conv_update(
            state: jax.Array, x: jax.Array, weight: jax.Array, slots: jax.Array, initial: jax.Array, dyn: jax.Array
        ) -> tuple[jax.Array, jax.Array]:
            outputs = []
            for start in range(0, x.shape[0], sglang_tpu_gdn.MAX_ROWS):
                rows = slice(start, start + sglang_tpu_gdn.MAX_ROWS)
                state, out = jax.ffi.ffi_call(
                    "tt.causal_conv1d_update",
                    (
                        jax.ShapeDtypeStruct(state.shape, state.dtype),
                        jax.ShapeDtypeStruct(x[rows].shape, jnp.bfloat16),
                    ),
                    input_output_aliases={0: 0},
                    vmap_method="sequential",
                )(
                    state,
                    x[rows],
                    weight,
                    slots[rows],
                    initial[rows],
                    dyn[rows],
                    apply_silu=False,
                    has_dynamic_weight=True,
                    head_dim=np.int64(head_dim),
                    output_scale=np.float32(output_scale),
                )
                outputs.append(out)
            return state, jnp.concatenate(outputs)

        op = pallas.jax_op(
            f"libtt::inkling_dynamic_conv_update_{head_dim}_{kernel_size}", conv_update, donate_argnums=(0,)
        )

        @op.register_fake
        def fake(state, x, weight, slots, initial, dyn):
            return torch.empty_like(state), x.new_empty(x.shape, dtype=torch.bfloat16)

        return op

    def _xla_sconv_decode(self, x, *, state, weight, dyn, use_residual, decode_weight=None):
        # Dynamic taps need the model's decode weight (static taps in tap order with the residual), at most 8 heads
        # of whole tiles.
        dynamic_ok = (
            dyn is not None
            and decode_weight is not None
            and use_residual
            and dyn.shape[1] * dyn.shape[2] <= 32
            and dyn.shape[2] == 4
            and (x.shape[1] // dyn.shape[1]) % 32 == 0
        )
        if (dyn is not None and not dynamic_ok) or state.dim() != 3 or self._track_end is not None:
            return torch_decode(
                self, x, state=state, weight=weight, dyn=dyn, use_residual=use_residual, decode_weight=decode_weight
            )
        num_seqs, dim = x.shape
        # Padding rows (index -1) go to slot 0, which the kernel never writes.
        slots = torch.where(self._state_indices[:num_seqs] >= 0, self._state_indices[:num_seqs], 0)
        initial = (self._seq_lens[:num_seqs] > 1).to(torch.bfloat16)
        if dyn is not None:
            heads, kernel_size = dyn.shape[1], dyn.shape[2]
            pool, conv = dynamic_conv_update_op(dim // heads, kernel_size)(
                state,
                x.to(torch.bfloat16),
                decode_weight.to(torch.bfloat16),
                slots.to(torch.int32),
                initial,
                dyn.reshape(num_seqs, heads * kernel_size).to(torch.bfloat16),
            )
            state.copy_(pool)
            return conv.to(x.dtype)
        kernel_size = weight.shape[-1]
        if decode_weight is not None and use_residual:
            # The model's weight with the residual tap and the scale folded in: the
            # kernel's output is the result.
            pool, conv = conv_update_op("bfloat16")(
                state, x.to(torch.bfloat16), decode_weight.to(torch.bfloat16), slots.to(torch.int32), initial
            )
            state.copy_(pool)
            return conv.to(x.dtype)
        pool, conv = conv_update_op()(
            state, x.to(torch.bfloat16), weight.reshape(dim, kernel_size), slots.to(torch.int32), initial
        )
        state.copy_(pool)
        acc = conv + x.float() if use_residual else conv
        return (acc * (kernel_size + int(use_residual)) ** -0.5).to(x.dtype)

    backend._xla_sconv_decode = _xla_sconv_decode
    module.libtt_conv_update_op = conv_update_op  # for tests
    module.libtt_dynamic_conv_update_op = dynamic_conv_update_op


def patch_mamba_pool(module):
    from jax_plugins.libtt import sglang_tpu_gdn

    sglang_tpu_gdn.patch_mamba_pool(module)


def _fused_rope_write_kv(dst_starts, src_starts, span_lens, k, v, cos, sin, *kv_bufs, max_span_len):
    """DFlash's draft KV write (sglang_tpu.kv_fused_rope_write) for the
    `[pages, kv_heads, page_size, 2 * head_dim]` pool: neox RoPE on K, then each
    draft layer's K and V rows to their slots, through TTNN's paged cache update
    as the attention kernel writes them.

    k, v: [layers, tokens, kv_heads, head_dim]; cos, sin: [tokens, 1, head_dim // 2].
    Span i writes source tokens src_starts[i] + j to slots dst_starts[i] + j for
    j < span_lens[i]; spans don't overlap, and padding spans have length 0.
    Returns every layer's updated buffer.
    """
    import jax
    import jax.numpy as jnp

    del max_span_len  # the Pallas kernel's VMEM bound
    _, tokens, heads, head_dim = k.shape
    position = jnp.arange(tokens, dtype=jnp.int32)
    src = src_starts.astype(jnp.int32)[:, None]
    inside = (position[None, :] >= src) & (position[None, :] < src + span_lens.astype(jnp.int32)[:, None])
    slot = jnp.sum(jnp.where(inside, dst_starts.astype(jnp.int32)[:, None] + position[None, :] - src, 0), axis=0)
    live = jnp.any(inside, axis=0)

    half = head_dim // 2
    x1, x2 = k[..., :half], k[..., half:]
    rotated = jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)

    outputs = []
    for layer, buf in enumerate(kv_bufs):
        _, cache_heads, page_size, row = buf.shape
        pad = ((0, 0), (0, cache_heads - heads), (0, row // 2 - head_dim))
        new_kv = jnp.concatenate([jnp.pad(rotated[layer], pad), jnp.pad(v[layer], pad)], axis=-1).astype(buf.dtype)
        # Each token is a user of one page, its slot's, at the slot's offset;
        # tokens outside every span don't write.
        pages = (slot // page_size)[:, None]
        offsets = jnp.where(live, slot % page_size, -1)
        cache = buf
        for start in range(0, tokens, MAX_UPDATE_USERS):
            end = min(start + MAX_UPDATE_USERS, tokens)
            cache = jax.ffi.ffi_call(
                "tt.paged_update_cache",
                jax.ShapeDtypeStruct(buf.shape, buf.dtype),
                input_output_aliases={0: 0},
            )(cache, new_kv[None, start:end], offsets[start:end], pages[start:end], share_cache=True)
        outputs.append(cache)
    return outputs


def patch_kv_fused_rope_write(module):
    module.fused_rope_write_kv_jax = _fused_rope_write_kv


def patch_block_draft_worker(module):
    """DFlash's draft worker returns a value read from the last draft layer's KV
    buffer after its write (the KV free fence's tail), reduced over the
    buffer's other dims in the TPU layout's rank; the TT pool's is one fewer."""
    worker = module.TpuBlockDraftWorker
    original = worker._write_draft_kv_eager

    def _write_draft_kv_eager(self, **kwargs):
        kv_pool = self.model_runner.token_to_kv_pool
        if isinstance(kv_pool, module.SWAKVPool):
            kv_pool = kv_pool.swa_kv_pool
        if getattr(kv_pool, "KV_LAYOUT_NAME", None) != "tt-head-major":
            return original(self, **kwargs)
        draft_model = self._draft_model()
        module.dflash_write_draft_kv(
            draft_model=draft_model,
            kv_pool=kv_pool,
            ctx_hidden=draft_model.project_target_hidden(kwargs["target_hidden"]),
            ctx_positions=kwargs["positions"],
            ctx_cache_loc=kwargs["cache_loc"],
            cache_loc_2d=kwargs.get("cache_loc_2d"),
            commit_lens=kwargs.get("commit_lens"),
            max_num_reqs=self.model_runner.req_to_token_pool.size,
        )
        buf = kv_pool.kv_buffer[-1]
        return buf[:1, :1].to(module.torch.float32).sum(dim=tuple(range(1, buf.dim())))

    worker._write_draft_kv_eager = _write_draft_kv_eager


def patch_all_gather_via_all_reduce(module):
    """Keeps tensor-parallel all-gathers native in compiled graphs.

    SGLang-TorchTPU rebuilds all-gathers from an all-reduce over a zero-padded
    [world, *block] buffer while Dynamo traces, to work around tpu7x's compiled
    all-gather returning wrong blocks from cross-chip peers
    (sgl-project/sglang-torchtpu#1700): every GroupCoordinator
    all_gather_into_tensor, through all_gather_into_tensor_around, and since
    sgl-project/sglang-torchtpu#2041 only the logits gathers, whose hooks ask
    _use_all_reduce_gather. libtt's compiled all-gather is correct, and the
    all-reduce moves about twice its bytes: Qwen3-8B on four chips decoded
    103.8 instead of 112.0 tokens/s.
    """

    def all_gather_into_tensor_around(original, group, output, input):  # noqa: A002
        return original(group, output, input)

    module.all_gather_into_tensor_around = all_gather_into_tensor_around
    module._use_all_reduce_gather = lambda group: False


def _fp8_scales(w13, w13_scale, w2_scale):
    """The scales tt.fused_moe_ep takes with FP8 experts `w13` ([experts, k,
    2 * n], gate then up projections) and w2 ([experts, n, k]), from a scale
    per expert and output column (or one per expert): the gate's and the up
    projection's ([experts, n] each), then the down projection's ([experts,
    k]), float32."""
    import jax.numpy as jnp

    experts, k, gate_up = w13.shape

    def scales(scale, columns):
        return jnp.broadcast_to(scale.astype(jnp.float32).reshape(experts, -1), (experts, columns))

    scale13 = scales(w13_scale, gate_up)
    return scale13[:, : gate_up // 2], scale13[:, gate_up // 2 :], scales(w2_scale, k)


def _mark_weight(w):
    """`w`, marked as a weight (a parameter of the program): ops that depend on weights alone run once, in the
    weight preparation."""
    import jax

    return jax.ffi.ffi_call("tt.weight_dtype_override", jax.ShapeDtypeStruct(w.shape, w.dtype))(
        w, **{"ttcore.weight_dtype": "bf16"}
    )


def _nvfp4_dequant(packed, scale):
    """BF16 [experts, k, n] experts from NVFP4 ones as SGLang-TorchTPU keeps them packed for its TPU kernels: uint8
    [experts, k / 2, n], two E2M1 codes per byte along k (the even row in the low nibble), and a scale per block of
    rows and column with the global scale folded in ([experts, blocks, 1, n]). E2M1 code s e1 e0 m is (-1)^s m / 2
    for e = 0 and (-1)^s (1 + m / 2) 2^(e - 1) otherwise.

    Everything runs in BF16 on [experts, k / 2, n] planes, which holds the bytes, the codes and the E2M1 values
    exactly (the scaled values round to BF16 once, as they would from FP32): the nibbles are floor(b / 16) and
    b - 16 floor(b / 16), each plane is decoded and scaled, and the two interleave along k at the end. Integer bit
    operations and FP32 would take [experts, k, n] tensors of 4 bytes per value."""
    import jax.numpy as jnp

    experts, half_k, n = packed.shape
    blocks = scale.shape[1]
    scale = scale.astype(jnp.bfloat16)
    byte = packed.astype(jnp.bfloat16)
    high = jnp.floor(byte * 0.0625)

    def decode(code):
        negative = code >= 8
        code = jnp.where(negative, code - 8, code)
        exponent = jnp.floor(code * 0.5)
        mantissa = code - 2 * exponent
        power = jnp.where(exponent == 1, 1.0, jnp.where(exponent == 2, 2.0, 4.0)).astype(jnp.bfloat16)
        magnitude = jnp.where(exponent == 0, 0.5 * mantissa, (1.0 + 0.5 * mantissa) * power)
        value = jnp.where(negative, -magnitude, magnitude)
        return (value.reshape(experts, blocks, -1, n) * scale).reshape(experts, half_k, n)

    # [experts, k / 2, 2n]: row j holds rows 2j and 2j + 1 side by side, which a row-major reshape interleaves (a
    # stacked [experts, k / 2, 2, n] would pad its size-2 axis to a tile's 32 rows).
    planes = jnp.concatenate([decode(byte - 16 * high), decode(high)], axis=-1)
    return planes.reshape(experts, 2 * half_k, n)


def _fused_moe_jax(
    hidden_states,
    w1,
    w2,
    topk_weights,
    topk_ids,
    w1_scale,
    w2_scale,
    w1_bias,
    w2_bias,
    topk,
    activation_name,
    no_combine=False,
    global_num_experts=0,
    expert_group_offset=0,
    use_sc_gather=0,
    w_quant_bits=0,
    permute_dense_threshold=0,
    w1_channel_scale=None,
    w2_channel_scale=None,
    expert_group_offset_tensor=None,
    gather_permute_with_sc_combine=0,
):
    """The fused MoE op on TTNN's fused MoE kernel (tt.fused_moe_ep), with the
    experts in BF16 or FP8, as loaded, or NVFP4 (packed, with block scales),
    which run as BFP4.

    FP8 experts come with SGLang-TorchTPU's channelwise runtime scales, one per
    expert and output column. The kernel reads the FP8 weights as they are and
    applies the scales. The activations stay unquantized (FLOAT16 inside the
    kernel, which the matrix unit reads next to FP8): the Pallas kernel
    quantizes them to FP8 too, but only on TPUs with FP8 arithmetic.

    The kernel runs whole experts, split evenly over the chips in rank order,
    which is how SGLang-TorchTPU splits them with --ep-size equal to --tp-size.
    Each rank returns its experts' routing-weighted sum for every token, and
    the layer's all-reduce adds them up. Like the Pallas kernel, it reads the
    router's global expert ids. w1 holds each expert's gate and up projections
    side by side, [experts, hidden, 2 * intermediate].

    The experts are SiLU-gated, or SiTU-gated with activation_name
    "situ:{beta}:{linear_beta}" (Inkling 2): beta * tanh(g / beta) * sigmoid(g)
    * linear_beta * tanh(u / linear_beta) for gate g and up u.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np
    import torch.distributed as dist

    activation = {}
    if activation_name.startswith("situ:"):
        beta, linear_beta = activation_name.split(":")[1:]
        if linear_beta == "none":
            raise NotImplementedError("SiTU MoE on Tenstorrent takes an up softcap (linear_beta)")
        activation = dict(
            activation="situ", activation_beta=np.float32(beta), activation_limit=np.float32(linear_beta)
        )
    elif activation_name != "silu":
        raise NotImplementedError(f"MoE on Tenstorrent runs SiLU or SiTU experts, not {activation_name}")
    extras = (w1_bias, w2_bias, w1_channel_scale, w2_channel_scale)
    if no_combine or w_quant_bits or any(e is not None for e in extras):
        raise NotImplementedError("MoE on Tenstorrent runs BF16 or FP8 experts without biases, combined")
    nvfp4 = w1.dtype == w2.dtype == jnp.uint8
    fp8 = w1.dtype == w2.dtype == jnp.float8_e4m3fn
    if nvfp4:
        if w1_scale is None or w2_scale is None or w1_scale.ndim != 4 or w2_scale.ndim != 4:
            raise NotImplementedError("NVFP4 MoE on Tenstorrent takes packed E2M1 experts with block scales")
    elif fp8:
        if w1_scale is None or w2_scale is None or w1_scale.shape[1] != 1 or w2_scale.shape[1] != 1:
            raise NotImplementedError(
                "FP8 MoE on Tenstorrent takes channelwise scales: pass --tpu-fp8-runtime channelwise"
            )
    elif w1_scale is not None or w2_scale is not None or not w1.dtype == w2.dtype == jnp.bfloat16:
        raise NotImplementedError("MoE on Tenstorrent runs BF16 or FP8 experts")
    if hidden_states.dtype != jnp.bfloat16:
        raise NotImplementedError("MoE on Tenstorrent runs on BF16 activations")
    experts, _, gate_up = w1.shape
    ranks = dist.get_world_size() if dist.is_initialized() else 1
    if (global_num_experts or experts) != experts * ranks:
        raise NotImplementedError(
            "MoE on Tenstorrent splits whole experts over the chips: pass --ep-size equal to --tp-size"
        )

    # Mark the experts, and FP8 experts' scales, as weights, so that the kernel's
    # weight preparation runs once rather than on every call, and keep BF16
    # experts in BF16.
    def weight(w):
        return jax.ffi.ffi_call("tt.weight_dtype_override", jax.ShapeDtypeStruct(w.shape, w.dtype))(
            w, **{"ttcore.weight_dtype": "bf16"}
        )

    fused_moe_ep = jax.ffi.ffi_call("tt.fused_moe_ep", jax.ShapeDtypeStruct(hidden_states.shape, hidden_states.dtype))
    if nvfp4:
        # NVFP4 experts map to BFP4, TT's 4-bit block format: decoded once, in the weight preparation (the packed
        # codes and their scales are marked as weights first), then prepared by the kernel as BFP4. BFP4 shares one
        # exponent per 16 values of a tile row and has 3-bit mantissas, so this re-encodes the weights.
        w1 = _nvfp4_dequant(weight(w1), weight(w1_scale))
        w2 = _nvfp4_dequant(weight(w2), weight(w2_scale))
        operands = (hidden_states, topk_weights, topk_ids, w1[..., : gate_up // 2], w1[..., gate_up // 2 :], w2)
        return fused_moe_ep(*operands, weight_dtype="bfp4", **activation)
    w1, w2 = weight(w1), weight(w2)
    operands = (hidden_states, topk_weights, topk_ids, w1[..., : gate_up // 2], w1[..., gate_up // 2 :], w2)
    if fp8:
        return fused_moe_ep(*operands, *_fp8_scales(w1, weight(w1_scale), weight(w2_scale)), **activation)
    return fused_moe_ep(*operands, weight_dtype="bf16", **activation)


def patch_fused_moe(module):
    module._fused_moe_jax = _fused_moe_jax


def patch_fp8_moe_fused_rs(module):
    """SGLang-TorchTPU runs FP8 experts on a Pallas kernel that fuses the MoE
    with its reduce-scatter where the layout allows. Report every layout as one
    it does not cover: the experts then run on the fused MoE op, which is
    TTNN's fused MoE kernel here, followed by an all-reduce."""

    def prepare_fp8_fused_rs(layer):
        layer.__dict__["_tpu_fused_rs_unsupported_reason"] = (
            "Tenstorrent runs the fused MoE kernel followed by an all-reduce"
        )
        return False

    module.prepare_fp8_fused_rs = prepare_fp8_fused_rs


def patch_fp4_moe(module):
    """SGLang-TorchTPU keeps NVFP4 experts packed with SGLANG_TPU_INKLING2_NVFP4_EXPERTS for its Pallas kernels,
    which take this rank's local expert ids (or the fused-RS kernel). Here they run on the fused MoE op like FP8
    experts: global expert ids, TTNN's fused MoE kernel (NVFP4 as BFP4) and the layer's all-reduce."""
    import torch

    from sglang.srt.runtime_context import get_parallel

    def unmet_tpu_nvfp4_fused_rs_requirements(*args, **kwargs):
        return ["Tenstorrent runs the fused MoE kernel followed by an all-reduce"]

    def apply_tpu_nvfp4_moe(
        *,
        layer,
        dispatch_output,
        activation,
        apply_router_weight_on_input,
        no_combine=False,
        use_fused_rs=False,
        swiglu_limit=None,
    ):
        if apply_router_weight_on_input or use_fused_rs:
            raise NotImplementedError("NVFP4 MoE on Tenstorrent runs the fused MoE op, router weights on the output")
        x = dispatch_output.hidden_states
        topk_weights, topk_ids, _ = dispatch_output.topk_output
        global_num_experts = expert_group_offset = 0
        ep_size, ep_rank = get_parallel().moe_ep_size, get_parallel().moe_ep_rank
        if ep_size > 1:
            global_num_experts = layer.num_local_experts * ep_size
            expert_group_offset = ep_rank * layer.num_local_experts
        if swiglu_limit is not None:
            if activation != "silu":
                raise ValueError("TPU clamped MoE requires silu activation")
            activation = f"silu_and_mul_clamp:{float(swiglu_limit)}"
        output = module.fused_moe_call(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights.to(x.dtype),
            topk_ids=topk_ids.to(torch.int32),
            topk=int(topk_ids.shape[-1]),
            activation=module.pallas_moe_activation(activation, layer),
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            no_combine=no_combine,
            global_num_experts=global_num_experts,
            expert_group_offset=expert_group_offset,
        )
        return module.StandardCombineInput(hidden_states=output)

    prepare = module.prepare_tpu_nvfp4_moe_weights

    def prepare_tpu_nvfp4_moe_weights(layer, group_size=module.TPU_NVFP4_GROUP_SIZE, **kwargs):
        # The Pallas kernels' K and N alignment (512) would pad Inkling 2's intermediate size 1152 to 1536; the
        # fused MoE kernel needs whole scale blocks only.
        kwargs.setdefault("align_to", group_size)
        return prepare(layer, group_size=group_size, **kwargs)

    module.prepare_tpu_nvfp4_moe_weights = prepare_tpu_nvfp4_moe_weights
    module.unmet_tpu_nvfp4_fused_rs_requirements = unmet_tpu_nvfp4_fused_rs_requirements
    module.can_use_tpu_nvfp4_fused_rs = lambda *args, **kwargs: False
    module.apply_tpu_nvfp4_moe = apply_tpu_nvfp4_moe


def patch_inkling2_attn_res(module):
    """Inkling 2's block attention residuals for decode steps: up to four tokens and eight
    sources run on `ttnn::experimental::attn_res` (the `tt.attn_res` call), one program where
    the stack, its RMS norm, the logits matmul, the softmax over the sources, the weighted sum
    and the next norm took eleven. Both upstream's fused form with the RMS norm that follows
    (its Pallas prefill kernel `inkling2_attn_res_norm`) and the mixture alone
    (`attn_res_tpu`: the later loop passes, the loop messages and the final output). And the
    loop-message controller (`loop_mix_tpu`) on `ttnn::experimental::loop_mix` (`tt.loop_mix`):
    the RMS norms, the down, gain and forget matmuls, the tanh gates and the mix in one program.
    Larger batches keep upstream's XLA path; the Pallas kernel itself only runs on TPUs."""
    import functools
    from typing import Optional

    import jax
    import jax.numpy as jnp
    import numpy as np
    import torch

    max_tokens, max_sources = 4, 8

    def supports_attn_res_norm(tokens: int, num_sources: int) -> bool:
        return tokens <= max_tokens and num_sources <= max_sources

    def attn_res_call(sources, addend, queries, norm_weights, *, normalize, eps_res, eps_norm=0.0):
        """[Q, T, W]: BF16 normed mixtures with norm weights, else the FP32 mixtures."""
        tokens, width = sources[0].shape
        operands = [
            *(s.astype(jnp.bfloat16) for s in sources),
            *([addend] if addend is not None else []),
            queries.astype(jnp.float32),
            *([norm_weights.astype(jnp.float32)] if norm_weights is not None else []),
        ]
        dtype = jnp.bfloat16 if norm_weights is not None else jnp.float32
        return jax.ffi.ffi_call("tt.attn_res", jax.ShapeDtypeStruct((queries.shape[0], tokens, width), dtype))(
            *operands,
            num_sources=np.int64(len(sources)),
            has_addend=addend is not None,
            has_weight=norm_weights is not None,
            normalize=bool(normalize),
            eps_res=np.float32(eps_res),
            eps_norm=np.float32(eps_norm),
        )

    def attn_res_norm_jax(sources, addend, queries, norm_weights, *, normalize, eps_res, eps_norm):
        return attn_res_call(
            sources, addend, queries, norm_weights, normalize=normalize, eps_res=eps_res, eps_norm=eps_norm
        )

    @functools.cache
    def register_inkling2_attn_res_op():
        from torch_tpu._internal import pallas

        # Upstream's signature: ten source slots, unused ones None.
        def _op(
            s0: jax.Array,
            s1: Optional[jax.Array],
            s2: Optional[jax.Array],
            s3: Optional[jax.Array],
            s4: Optional[jax.Array],
            s5: Optional[jax.Array],
            s6: Optional[jax.Array],
            s7: Optional[jax.Array],
            s8: Optional[jax.Array],
            s9: Optional[jax.Array],
            addend: Optional[jax.Array],
            queries: jax.Array,
            norm_weights: jax.Array,
            normalize: bool,
            eps_res: float,
            eps_norm: float,
        ) -> jax.Array:
            sources = [s for s in (s0, s1, s2, s3, s4, s5, s6, s7, s8, s9) if s is not None]
            return attn_res_norm_jax(
                sources, addend, queries, norm_weights, normalize=normalize, eps_res=eps_res, eps_norm=eps_norm
            )

        op = pallas.jax_op("libtt::inkling2_attn_res_norm", _op)

        @op.register_fake
        def fake(s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, addend, queries, norm_weights, normalize, eps_res, eps_norm):
            return s0.new_empty((queries.shape[0], *s0.shape), dtype=torch.bfloat16)

        return op

    @functools.cache
    def mixture_op():
        from torch_tpu._internal import pallas

        def _op(
            s0: jax.Array,
            s1: Optional[jax.Array],
            s2: Optional[jax.Array],
            s3: Optional[jax.Array],
            s4: Optional[jax.Array],
            s5: Optional[jax.Array],
            s6: Optional[jax.Array],
            s7: Optional[jax.Array],
            addend: Optional[jax.Array],
            query_rows: jax.Array,
            normalize: bool,
            eps_res: float,
        ) -> jax.Array:
            sources = [s for s in (s0, s1, s2, s3, s4, s5, s6, s7) if s is not None]
            return attn_res_call(sources, addend, query_rows, None, normalize=normalize, eps_res=eps_res)

        op = pallas.jax_op("libtt::inkling2_attn_res", _op)

        @op.register_fake
        def fake(s0, s1, s2, s3, s4, s5, s6, s7, addend, query_rows, normalize, eps_res):
            return s0.new_empty((query_rows.shape[0], *s0.shape), dtype=torch.float32)

        return op

    def attn_res_tpu(sources, addend, query_rows, *, normalize, eps_res):
        slots = list(sources) + [None] * (max_sources - len(sources))
        return mixture_op()(*slots, addend, query_rows, bool(normalize), float(eps_res))

    def supports_loop_mix(tokens: int, num_queries: int) -> bool:
        return tokens <= max_tokens and num_queries <= 3

    def loop_mix_call(dest, messages, down, gain, forget, norm_weight, eps, eps_norm):
        operands = [
            dest.astype(jnp.float32),
            messages.astype(jnp.float32),
            down.astype(jnp.bfloat16),
            gain.astype(jnp.bfloat16),
            forget.astype(jnp.bfloat16),
        ]
        if norm_weight is None:
            results = jax.ShapeDtypeStruct(messages.shape, jnp.float32)
        else:
            operands.append(norm_weight.astype(jnp.float32))
            results = tuple(jax.ShapeDtypeStruct(messages.shape[1:], jnp.bfloat16) for _ in range(messages.shape[0]))
        return jax.ffi.ffi_call("tt.loop_mix", results)(
            *operands,
            epsilon=np.float32(eps),
            has_norm=norm_weight is not None,
            eps_norm=np.float32(eps_norm),
        )

    @functools.cache
    def loop_mix_op():
        from torch_tpu._internal import pallas

        def _op(
            dest: jax.Array, messages: jax.Array, down: jax.Array, gain: jax.Array, forget: jax.Array, eps: float
        ) -> jax.Array:
            return loop_mix_call(dest, messages, down, gain, forget, None, eps, 0.0)

        op = pallas.jax_op("libtt::inkling2_loop_mix", _op)

        @op.register_fake
        def fake(dest, messages, down, gain, forget, eps):
            return torch.empty_like(messages, dtype=torch.float32)

        return op

    @functools.cache
    def loop_mix_normed_op(num_queries):
        from torch_tpu._internal import pallas

        def _op(
            dest: jax.Array,
            messages: jax.Array,
            down: jax.Array,
            gain: jax.Array,
            forget: jax.Array,
            norm_weight: jax.Array,
            eps: float,
            eps_norm: float,
        ) -> tuple[jax.Array, ...]:
            return tuple(loop_mix_call(dest, messages, down, gain, forget, norm_weight, eps, eps_norm))

        op = pallas.jax_op(f"libtt::inkling2_loop_mix_normed_{num_queries}", _op)

        @op.register_fake
        def fake(dest, messages, down, gain, forget, norm_weight, eps, eps_norm):
            return tuple(messages.new_empty(messages.shape[1:], dtype=torch.bfloat16) for _ in range(num_queries))

        return op

    def loop_mix_tpu(dest, messages, down_t, gain_t, forget_t, *, eps, norm_weight=None, eps_norm=None):
        if norm_weight is None:
            return loop_mix_op()(dest, messages, down_t, gain_t, forget_t, float(eps))
        return loop_mix_normed_op(messages.shape[0])(
            dest, messages, down_t, gain_t, forget_t, norm_weight, float(eps), float(eps_norm)
        )

    module.supports_attn_res_norm = supports_attn_res_norm
    module.attn_res_norm_jax = attn_res_norm_jax
    module.supports_loop_mix = supports_loop_mix
    module.loop_mix_tpu = loop_mix_tpu
    module.register_inkling2_attn_res_op = register_inkling2_attn_res_op
    module.supports_attn_res = supports_attn_res_norm
    module.attn_res_tpu = attn_res_tpu


def patch_inkling2_l2_clip(module):
    """Inkling 2's per-head L2 clips of its queries, keys and relative projections, and its
    attention output gate (an infinite threshold and a sigmoid gate), on
    `ttnn::experimental::l2_clip` (the `tt.l2_clip` call): one program where the typecast, the
    scale, the RMS norm and the abs, minimum, sign and tau products took seven, reading its
    columns (and the gate's) in place from the projection."""
    import functools
    from typing import Optional

    import jax
    import jax.numpy as jnp
    import numpy as np
    import torch

    def supports_l2_clip(tokens: int) -> bool:
        return True

    @functools.cache
    def l2_clip_op():
        from torch_tpu._internal import pallas

        def _op(
            x: jax.Array,
            scale: Optional[jax.Array],
            row_scale: Optional[jax.Array],
            gate: Optional[jax.Array],
            group: int,
            threshold: float,
            eps: float,
            offset: int,
            width: int,
            gate_offset: int,
        ) -> jax.Array:
            operands = [x]
            if scale is not None:
                operands.append(scale.astype(jnp.float32))
            if row_scale is not None:
                operands.append(row_scale.astype(jnp.float32).reshape(-1, 1))
            if gate is not None:
                operands.append(gate)
            return jax.ffi.ffi_call("tt.l2_clip", jax.ShapeDtypeStruct((*x.shape[:-1], width), x.dtype))(
                *operands,
                group=np.int64(group),
                offset=np.int64(offset),
                width=np.int64(width),
                threshold=np.float32(threshold),
                epsilon=np.float32(eps),
                has_scale=scale is not None,
                has_row_scale=row_scale is not None,
                has_gate=gate is not None,
                gate_offset=np.int64(gate_offset),
            )

        op = pallas.jax_op("libtt::inkling2_l2_clip", _op)

        @op.register_fake
        def fake(x, scale, row_scale, gate, group, threshold, eps, offset, width, gate_offset):
            return x.new_empty((*x.shape[:-1], width))

        return op

    def l2_clip_tpu(
        x, *, group, threshold, eps, scale=None, row_scale=None, offset=0, width=None, gate=None, gate_offset=0
    ):
        width = x.shape[-1] - offset if width is None else width
        return l2_clip_op()(
            x, scale, row_scale, gate, int(group), float(threshold), float(eps), int(offset), int(width), int(gate_offset)
        )

    module.supports_l2_clip = supports_l2_clip
    module.l2_clip_tpu = l2_clip_tpu


def patch_inkling2_route(module):
    """Inkling 2's router selection for decode steps (up to four tokens) on
    `ttnn::experimental::moe_route` (the `tt.moe_route` call): the sigmoid, the top k of score
    + bias, the selected scores' renormalization and the padding rows' masks in one program
    where the add, the top k, the one-hot gather and the sums, divide, multiply and wheres took
    thirteen."""
    import functools
    from typing import Optional

    import jax
    import jax.numpy as jnp
    import numpy as np
    import torch

    def supports_route(tokens: int, num_experts: int, top_k: int) -> bool:
        return tokens <= 4 and num_experts % 32 == 0 and num_experts <= 256 and top_k <= 16

    @functools.cache
    def route_op():
        from torch_tpu._internal import pallas

        def _op(
            logits: jax.Array, bias: jax.Array, num_valid: Optional[jax.Array], top_k: int, scale: float
        ) -> tuple[jax.Array, jax.Array]:
            tokens = logits.shape[0]
            operands = [logits.astype(jnp.float32), bias]
            if num_valid is not None:
                operands.append(num_valid.astype(jnp.int32).reshape(1))
            return jax.ffi.ffi_call(
                "tt.moe_route",
                (
                    jax.ShapeDtypeStruct((tokens, top_k), jnp.float32),
                    jax.ShapeDtypeStruct((tokens, top_k), jnp.int32),
                ),
            )(*operands, k=np.int64(top_k), scale=np.float32(scale), has_num_valid=num_valid is not None)

        op = pallas.jax_op("libtt::inkling2_route", _op)

        @op.register_fake
        def fake(logits, bias, num_valid, top_k, scale):
            shape = (logits.shape[0], top_k)
            return logits.new_empty(shape, dtype=torch.float32), logits.new_empty(shape, dtype=torch.int32)

        return op

    def route_tpu(logits, bias, top_k, scale, num_valid=None):
        return route_op()(logits, bias, num_valid, int(top_k), float(scale))

    module.supports_route = supports_route
    module.route_tpu = route_tpu


PATCHES = {
    RPA_V3_MODULE: patch_rpa_v3,
    KV_POOL_MODULE: patch_kv_pool,
    FP8_LINEAR_SMALL_M_MODULE: patch_fp8_linear_small_m,
    GDN_MODULE: patch_gdn,
    MAMBA_POOL_MODULE: patch_mamba_pool,
    MAMBA_STATE_ROWS_MODULE: patch_mamba_state_rows,
    INKLING_SCONV_MODULE: patch_inkling_sconv,
    KV_FUSED_ROPE_WRITE_MODULE: patch_kv_fused_rope_write,
    BLOCK_DRAFT_WORKER_MODULE: patch_block_draft_worker,
    ALL_GATHER_VIA_ALL_REDUCE_MODULE: patch_all_gather_via_all_reduce,
    FUSED_MOE_MODULE: patch_fused_moe,
    FP8_MOE_FUSED_RS_MODULE: patch_fp8_moe_fused_rs,
    FP4_MOE_MODULE: patch_fp4_moe,
    INKLING2_ATTN_RES_MODULE: patch_inkling2_attn_res,
    INKLING2_L2_CLIP_MODULE: patch_inkling2_l2_clip,
    INKLING2_ROUTE_MODULE: patch_inkling2_route,
}
