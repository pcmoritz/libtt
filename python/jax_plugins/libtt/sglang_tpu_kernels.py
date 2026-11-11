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
attention: given the cache as both keys and values and the queries padded with
zeros, it computes softmax(q k) [k v], whose second half is the output. A
prefill runs TTNN's chunked paged attention the same way, one sequence per
user, so it reads only each sequence's pages.

Hybrid models' gated DeltaNet kernel runs on libtt's recurrent kernels, with
the recurrent-state pool in their layout (`sglang_tpu_gdn`). Mixture-of-experts
models' fused MoE op runs on TTNN's fused MoE kernel, with the experts in BF16
or FP8, as loaded.
"""

# Users per TTNN paged decode attention call.
MAX_DECODE_USERS = 32
# Tokens per TTNN paged cache update, one core each.
MAX_UPDATE_USERS = 32

RPA_V3_MODULE = "sglang_tpu.attention.pallas.ragged_paged_attention_v3"
KV_POOL_MODULE = "sglang_tpu.kv_pool"
FP8_LINEAR_SMALL_M_MODULE = "sglang_tpu.kernels.fp8_linear_small_m"
GDN_MODULE = "sglang_tpu.attention.pallas.gdn"
MAMBA_POOL_MODULE = "sglang_tpu.mamba_pool"
KV_FUSED_ROPE_WRITE_MODULE = "sglang_tpu.kv_fused_rope_write"
BLOCK_DRAFT_WORKER_MODULE = "sglang_tpu.experimental.speculative.block_draft_worker"
ALL_GATHER_VIA_ALL_REDUCE_MODULE = "sglang_tpu.all_gather_via_all_reduce"
FUSED_MOE_MODULE = "sglang_tpu.moe.pallas.fused_moe_op"
FP8_MOE_FUSED_RS_MODULE = "sglang_tpu.fp8_moe_fused_rs"


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
        if (
            tokens <= max(seqs, MAX_DECODE_USERS)
            and cache_heads == kv_heads
            and causal
            and soft_cap is None
            and attention_sink is None
            and q_scale is None
            and k_scale is None
            and v_scale is None
        ):
            out = _paged_decode(
                queries, cache, user_pages, user_positions, sm_scale, sliding_window
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
        scores = jnp.where(allowed[None, None], scores, mask_value)
        if attention_sink is not None:
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


def _paged_decode(queries, cache, page_table, positions, sm_scale, sliding_window):
    """Each token as a TTNN decode user, attending to its sequence up to its position."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    tokens, q_heads, head_dim = queries.shape
    cache_head_dim = cache.shape[-1] // 2
    q = jnp.pad(queries, ((0, 0), (0, 0), (0, 2 * cache_head_dim - head_dim))).astype(cache.dtype)
    window = {} if sliding_window is None else {"sliding_window_size": np.int32(sliding_window)}
    outputs = []
    for start in range(0, tokens, MAX_DECODE_USERS):
        users = q[None, start : start + MAX_DECODE_USERS]
        outputs.append(
            jax.ffi.ffi_call(
                "tt.paged_scaled_dot_product_attention_decode",
                jax.ShapeDtypeStruct(users.shape, users.dtype),
            )(
                users,
                cache,
                cache,
                page_table[start : start + MAX_DECODE_USERS],
                positions[start : start + MAX_DECODE_USERS],
                is_causal=True,
                has_attention_mask=False,
                has_cur_pos_tensor=True,
                has_attention_sink=False,
                scale=np.float32(sm_scale),
                **window,
            )
        )
    return jnp.concatenate(outputs, axis=1)[0, :, :, cache_head_dim : cache_head_dim + head_dim]


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
        # The op returns the float8 codes as uint8, as the compiler stores
        # float8 tensors; the bitcast is a view.
        codes, scale = jax.ffi.ffi_call(
            "tt.quantize_rows",
            (
                jax.ShapeDtypeStruct(x.shape, jnp.uint8),
                jax.ShapeDtypeStruct((tokens, 1), jnp.float32),
            ),
        )(x, min_scale=np.float32(min_scale), e5m2=np.bool_(dtype == jnp.float8_e5m2))
        return jax.lax.bitcast_convert_type(codes, dtype), scale
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

    SGLang-TorchTPU rebuilds GroupCoordinator.all_gather_into_tensor from an
    all-reduce over a zero-padded [world, *block] buffer while Dynamo traces,
    to work around tpu7x's compiled all-gather returning wrong blocks from
    cross-chip peers (sgl-project/sglang-torchtpu#1700). libtt's compiled
    all-gather is correct, and the all-reduce moves about twice its bytes:
    Qwen3-8B on four chips decoded 103.8 instead of 112.0 tokens/s.
    """

    def all_gather_into_tensor_around(original, group, output, input):  # noqa: A002
        return original(group, output, input)

    module.all_gather_into_tensor_around = all_gather_into_tensor_around


def _fp8_experts(w13, w13_scale, w2, w2_scale):
    """FP8 experts as tt.fused_moe_ep takes them: `w13` ([experts, k, 2 * n],
    gate then up projections) and `w2` ([experts, n, k]) with an FP32 scale per
    expert and output column (or one per expert). Returns the gate, the up and
    the down projections ([experts, k, n] twice and [experts, n, k]), then
    their scales ([experts, n] twice and [experts, k] float32)."""
    import jax.numpy as jnp

    experts, k, gate_up = w13.shape
    n = gate_up // 2

    def scales(scale, columns):
        return jnp.broadcast_to(scale.astype(jnp.float32).reshape(experts, -1), (experts, columns))

    scale13 = scales(w13_scale, gate_up)
    return w13[..., :n], w13[..., n:], w2, scale13[:, :n], scale13[:, n:], scales(w2_scale, k)


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
    experts in BF16 or FP8, as loaded.

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
    """
    import jax
    import jax.numpy as jnp
    import torch.distributed as dist

    extras = (w1_bias, w2_bias, w1_channel_scale, w2_channel_scale)
    if activation_name != "silu" or no_combine or w_quant_bits or any(e is not None for e in extras):
        raise NotImplementedError("MoE on Tenstorrent runs BF16 or FP8 SiLU experts without biases, combined")
    fp8 = w1.dtype == w2.dtype == jnp.float8_e4m3fn
    if fp8:
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

    w1, w2 = weight(w1), weight(w2)
    out = jax.ShapeDtypeStruct(hidden_states.shape, hidden_states.dtype)
    if fp8:
        return jax.ffi.ffi_call("tt.fused_moe_ep", out)(
            hidden_states,
            topk_weights,
            topk_ids,
            *_fp8_experts(w1, weight(w1_scale), w2, weight(w2_scale)),
            weight_dtype="fp8",
        )
    return jax.ffi.ffi_call("tt.fused_moe_ep", out)(
        hidden_states,
        topk_weights,
        topk_ids,
        w1[..., : gate_up // 2],
        w1[..., gate_up // 2 :],
        w2,
        weight_dtype="bf16",
    )


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


PATCHES = {
    RPA_V3_MODULE: patch_rpa_v3,
    KV_POOL_MODULE: patch_kv_pool,
    FP8_LINEAR_SMALL_M_MODULE: patch_fp8_linear_small_m,
    GDN_MODULE: patch_gdn,
    MAMBA_POOL_MODULE: patch_mamba_pool,
    KV_FUSED_ROPE_WRITE_MODULE: patch_kv_fused_rope_write,
    BLOCK_DRAFT_WORKER_MODULE: patch_block_draft_worker,
    ALL_GATHER_VIA_ALL_REDUCE_MODULE: patch_all_gather_via_all_reduce,
    FUSED_MOE_MODULE: patch_fused_moe,
    FP8_MOE_FUSED_RS_MODULE: patch_fp8_moe_fused_rs,
}
