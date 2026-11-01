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
operations, and the only one the kernel reads. A single token's row is
written by TTNN's paged cache update, and a batch of no more
tokens than sequence slots, such as a decode batch, runs TTNN's paged decode
attention: given the cache as both keys and values and the queries padded with
zeros, it computes softmax(q k) [k v], whose second half is the output.
"""

# Users per TTNN paged decode attention call.
MAX_DECODE_USERS = 32

RPA_V3_MODULE = "sglang_tpu.attention.pallas.ragged_paged_attention_v3"
KV_POOL_MODULE = "sglang_tpu.kv_pool"


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
        user_pages = table[seq]
        user_positions = jnp.where(live, position, -1)
        if tokens == 1:
            cache = jax.ffi.ffi_call(
                "tt.paged_update_cache",
                jax.ShapeDtypeStruct(cache_shape, kv_cache_fused.dtype),
                input_output_aliases={0: 0},
            )(kv_cache_fused, new_kv[None], user_positions, user_pages)
        else:
            # Tokens of one sequence may share a page, which TTNN's paged
            # update writes concurrently.
            page = table.reshape(-1)[seq * pages_per_seq + position // page_size]
            rows = (page[:, None] * cache_heads + jnp.arange(cache_heads)[None, :]) * page_size
            rows = rows + (position % page_size)[:, None]
            rows = jnp.where(live[:, None], rows, pages * cache_heads * page_size)
            cache = kv_cache_fused.reshape(pages * cache_heads * page_size, row)
            cache = cache.at[rows.reshape(-1)].set(new_kv.reshape(-1, row), mode="drop")
            cache = cache.reshape(cache_shape)
        # Each token as a decode user reads its sequence's pages once, which
        # beats gathering them below when there are few tokens per sequence.
        if (
            tokens <= seqs
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


PATCHES = {RPA_V3_MODULE: patch_rpa_v3, KV_POOL_MODULE: patch_kv_pool}
