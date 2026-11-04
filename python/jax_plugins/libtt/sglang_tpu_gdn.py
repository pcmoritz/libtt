"""SGLang-TorchTPU's fused Conv1D + gated delta rule kernel, without Pallas.

Hybrid models such as Qwen3.5 run their gated DeltaNet layers through
`sglang_tpu.attention.pallas.gdn.fused_conv1d_gdn`, a Pallas TPU kernel. In
TPU compatibility mode, `sglang_tpu_kernels` replaces it with
`fused_conv1d_gdn` below, which takes the same arguments and returns the same
results with static shapes. A decode batch (`batched_only`, one token per
request) takes one recurrence step for every request at once. An extend batch
(`prefill_only`) runs the chunkwise-parallel recurrence over 64-token chunks.

The chunked recurrence comes from SGLang-JAX's gated delta rule
(`sgl_jax/srt/kernels/gdn/gated_delta.py`, Apache-2.0), which serves Qwen3.5 on
Tenstorrent. State layouts are SGLang-TorchTPU's: conv state
`[slots, kernel_size - 1, 1, conv_dim]` (or without the 1), recurrent state
`[slots, n_v, d_k, d_v]`.
Slot 0 is the reserved null slot, which padded rows point at and which is never
written.
"""

import jax
import jax.numpy as jnp

CHUNK = 64


def _l2norm(x, eps=1e-6):
    x = x.astype(jnp.float32)
    return x / jnp.sqrt(jnp.sum(x * x, axis=-1, keepdims=True) + eps)


def _scatter_slots(pool, slots, values):
    """pool.at[slots].set(values), keeping slot 0 as it is."""
    keep = (slots == 0).reshape((-1,) + (1,) * (pool.ndim - 1))
    values = jnp.where(keep, pool[slots], values.astype(pool.dtype))
    return pool.at[slots].set(values)


def _split_heads(mixed, n_kq, n_v, d_k, d_v):
    """q, k scaled and normalized as in the reference, v; per token, n_v heads."""
    key_dim = n_kq * d_k
    q = mixed[..., :key_dim].reshape(*mixed.shape[:-1], n_kq, d_k)
    k = mixed[..., key_dim : 2 * key_dim].reshape(*mixed.shape[:-1], n_kq, d_k)
    v = mixed[..., 2 * key_dim :].reshape(*mixed.shape[:-1], n_v, d_v)
    if n_v > n_kq:
        q = jnp.repeat(q, n_v // n_kq, axis=-2)
        k = jnp.repeat(k, n_v // n_kq, axis=-2)
    return _l2norm(q) * d_k**-0.5, _l2norm(k), v.astype(jnp.float32)


def _gates(b, a, a_log, dt_bias):
    beta = jax.nn.sigmoid(b.astype(jnp.float32))
    g = -jnp.exp(a_log.astype(jnp.float32)) * jax.nn.softplus(
        a.astype(jnp.float32) + dt_bias.astype(jnp.float32)
    )
    return beta, g


def _history(conv_state, slots, has_init, dim):
    """Each request's conv state, [B, K-1, conv_dim], or zeros for a new request."""
    history = conv_state[slots].reshape(slots.shape[0], conv_state.shape[1], dim)
    return jnp.where(has_init[:, None, None], history, 0)


def _conv_decode(qkv, conv_state, conv_weight, conv_bias, slots, has_init):
    """One token per request: [B, conv_dim] after the depthwise causal conv."""
    history = _history(conv_state, slots, has_init, qkv.shape[1])
    window = jnp.concatenate([history, qkv[:, None, :]], axis=1)  # [B, K, D]
    out = jnp.einsum(
        "bkd,dk->bd", window.astype(jnp.float32), conv_weight[:, 0, :].astype(jnp.float32)
    )
    if conv_bias is not None:
        out = out + conv_bias.astype(jnp.float32)
    new = window[:, 1:].reshape(slots.shape[0], *conv_state.shape[1:])
    return out, _scatter_slots(conv_state, slots, new)


def _conv_extend(qkv, conv_state, conv_weight, conv_bias, starts, slots, has_init):
    """Packed requests: [T, conv_dim] after the depthwise causal conv.

    starts is [B + 1]; a token past starts[-1] is padding. Each request's
    lookback before its first token comes from its conv state, or zeros.
    """
    tokens, dim = qkv.shape
    taps = conv_weight.shape[-1]
    reqs = slots.shape[0]
    history = _history(conv_state, slots, has_init, dim)  # [B, K-1, D]
    token = jnp.arange(tokens, dtype=jnp.int32)
    req = jnp.clip(jnp.sum(token[:, None] >= starts[None, 1:], axis=1), 0, reqs - 1)
    pos = token - starts[req]
    flat_history = history.reshape(reqs * (taps - 1), dim)
    out = jnp.zeros((tokens, dim), jnp.float32)
    for back in range(taps):
        prev = qkv[jnp.maximum(token - back, 0)]
        lookback = flat_history[req * (taps - 1) + jnp.clip(taps - 1 - back + pos, 0, taps - 2)]
        x = jnp.where((pos >= back)[:, None], prev, lookback)
        out = out + x.astype(jnp.float32) * conv_weight[:, 0, taps - 1 - back].astype(jnp.float32)
    if conv_bias is not None:
        out = out + conv_bias.astype(jnp.float32)
    # The new conv state: each request's last K-1 inputs, its history before them.
    lens = starts[1:] - starts[:-1]
    m = jnp.arange(taps - 1, dtype=jnp.int32)
    src = starts[1:, None] - (taps - 1) + m[None, :]  # [B, K-1] token indices
    from_tokens = (lens[:, None] + m[None, :]) >= taps - 1
    hist_idx = jnp.clip(lens[:, None] + m[None, :], 0, taps - 2)
    new = jnp.where(
        from_tokens[..., None],
        qkv[jnp.clip(src, 0, tokens - 1)],
        jnp.take_along_axis(history, hist_idx[..., None], axis=1),
    )
    return out, _scatter_slots(conv_state, slots, new.reshape(reqs, *conv_state.shape[1:]))


def _decode_step(state, q, k, v, g, beta):
    """One gated delta step for every request: state [B, H, K, V]."""
    state = state * jnp.exp(g)[..., None, None]
    kv_mem = jnp.einsum("bhkv,bhk->bhv", state, k)
    delta = (v - kv_mem) * beta[..., None]
    state = state + k[..., None] * delta[..., None, :]
    return state, jnp.einsum("bhkv,bhk->bhv", state, q)


# The chunked recurrence, from SGLang-JAX's gated_delta.py.


def _solve_unit_lower_triangular(A, b):
    """Solve (I + A) x = b where A is strictly lower triangular [BT, BT]."""
    BT = b.shape[0]
    BS = 16
    num_blocks = BT // BS
    A = A.astype(jnp.float32)
    blocks = list(jnp.split(b.astype(jnp.float32), num_blocks, axis=0))
    for i in range(num_blocks):
        start, end = i * BS, (i + 1) * BS
        A_ii = A[start:end, start:end]
        rows = [blocks[i][r] for r in range(BS)]
        for j in range(1, BS):
            correction = jax.lax.dot_general(
                A_ii[j, :j][None, :],
                jnp.stack(rows[:j]),
                (((1,), (0,)), ((), ())),
                preferred_element_type=jnp.float32,
            ).squeeze(axis=0)
            rows[j] = rows[j] - correction
        blocks[i] = jnp.stack(rows)
        if i < num_blocks - 1:
            x_rest = jnp.concatenate(blocks[i + 1 :], axis=0)
            update = jax.lax.dot_general(
                A[end:, start:end],
                blocks[i],
                (((1,), (0,)), ((), ())),
                preferred_element_type=jnp.float32,
            )
            for idx, nb in enumerate(jnp.split(x_rest - update, num_blocks - 1 - i, axis=0)):
                blocks[i + 1 + idx] = nb
    return jnp.concatenate(blocks, axis=0)


def _chunk_intra(q, k, v, beta, g):
    """Intra-chunk solve for one (chunk, head) block of BT tokens."""
    BT = q.shape[0]
    gamma = jnp.cumsum(g)
    causal = jnp.tril(jnp.ones((BT, BT), dtype=bool))
    strict = jnp.tril(jnp.ones((BT, BT), dtype=bool), k=-1)
    decay = jnp.exp(jnp.where(causal, gamma[:, None] - gamma[None, :], -126.0))
    L = (k @ k.T) * decay * beta[:, None] * strict
    A_qk = (q @ k.T) * decay * causal
    safe_gamma = jnp.maximum(gamma, -126.0)
    rhs = jnp.concatenate(
        [v * beta[:, None], k * jnp.exp(safe_gamma)[:, None] * beta[:, None]], axis=-1
    )
    sol = _solve_unit_lower_triangular(L, rhs)
    d_v = v.shape[-1]
    u, w = sol[:, :d_v], sol[:, d_v:]
    k_g = k * jnp.exp(jnp.maximum(gamma[-1] - gamma, -126.0))[:, None]
    q_g = q * jnp.exp(safe_gamma)[:, None]
    return u, w, k_g, q_g, A_qk, gamma[-1]


def _chunked_rule(q, k, v, beta, g, init_state, starts):
    """Chunkwise-parallel recurrence over packed requests.

    q, k [T, H, d_k], v [T, H, d_v], beta, g [T, H] (zero past starts[-1]);
    init_state [B, H, d_k, d_v]. Returns (final states [B, H, d_k, d_v],
    output [T, H, d_v]).
    """
    tokens, heads, d_k = q.shape
    d_v = v.shape[-1]
    reqs = starts.shape[0] - 1
    lens = starts[1:] - starts[:-1]
    padded_lens = (lens + CHUNK - 1) // CHUNK * CHUNK
    padded_starts = jnp.concatenate([jnp.zeros(1, jnp.int32), jnp.cumsum(padded_lens)])
    aligned = (tokens + reqs * (CHUNK - 1) + CHUNK - 1) // CHUNK * CHUNK
    chunks = aligned // CHUNK

    # Each aligned position's source token, or `tokens` (a zero row) for padding.
    j = jnp.arange(aligned, dtype=jnp.int32)
    j_req = jnp.clip(jnp.sum(j[:, None] >= padded_starts[None, 1:], axis=1), 0, reqs - 1)
    offset = j - padded_starts[j_req]
    gather = jnp.where(offset < lens[j_req], starts[j_req] + offset, tokens)

    def align(x):
        x = jnp.pad(x, ((0, 1),) + ((0, 0),) * (x.ndim - 1))[gather]
        return jnp.moveaxis(x.reshape(chunks, CHUNK, *x.shape[1:]), 2, 1)

    u, w, k_g, q_g, A_qk, gamma_end = jax.vmap(jax.vmap(_chunk_intra))(
        align(q), align(k), align(v), align(beta), align(g)
    )

    chunk_starts = jnp.arange(chunks, dtype=jnp.int32) * CHUNK
    chunk_req = jnp.clip(jnp.sum(chunk_starts[:, None] >= padded_starts[None, 1:], axis=1), 0, reqs - 1)
    first = chunk_starts == padded_starts[chunk_req]

    # Carry each request's state across its chunk boundaries. The chunk count
    # is static and small, so the loop unrolls: TT traces don't run inside a
    # while loop's body.
    states = init_state
    state_in, v_new = [], []
    for c in range(chunks):
        r = chunk_req[c]
        s_in = jnp.where(first[c], init_state[r], states[r])
        v_c = u[c] - jnp.einsum("htk,hkv->htv", w[c], s_in)
        s_out = s_in * jnp.exp(gamma_end[c])[:, None, None] + jnp.einsum(
            "htk,htv->hkv", k_g[c], v_c
        )
        states = states.at[r].set(s_out)
        state_in.append(s_in)
        v_new.append(v_c)
    final, state_in, v_new = states, jnp.stack(state_in), jnp.stack(v_new)
    out = jnp.einsum("chtk,chkv->chtv", q_g, state_in) + jnp.einsum(
        "chts,chsv->chtv", A_qk, v_new
    )
    out = jnp.moveaxis(out, 1, 2).reshape(aligned, heads, d_v)

    # Back to packed order; a token past starts[-1] gets zeros.
    t = jnp.arange(tokens, dtype=jnp.int32)
    t_req = jnp.clip(jnp.sum(t[:, None] >= starts[None, 1:], axis=1), 0, reqs - 1)
    # Clamped: a padding token's index can run past the aligned buffer.
    out = out[jnp.clip(padded_starts[t_req] + t - starts[t_req], 0, aligned - 1)]
    out = jnp.where((t < starts[-1])[:, None, None], out, 0)
    # A request that holds no tokens keeps its state.
    final = jnp.where((lens > 0)[:, None, None, None], final, init_state)
    return final, out


def fused_conv1d_gdn(
    qkv,
    b,
    a,
    conv_state,
    recurrent_state,
    conv_weight,
    conv_bias,
    a_log,
    dt_bias,
    query_start_loc,
    state_indices,
    distribution,
    seq_lens,
    read_offsets=None,
    *,
    n_kq,
    n_v,
    d_k,
    d_v,
    kernel_size,
    attention_mode=None,
    gate_lower_bound=None,
    batched_only=False,
    prefill_only=False,
    packed_weights=None,
    track_indices=None,
    track_end=None,
    num_spec_tokens=0,
):
    """`fused_conv1d_gdn` of SGLang-TorchTPU's GDN kernel package, in JAX."""
    if getattr(attention_mode, "name", "GDN") != "GDN" or gate_lower_bound is not None:
        raise NotImplementedError("KDA attention needs the Pallas kernel")
    if packed_weights is not None or track_indices is not None:
        raise NotImplementedError(
            "packed weights and prefix-cache snapshots need the Pallas kernel"
        )
    if num_spec_tokens or read_offsets is not None:
        raise NotImplementedError("speculative verify windows need the Pallas kernel")
    if batched_only == prefill_only:
        raise NotImplementedError("mixed decode and extend batches need the Pallas kernel")
    if conv_state.shape[1] != kernel_size - 1 or conv_state.shape[-1] != qkv.shape[1]:
        raise NotImplementedError(f"conv state of shape {conv_state.shape}")

    tokens = qkv.shape[0]
    reqs = state_indices.shape[0]
    live = jnp.arange(reqs, dtype=jnp.int32) < distribution[2]
    slots = jnp.where(live, state_indices, 0)
    starts = jnp.minimum(query_start_loc, query_start_loc[distribution[2]])
    lens = starts[1:] - starts[:-1]
    has_init = (seq_lens - lens) > 0
    beta, g = _gates(b, a, a_log, dt_bias)

    if batched_only:
        if tokens != reqs:
            raise NotImplementedError("decode batches of one token per request only")
        mixed, conv_state = _conv_decode(qkv, conv_state, conv_weight, conv_bias, slots, has_init)
        q, k, v = _split_heads(jax.nn.silu(mixed), n_kq, n_v, d_k, d_v)
        state = jnp.where(has_init[:, None, None, None], recurrent_state[slots], 0)
        state, out = _decode_step(state.astype(jnp.float32), q, k, v, g, beta)
    else:
        mixed, conv_state = _conv_extend(
            qkv, conv_state, conv_weight, conv_bias, starts, slots, has_init
        )
        q, k, v = _split_heads(jax.nn.silu(mixed), n_kq, n_v, d_k, d_v)
        valid = (jnp.arange(tokens, dtype=jnp.int32) < starts[-1])[:, None]
        init = jnp.where(has_init[:, None, None, None], recurrent_state[slots], 0)
        state, out = _chunked_rule(
            jnp.where(valid[..., None], q, 0),
            jnp.where(valid[..., None], k, 0),
            jnp.where(valid[..., None], v, 0),
            jnp.where(valid, beta, 0),
            jnp.where(valid, g, 0),
            init.astype(jnp.float32),
            starts,
        )
    recurrent_state = _scatter_slots(recurrent_state, slots, state)
    return (conv_state, recurrent_state), out.astype(qkv.dtype).reshape(tokens, n_v * d_v)
