"""SGLang-TorchTPU's fused Conv1D + gated delta rule kernel on libtt's recurrent kernels.

Hybrid models such as Qwen3.5 run their gated DeltaNet layers through
`sglang_tpu.attention.pallas.gdn.fused_conv1d_gdn`, a Pallas TPU kernel. In
TPU compatibility mode, `sglang_tpu_kernels` replaces it with
`fused_conv1d_gdn` below, which runs the kernels SGLang-JAX's Tenstorrent
backend runs (`sgl_jax/srt/hardware_backend/tt/attention/gdn_backend.py`):

- a decode batch: `tt.causal_conv1d_update` and `tt.gated_delta_decode`, one
  step per request, on the state pools in place;
- a speculative verify batch (`num_spec_tokens`): the same kernels taking one
  step per token of each request's window, the states after each step to the
  window's checkpoint slots;
- an extend batch: the convolution in JAX and `tt.gated_delta_rule` over each
  request's tokens, the new states to the pools with `tt.state_pool_update`.

The convolution kernel keeps a request's last three inputs as a BF16
`[channels, 3]` slot, so `patch_mamba_pool` gives the pool that layout
(SGLang-TorchTPU allocates `[slots, 3, 1, channels]` in FP32). The inputs are
BF16 already, so the BF16 window loses nothing. The recurrent pool keeps
SGLang-TorchTPU's `[slots, n_v, d_k, d_v]` FP32 layout, the kernels' own.
Slot 0 is the reserved null slot of padding rows: the kernels never write it,
and nothing reads it unmasked.
"""

# Rows per call of the decode kernels and of TTNN's state pool update.
MAX_ROWS = 32


def _l2norm(x, eps=1e-6):
    import jax.numpy as jnp

    x = x.astype(jnp.float32)
    return x / jnp.sqrt(jnp.sum(x * x, axis=-1, keepdims=True) + eps)


def _recurrent_call(name, state, output, *operands):
    import jax

    return jax.ffi.ffi_call(
        name,
        (jax.ShapeDtypeStruct(state.shape, state.dtype), output),
        input_output_aliases={0: 0},
        vmap_method="sequential",
    )(state, *operands)


def _decode(conv, recurrent, qkv, b, a, weight, a_log, dt_bias, slots, initial):
    """One step for each request, a row of qkv, b and a, from and to its slot
    (from zeros without initial). Returns both pools and the output, [B, n_v, d_v]."""
    import jax
    import jax.numpy as jnp

    outputs = []
    for start in range(0, slots.shape[0], MAX_ROWS):
        rows = slice(start, start + MAX_ROWS)
        conv, convolved = _recurrent_call(
            "tt.causal_conv1d_update",
            conv,
            jax.ShapeDtypeStruct(qkv[rows].shape, jnp.float32),
            qkv[rows],
            weight,
            slots[rows],
            initial[rows],
        )
        # The kernel reads q, k and v as heads of the convolution output, and
        # normalizes and scales q and k.
        recurrent, out = _recurrent_call(
            "tt.gated_delta_decode",
            recurrent,
            jax.ShapeDtypeStruct((*b[rows].shape, recurrent.shape[-1]), jnp.float32),
            convolved,
            b[rows],
            a[rows],
            a_log,
            dt_bias,
            slots[rows],
            initial[rows],
        )
        outputs.append(out)
    return conv, recurrent, jnp.concatenate(outputs)


def _verify_windows(conv, recurrent, qkv, b, a, weight, a_log, dt_bias, starts, slots, reads, initial, window, dtype):
    """Speculative verify windows of up to `window` tokens per request, packed
    in qkv from starts[i] to starts[i + 1].

    Request i reads its states from slot reads[i] (zeros without initial) and
    writes the states after its token t to slot slots[i] + t, so that the next
    step can continue from any accepted prefix. Returns both pools and the
    output, [T, n_v, d_v].
    """
    import jax
    import jax.numpy as jnp

    tokens = qkv.shape[0]
    batch = slots.shape[0]
    step = jnp.arange(window, dtype=jnp.int32)
    valid = step < (starts[1:] - starts[:-1])[:, None]
    targets = jnp.where(valid, slots[:, None] + step, 0).astype(jnp.int32)
    positions = jnp.minimum(starts[:-1, None] + step, tokens - 1).reshape(-1)
    qkv, b, a = (x[positions] for x in (qkv, b, a))
    # The kernels take up to 32 rows, a step of a request each.
    per_call = max(1, 32 // window)
    outputs = []
    for start in range(0, batch, per_call):
        reqs = slice(start, start + per_call)
        rows = slice(start * window, (start + per_call) * window)
        conv, convolved = _recurrent_call(
            "tt.causal_conv1d_update",
            conv,
            jax.ShapeDtypeStruct(qkv[rows].shape, jnp.float32),
            qkv[rows],
            weight,
            reads[reqs],
            initial[reqs],
            targets[reqs],
        )
        recurrent, out = _recurrent_call(
            "tt.gated_delta_decode",
            recurrent,
            jax.ShapeDtypeStruct((*b[rows].shape, recurrent.shape[-1]), jnp.float32),
            convolved,
            b[rows],
            a[rows],
            a_log,
            dt_bias,
            reads[reqs],
            initial[reqs],
            targets[reqs],
        )
        outputs.append(out)
    return conv, recurrent, _packed(jnp.concatenate(outputs), starts, valid, tokens, dtype)


def _packed(rows, starts, valid, tokens, dtype):
    """Request-major rows (row i * width + t: request i's token t, for the
    `valid` ones) back to packed order, at row starts[i] + t, in `dtype`. Rows
    no request's token lands on are zeros.

    One scatter: gathering each packed row's window row instead needs each
    token's request, a compare, sum and gather of its own on every layer. The
    rows take the layer output's type first: the gated RMS norm after the
    layer then reads them in it, while an FP32 scatter left a BF16 round trip
    in front of the norm that kept it from fusing.
    """
    import jax.numpy as jnp

    width = valid.shape[1]
    dest = jnp.where(valid, starts[:-1, None] + jnp.arange(width), tokens).reshape(-1)
    return jnp.zeros((tokens, *rows.shape[1:]), dtype).at[dest].set(rows.astype(dtype), mode="drop")


def _write_rows(pool, slots, rows):
    """pool[slots[i]] = rows[i], in place; slot 0 is skipped."""
    import jax

    update = jax.ffi.ffi_call(
        "tt.state_pool_update",
        jax.ShapeDtypeStruct(pool.shape, pool.dtype),
        input_output_aliases={0: 0},
        vmap_method="sequential",
    )
    for start in range(0, slots.shape[0], MAX_ROWS):
        part = slice(start, start + MAX_ROWS)
        pool = update(pool, slots[part], rows[part].astype(pool.dtype))
    return pool


def _extend(conv, recurrent, qkv, b, a, weight, a_log, dt_bias, starts, slots, initial, heads, dtype):
    """Each request's tokens in turn, packed in qkv from starts[i] to starts[i + 1].

    As SGLang-JAX's prefill on Tenstorrent: the convolution in JAX over the
    request's saved window and its tokens, then TTNN's chunked gated delta rule
    over each request's tokens as a row of `width` positions.
    """
    import jax
    import jax.numpy as jnp

    n_kq, n_v, d_k, d_v = heads
    tokens = qkv.shape[0]
    batch = slots.shape[0]
    taps = weight.shape[-1]
    lengths = starts[1:] - starts[:-1]
    width = tokens
    valid = jnp.arange(width) < lengths[:, None]

    def pack(value):
        if batch == 1:
            return value[None]
        positions = starts[:-1, None] + jnp.arange(width)
        return value.at[positions].get(mode="clip")

    def saved(pool):
        return jnp.where(initial.reshape((-1,) + (1,) * (pool.ndim - 1)) > 0, pool[slots], 0)

    history = jnp.concatenate((saved(conv).swapaxes(-1, -2), pack(qkv)), axis=-2)
    # Summed in FP32.
    convolved = jax.nn.silu(
        sum(
            history[:, tap : tap + width].astype(jnp.float32) * weight[:, tap].astype(jnp.float32)
            for tap in range(taps)
        )
    )
    tail = lengths[:, None] + jnp.arange(taps - 1)
    tail += jnp.arange(batch)[:, None] * history.shape[1]
    tail = history.reshape(-1, history.shape[-1]).at[tail].get(mode="clip")
    conv = _write_rows(conv, slots, tail.swapaxes(-1, -2))

    key_dim = n_kq * d_k
    qk = convolved[..., : 2 * key_dim].reshape(batch, width, 2 * n_kq, d_k)
    q, k = jnp.split(_l2norm(qk), 2, axis=-2)
    v = convolved[..., 2 * key_dim :].reshape(batch, width, n_v, d_v).astype(jnp.float32)
    beta = jax.nn.sigmoid(pack(b).astype(jnp.float32))
    gate = -jnp.exp(a_log.astype(jnp.float32)) * jax.nn.softplus(
        pack(a).astype(jnp.float32) + dt_bias.astype(jnp.float32)
    )
    # Zero beta and log decay keep the state past a request's tokens; zero
    # queries zero their output.
    q = jnp.where(valid[..., None, None], q * d_k**-0.5, 0)
    beta, gate = (jnp.where(valid[..., None], x, 0) for x in (beta, gate))
    # The kernel lets each q/k head serve its group of value heads.
    state, out = jax.ffi.ffi_call(
        "tt.gated_delta_rule",
        (
            jax.ShapeDtypeStruct((batch, n_v, d_k, d_v), jnp.float32),
            jax.ShapeDtypeStruct(v.shape, jnp.float32),
        ),
        vmap_method="sequential",
    )(q, k, v, gate, beta, saved(recurrent).astype(jnp.float32))
    recurrent = _write_rows(recurrent, slots, state)

    out = out.reshape(-1, n_v, d_v)
    if batch > 1:
        out = _packed(out, starts, valid, tokens, dtype)
    return conv, recurrent, out


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
    *_verify,  # a speculative verify's read offsets and checkpoints
    n_kq,
    n_v,
    d_k,
    d_v,
    kernel_size,
    attention_mode=None,
    gate_lower_bound=None,
    state_source=None,
    num_spec_tokens=0,
    batched_only=False,
    prefill_only=False,
    track_indices=None,
    packed_weights=None,
    **_options,  # the Pallas kernel's tiling and precision options
):
    """`fused_conv1d_gdn` of SGLang-TorchTPU's GDN kernel package, on libtt's kernels."""
    import jax.numpy as jnp

    if getattr(attention_mode, "name", "GDN") != "GDN" or gate_lower_bound is not None:
        raise NotImplementedError("KDA attention needs the Pallas kernel")
    if (d_k, d_v, kernel_size) != (128, 128, 4) or conv_bias is not None:
        raise NotImplementedError(
            "libtt's GDN kernels take 128-wide heads and a four-tap convolution without bias"
        )
    if num_spec_tokens and not (batched_only and _verify):
        raise NotImplementedError("speculative extend batches and verify without read offsets need the Pallas kernel")
    if any(x is not None for x in (state_source, track_indices, packed_weights)):
        raise NotImplementedError(
            "state sources, prefix-cache snapshots and packed weights need the Pallas kernel"
        )
    if batched_only == prefill_only:
        raise NotImplementedError("mixed decode and extend batches need the Pallas kernel")
    if conv_state.shape[1:] != (qkv.shape[1], kernel_size - 1):
        raise NotImplementedError(f"conv state of shape {conv_state.shape}; see patch_mamba_pool")

    tokens = qkv.shape[0]
    batch = state_indices.shape[0]
    live = jnp.arange(batch) < distribution[2]
    slots = jnp.where(live, state_indices, 0).astype(jnp.int32)
    starts = jnp.minimum(query_start_loc, query_start_loc[distribution[2]])
    initial = (live & (seq_lens - (starts[1:] - starts[:-1]) > 0)).astype(jnp.bfloat16)
    dtype = qkv.dtype
    qkv = qkv.astype(jnp.bfloat16)
    weight = conv_weight.reshape(qkv.shape[1], kernel_size).astype(jnp.bfloat16)
    if b.dtype != a.dtype:
        b, a = b.astype(jnp.float32), a.astype(jnp.float32)
    a_log, dt_bias = a_log.astype(jnp.float32), dt_bias.astype(jnp.float32)

    if num_spec_tokens:
        # A verify batch's windows read row state_indices + read_offsets.
        reads = jnp.where(live, slots + _verify[0].astype(jnp.int32), 0)
        conv_state, recurrent_state, out = _verify_windows(
            conv_state, recurrent_state, qkv, b, a, weight, a_log, dt_bias, starts, slots, reads, initial,
            num_spec_tokens + 1, dtype,
        )
    elif prefill_only:
        conv_state, recurrent_state, out = _extend(
            conv_state, recurrent_state, qkv, b, a, weight, a_log, dt_bias, starts, slots, initial,
            (n_kq, n_v, d_k, d_v), dtype,
        )
    elif tokens != batch:
        raise NotImplementedError("decode batches of one token per request")
    else:
        conv_state, recurrent_state, out = _decode(
            conv_state, recurrent_state, qkv, b, a, weight, a_log, dt_bias, slots, initial
        )
    return (conv_state, recurrent_state), out.astype(dtype).reshape(tokens, n_v * d_v)


def patch_mamba_pool(module):
    """Gives the recurrent-state pool's conv windows the BF16 `[slots, channels, 3]`
    layout of TTNN's causal conv update (SGLang-TorchTPU's: FP32 `[slots, 3, 1,
    channels]`, which tiles pad 32-fold)."""
    torch = module.torch
    pool = module.TpuMambaPool
    init = pool.__init__

    def __init__(self, **kwargs):
        init(self, **kwargs)
        conv = [
            [
                torch.zeros(
                    (state.shape[0], state.shape[-1], state.shape[1]),
                    dtype=torch.bfloat16,
                    device=state.device,
                )
                for state in layers
            ]
            for layers in self.mamba_cache.conv
        ]
        self.mamba_cache = module.TpuPerLayerState(conv=conv, temporal=self.mamba_cache.temporal)
        self.mem_usage = self.mamba_cache.mem_usage_bytes() / module.GB
        module.logger.info(
            "TT conv windows: per-layer conv=%s %s", list(conv[0][0].shape), conv[0][0].dtype
        )

    pool.__init__ = __init__
