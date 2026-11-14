"""TorchTitan's TPU splash attention and fused loss on Tenstorrent.

TorchTitan's TPU trainer (google-pytorch/torchtitan, experiments/tpu) runs its
models' attention as splash attention, a Pallas TPU kernel, through torch_tpu's
JAX bridge. It builds the kernel's forward and backward in
`_make_splash_attention_fn`: the forward returns the output and each query
row's log-sum-exp of the scaled scores, and the backward takes them back with
the output gradient instead of recomputing the forward. In TPU compatibility
mode, `tpu_compat` replaces that function with one whose forward and backward
run tt-train's fused attention (`tt.sdpa_fw` and `tt.sdpa_bw`,
ttml::metal::sdpa_fw and sdpa_bw), which save the same values: the
log-sum-exp is the first column of tt-train's [B, H, S, 32] FP32
intermediates.

A plain causal attention uses the kernels' causal mask. Packed documents
(segment IDs: a query attends only to keys of its own document) and sliding
windows pass the kernels a [1, 1, S, S] mask, one sequence at a time.

The trainer's fused output layer and loss (`--loss_kernel.use_loss_kernel`) is
tokamax's linear softmax cross-entropy, also a Pallas TPU kernel, which never
materializes the [tokens, vocabulary] logits. Here it runs a chunk of tokens at
a time: the chunk's BF16 logits, as the output layer computes them anyway, go
through tt-train's fused cross-entropy (`tt.cross_entropy_fw` and
`tt.cross_entropy_bw`, FP32 inside), so no FP32 copy of all the logits exists.
"""

SPLASH_MODULE = "torchtitan.experiments.tpu.kernels.splash_attention"
LOSS_KERNEL_MODULE = "tokamax._src.ops.linear_softmax_cross_entropy_loss.pallas_mosaic_tpu_kernel"
# Tokens per cross-entropy call: a chunk's BF16 logits and their gradient take
# 1.2 GB at Qwen3's 151,936-token vocabulary.
LOSS_TOKEN_CHUNK = 2048
# Vocabulary columns per FP32 weight-gradient accumulator: 537 MB at a hidden
# size of 4096, where one [4096, 151936] accumulator needs a free 311 MB block in
# each of a Blackhole's 8 DRAM banks, more than a busy chip has.
LOSS_WEIGHT_COLUMNS = 32768
# ttml::metal::AttentionMaskType.
_NO_MASK, _CAUSAL_MASK, _ARBITRARY_MASK = 0, 1, 2
# Columns of tt-train's intermediates; the log-sum-exp is the first.
_INTERMEDIATE_COLUMNS = 32


def _attention_mask(seq_len, is_causal, local_window_size, segment_ids):
    """[1, 1, S, S] BF16, 1 where a query attends to a key: causal, within the
    window and within the query's document (`segment_ids` [S] or None)."""
    import jax.numpy as jnp

    rows = jnp.arange(seq_len)[:, None]
    cols = jnp.arange(seq_len)[None, :]
    keep = jnp.ones((seq_len, seq_len), bool)
    if is_causal or local_window_size is not None:
        keep &= cols <= rows
    if local_window_size is not None:
        keep &= cols >= rows - local_window_size
    if segment_ids is not None:
        keep &= segment_ids[:, None] == segment_ids[None, :]
    return keep.astype(jnp.bfloat16)[None, None]


def _make_splash_attention_fn(
    seq_len, n_heads, n_kv_heads, is_causal=True, local_window_size=None, kv_seq_len=None, **_
):
    """`_make_splash_attention_fn` of TorchTitan's splash attention, without Pallas.

    The caller scales the queries for a scale other than 1 / sqrt(head_dim),
    which the kernels apply.
    """
    import jax
    import jax.numpy as jnp

    if kv_seq_len not in (None, seq_len):
        # Context parallelism: each rank's queries attend to all ranks' keys.
        raise NotImplementedError(
            f"splash attention with context parallelism ({seq_len} queries, {kv_seq_len} keys) "
            "isn't supported on Tenstorrent devices"
        )

    def call(target, outputs, tensors, segment_ids):
        """`target` on the batch with the kernels' own mask, or on each sequence
        with its mask (packed documents, sliding window)."""
        if segment_ids is None and local_window_size is None:
            mask_type = _CAUSAL_MASK if is_causal else _NO_MASK
            return jax.ffi.ffi_call(target, outputs(1, *tensors))(*tensors, mask_type=mask_type)
        results = []
        for b in range(tensors[0].shape[0]):
            sequence = [t[b : b + 1] for t in tensors]
            mask = _attention_mask(
                seq_len, is_causal, local_window_size, None if segment_ids is None else segment_ids[b]
            )
            results.append(
                jax.ffi.ffi_call(target, outputs(1, *sequence))(*sequence, mask, mask_type=_ARBITRARY_MASK)
            )
        return tuple(jnp.concatenate(parts) for parts in zip(*results))

    @jax.jit
    def splash_fn(q, k, v, segment_ids=None):
        """Forward: (out, logsumexp) for the whole batch."""
        # Multi-query attention passes K and V without their head dimension.
        if k.ndim == 3:
            k, v = k[:, None], v[:, None]

        def outputs(_, q, k, v):
            return (
                jax.ShapeDtypeStruct(q.shape, q.dtype),
                jax.ShapeDtypeStruct((*q.shape[:3], _INTERMEDIATE_COLUMNS), jnp.float32),
            )

        out, intermediates = call("tt.sdpa_fw", outputs, (q, k, v), segment_ids)
        return out, intermediates[..., 0]

    @jax.jit
    def splash_bwd_fn(q, k, v, out, logsumexp, g, segment_ids=None):
        """Backward from the saved (out, logsumexp), without recomputing the forward."""
        squeezed = k.ndim == 3
        if squeezed:
            k, v = k[:, None], v[:, None]
        intermediates = jnp.broadcast_to(logsumexp[..., None], (*logsumexp.shape, _INTERMEDIATE_COLUMNS))

        def outputs(_, g, out, q, k, v, intermediates):
            return tuple(jax.ShapeDtypeStruct(t.shape, t.dtype) for t in (q, k, v))

        dq, dk, dv = call("tt.sdpa_bw", outputs, (g, out, q, k, v, intermediates), segment_ids)
        if squeezed:
            dk, dv = dk[:, 0], dv[:, 0]
        return dq, dk, dv

    return splash_fn, splash_bwd_fn


def _patch_splash_module(module):
    module._make_splash_attention_fn = _make_splash_attention_fn


def _token_chunks(tokens):
    return [(start, min(LOSS_TOKEN_CHUNK, tokens - start)) for start in range(0, tokens, LOSS_TOKEN_CHUNK)]


def linear_cross_entropy_fwd(x, labels, w, *, reduction="sum", **_):
    """(loss, logsumexp [B] FP32) of softmax cross-entropy over the logits x [B, H] @ w [H, V], a chunk of
    tokens at a time; tokamax's `linear_softmax_cross_entropy_loss_fwd_pallas_mosaic_tpu`."""
    import jax
    import jax.numpy as jnp

    losses, logsumexps = [], []
    for start, size in _token_chunks(x.shape[0]):
        logits = jnp.dot(x[start : start + size], w)
        targets = labels[start : start + size]
        loss = jax.ffi.ffi_call("tt.cross_entropy_fw", jax.ShapeDtypeStruct((1, 1, size, 1), logits.dtype))(
            logits[None, None], targets[None]
        )
        loss = loss.reshape(size).astype(jnp.float32)
        target_logits = jnp.take_along_axis(logits, targets[:, None], axis=1)[:, 0].astype(jnp.float32)
        losses.append(loss)
        logsumexps.append(loss + target_logits)
    losses, logsumexp = jnp.concatenate(losses), jnp.concatenate(logsumexps)
    if reduction == "sum":
        return jnp.sum(losses), logsumexp
    if reduction == "mean":
        return jnp.mean(losses), logsumexp
    return losses, logsumexp


def linear_cross_entropy_bwd(dout, lse, x, labels, w, *, reduction="sum", **_):
    """(dx, dw) of linear_cross_entropy_fwd; tt-train's backward recomputes each chunk's softmax from its
    logits, so the saved logsumexp is unused. The weight gradient adds up chunk by chunk in FP32,
    LOSS_WEIGHT_COLUMNS vocabulary columns at a time."""
    import jax
    import jax.numpy as jnp

    del lse
    tokens = x.shape[0]
    # The kernel scales by a [1, 1, 1, 1] gradient; per-token gradients scale its result.
    per_token = reduction == "none"
    grad = jnp.ones((1, 1, 1, 1), x.dtype) if per_token else dout.reshape(1, 1, 1, 1).astype(x.dtype)
    scaler = 1.0 / tokens if reduction == "mean" else 1.0
    columns = [(start, min(LOSS_WEIGHT_COLUMNS, w.shape[1] - start)) for start in range(0, w.shape[1], LOSS_WEIGHT_COLUMNS)]
    dx, dw = [], [None] * len(columns)
    for start, size in _token_chunks(tokens):
        x_chunk = x[start : start + size]
        logits = jnp.dot(x_chunk, w)
        dlogits = jax.ffi.ffi_call("tt.cross_entropy_bw", jax.ShapeDtypeStruct((1, 1, size, w.shape[1]), logits.dtype))(
            logits[None, None], labels[start : start + size][None], grad, scaler=scaler
        )[0, 0]
        if per_token:
            dlogits = dlogits * dout[start : start + size, None].astype(dlogits.dtype)
        dx.append(jnp.dot(dlogits, w.T).astype(x.dtype))
        for i, (column, width) in enumerate(columns):
            part = jnp.dot(x_chunk.T, dlogits[:, column : column + width], preferred_element_type=jnp.float32)
            dw[i] = part if dw[i] is None else dw[i] + part
    return jnp.concatenate(dx), jnp.concatenate([part.astype(w.dtype) for part in dw], axis=1)


def _patch_loss_kernel_module(module):
    module.linear_softmax_cross_entropy_loss_fwd_pallas_mosaic_tpu = linear_cross_entropy_fwd
    module.linear_softmax_cross_entropy_loss_bwd_pallas_mosaic_tpu = linear_cross_entropy_bwd


PATCHES = {SPLASH_MODULE: _patch_splash_module, LOSS_KERNEL_MODULE: _patch_loss_kernel_module}
