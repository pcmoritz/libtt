"""TorchTitan's TPU splash attention on Tenstorrent.

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
"""

SPLASH_MODULE = "torchtitan.experiments.tpu.kernels.splash_attention"
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


PATCHES = {SPLASH_MODULE: _patch_splash_module}
