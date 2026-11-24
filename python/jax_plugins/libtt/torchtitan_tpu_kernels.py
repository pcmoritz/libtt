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

miles' torchtitan trainer (radixark/miles, `miles.kernels.tpu.flash_attention`)
calls splash attention's forward and backward kernels itself, on packed
sequences (segment IDs); its `_forward` and `_backward` run the same functions.

A plain causal attention uses the kernels' causal mask. Packed documents
(segment IDs: a query attends only to keys of its own document) and sliding
windows pass the kernels a [1, 1, S, S] mask, one sequence at a time. Causal
ones keep, in each query row, one interval of keys ending at the query, whose
start never decreases down the rows (TorchTitan's segment IDs never decrease,
as its splash attention's block skipping assumes): the kernels skip the mask's
tiles that keep no score.

The trainer's fused output layer and loss (`--loss_kernel.use_loss_kernel`) is
tokamax's linear softmax cross-entropy, also a Pallas TPU kernel, which never
materializes the [tokens, vocabulary] logits. Here it runs a chunk of tokens at
a time: the chunk's BF16 logits, as the output layer computes them anyway, go
through tt-train's fused cross-entropy (`tt.cross_entropy_fw` and
`tt.cross_entropy_bw`, FP32 inside), so no FP32 copy of all the logits exists.

Qwen3.5's GatedDeltaNet normalizes its output with a gated RMSNorm, a Pallas
TPU kernel where its inputs allow; here it always takes TorchTitan's PyTorch
path.

Llama 3's rotary embedding (TorchTitan's `ComplexRoPE`, and the TPU recipes'
real-arithmetic version of it) rotates adjacent pairs of each head's columns:
it views the queries and keys as [..., D/2, 2] in FP32. In 32 x 32 tiles the
trailing pair takes 32 columns, so each such tensor is 16 times its size: 2 GB
for the queries of an 8192-token sequence, of which a layer's forward and
backward make several. `apply_complex_rope` splits the head into its even and
odd columns instead, with a BF16 permutation matmul, rotates them in FP32 as
TorchTitan does, and interleaves the BF16 results back with another: the same
values and gradients, bit for bit.
"""

import math

SPLASH_MODULE = "torchtitan.experiments.tpu.kernels.splash_attention"
LOSS_KERNEL_MODULE = "tokamax._src.ops.linear_softmax_cross_entropy_loss.pallas_mosaic_tpu_kernel"
RMS_NORM_GATED_MODULE = "torchtitan.experiments.tpu.kernels.rms_norm_gated"
ROPE_MODULE = "torchtitan.models.common.rope"
REAL_ROPE_MODULE = "torchtitan.experiments.tpu.overrides.real_rope"
MILES_ATTENTION_MODULE = "miles.kernels.tpu.flash_attention"
# Tokens per cross-entropy call: a chunk's BF16 logits and their gradient take
# 1.2 GB at Qwen3's 151,936-token vocabulary.
LOSS_TOKEN_CHUNK = 2048
# Vocabulary columns per FP32 weight-gradient accumulator: 537 MB at a hidden
# size of 4096, where one [4096, 151936] accumulator needs a free 311 MB block in
# each of a Blackhole's 8 DRAM banks, more than a busy chip has.
LOSS_WEIGHT_COLUMNS = 32768
# ttml::metal::AttentionMaskType.
_NO_MASK, _CAUSAL_MASK, _ARBITRARY_MASK, _CAUSAL_INTERVALS_MASK = 0, 1, 2, 3
# Columns of tt-train's intermediates; the log-sum-exp is the first.
_INTERMEDIATE_COLUMNS = 32
# tt-train's attention takes head dimensions in tiles of 32 columns.
_HEAD_DIM_TILE = 32


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
        # A sliding window is causal too.
        mask_type = _CAUSAL_INTERVALS_MASK if is_causal or local_window_size is not None else _ARBITRARY_MASK
        for b in range(tensors[0].shape[0]):
            sequence = [t[b : b + 1] for t in tensors]
            mask = _attention_mask(
                seq_len, is_causal, local_window_size, None if segment_ids is None else segment_ids[b]
            )
            results.append(
                jax.ffi.ffi_call(target, outputs(1, *sequence))(*sequence, mask, mask_type=mask_type)
            )
        return tuple(jnp.concatenate(parts) for parts in zip(*results))

    def pad_head_dim(q, *tensors):
        """Zero-pads the head dimension to whole tiles: zero columns don't change the scores, and q scaled by
        sqrt(padded / head_dim) keeps the kernels' 1 / sqrt(padded) scale at 1 / sqrt(head_dim). Returns that
        factor too, 1 without padding."""
        head_dim = q.shape[-1]
        padding = -head_dim % _HEAD_DIM_TILE
        if not padding:
            return 1.0, (q, *tensors)
        factor = ((head_dim + padding) / head_dim) ** 0.5
        widths = [(0, 0)] * (q.ndim - 1) + [(0, padding)]
        return factor, tuple(jnp.pad(t, widths) for t in ((q * factor).astype(q.dtype), *tensors))

    @jax.jit
    def splash_fn(q, k, v, segment_ids=None):
        """Forward: (out, logsumexp) for the whole batch."""
        head_dim = q.shape[-1]
        # Multi-query attention passes K and V without their head dimension.
        if k.ndim == 3:
            k, v = k[:, None], v[:, None]
        _, (q, k, v) = pad_head_dim(q, k, v)

        def outputs(_, q, k, v):
            return (
                jax.ShapeDtypeStruct(q.shape, q.dtype),
                jax.ShapeDtypeStruct((*q.shape[:3], _INTERMEDIATE_COLUMNS), jnp.float32),
            )

        out, intermediates = call("tt.sdpa_fw", outputs, (q, k, v), segment_ids)
        return out[..., :head_dim], intermediates[..., 0]

    @jax.jit
    def splash_bwd_fn(q, k, v, out, logsumexp, g, segment_ids=None):
        """Backward from the saved (out, logsumexp), without recomputing the forward."""
        head_dim = q.shape[-1]
        squeezed = k.ndim == 3
        if squeezed:
            k, v = k[:, None], v[:, None]
        factor, (q, k, v, out, g) = pad_head_dim(q, k, v, out, g)
        intermediates = jnp.broadcast_to(logsumexp[..., None], (*logsumexp.shape, _INTERMEDIATE_COLUMNS))

        def outputs(_, g, out, q, k, v, intermediates):
            return tuple(jax.ShapeDtypeStruct(t.shape, t.dtype) for t in (q, k, v))

        dq, dk, dv = call("tt.sdpa_bw", outputs, (g, out, q, k, v, intermediates), segment_ids)
        # The kernels saw q times the padding factor.
        dq = (dq[..., :head_dim] * factor).astype(dq.dtype) if factor != 1.0 else dq
        dk, dv = dk[..., :head_dim], dv[..., :head_dim]
        if squeezed:
            dk, dv = dk[:, 0], dv[:, 0]
        return dq, dk, dv

    return splash_fn, splash_bwd_fn


def _patch_splash_module(module):
    module._make_splash_attention_fn = _make_splash_attention_fn


def _patch_miles_attention_module(module):
    """miles' torchtitan trainer calls splash attention's forward and backward
    kernels itself (`_forward` and `_backward`, on [B, H, S, D] queries it
    scales by 1 / sqrt(head_dim) first, with each token's segment ID). Here
    they run as TorchTitan's: the same outputs, log-sum-exp and gradients with
    respect to the unscaled queries."""

    def forward(q, k, v, segment_ids, *, interpret=False):
        splash_fn, _ = _make_splash_attention_fn(q.shape[2], q.shape[1], k.shape[1])
        return splash_fn(q, k, v, segment_ids)

    def backward(q, k, v, out, lse, grad_out, segment_ids, *, interpret=False):
        _, splash_bwd_fn = _make_splash_attention_fn(q.shape[2], q.shape[1], k.shape[1])
        return splash_bwd_fn(q, k, v, out, lse, grad_out, segment_ids)

    module._forward = forward
    module._backward = backward


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
        # TorchTitan's fused loss zeroes the hidden state of each ignored token (label -100) and subtracts
        # log(vocabulary) per such row from the sum, as the TPU kernel returns that row's loss, uniform logits,
        # exactly in FP32. tt-train's kernel returns BF16 losses, which round it (log(248,320) = 12.4225 becomes
        # 12.4375), and with most tokens ignored in chat data the leftover moved the logged loss by 0.006 to 0.2.
        zero_rows = jnp.all(x[start : start + size] == 0, axis=-1)
        loss = jnp.where(zero_rows, jnp.float32(math.log(w.shape[-1])), loss)
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


def _patch_rms_norm_gated_module(module):
    module.supports = lambda *args, **kwargs: False


def _split_pairs(dim, dtype, device):
    """[dim, dim] permutation: x @ it = [x0, x2, ..., x(dim-2) | x1, x3, ..., x(dim-1)]."""
    import torch

    rows = torch.arange(dim, device=device)[:, None]
    columns = torch.arange(dim, device=device)[None, :]
    half = dim // 2
    return (rows == torch.where(columns < half, 2 * columns, 2 * (columns - half) + 1)).to(dtype)


def _complex_rope(original):
    """`apply_rotary_emb` of TorchTitan's complex RoPE for BF16 queries and keys: pairs (x[2i], x[2i+1]) rotated by
    the cache's (cos, sin), in FP32, as `original` does, without [..., D/2, 2] tensors. The permutation matmuls only
    move values (one nonzero product per output), so they are exact in BF16; other types, and tensor-parallel
    DTensors, keep `original`."""

    def apply_complex_rope(query, key, rope_cache, *, inverse=False):
        import torch
        from torch.distributed.tensor import DTensor

        tensors = (query,) if key is None else (query, key)
        if any(t.dtype != torch.bfloat16 or isinstance(t, DTensor) for t in tensors):
            return original(query, key, rope_cache, inverse=inverse)
        cache = torch.view_as_real(rope_cache)
        # The inverse rotation multiplies by the conjugate: (cos, -sin).
        cos, sin = cache[..., 0], -cache[..., 1] if inverse else cache[..., 1]

        def rotate(x):
            half = x.shape[-1] // 2
            split = _split_pairs(x.shape[-1], x.dtype, x.device)
            halves = x @ split
            real, imag = halves[..., :half].float(), halves[..., half:].float()
            rotated = torch.cat([(real * cos - imag * sin).type_as(x), (real * sin + imag * cos).type_as(x)], dim=-1)
            return rotated @ split.T

        if key is None:
            return rotate(query)
        return rotate(query), rotate(key)

    return staticmethod(apply_complex_rope)


def _patch_rope_module(module):
    module.ComplexRoPE.apply_rotary_emb = _complex_rope(module.ComplexRoPE.apply_rotary_emb)


def _patch_real_rope_module(module):
    rope = module.TpuRealArithmeticComplexRoPE
    rope.apply_rotary_emb = _complex_rope(rope.apply_rotary_emb)


PATCHES = {
    SPLASH_MODULE: _patch_splash_module,
    LOSS_KERNEL_MODULE: _patch_loss_kernel_module,
    RMS_NORM_GATED_MODULE: _patch_rms_norm_gated_module,
    ROPE_MODULE: _patch_rope_module,
    REAL_ROPE_MODULE: _patch_real_rope_module,
    MILES_ATTENTION_MODULE: _patch_miles_attention_module,
}
