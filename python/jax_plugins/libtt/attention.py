"""jax.nn.dot_product_attention through tt-train's fused attention kernels.

On the tt backend, `install()` routes supported calls of
`jax.nn.dot_product_attention` to tt-mlir's `tenstorrent.sdpa_fw` and
`tenstorrent.sdpa_bw` composites, under a custom VJP: the forward also returns
each row's log-sum-exp, and the backward recomputes the scores from it block
by block, so neither pass materializes the [B, N, T, S] scores. Each composite
carries its decomposition, which backends without the kernels run. Other calls
take JAX's own implementation. `LIBTT_FUSED_ATTENTION=0` turns the routing off.
"""

import functools
import math
import os

import jax
import jax.numpy as jnp
from jax import custom_batching

# tt-mlir's mask_type attribute.
_NO_MASK = 0
_CAUSAL = 1

# The forward's log-sum-exp intermediates are (B, N, T, 32) FP32: one tile
# column per row.
_LSE_WIDTH = 32
_TILE = 32


def _dot(spec, a, b):
    """einsum of `a` and `b` in their dtype, accumulated and returned in FP32."""
    return jnp.einsum(spec, a, b.astype(a.dtype), preferred_element_type=jnp.float32)


def _scores(q, k, causal):
    """FP32 scaled scores of q [B, N, T, D] against k [B, K, S, D], N = G * K."""
    k = jnp.repeat(k, q.shape[1] // k.shape[1], axis=1)
    s = _dot("bntd,bnsd->bnts", q, k) / math.sqrt(q.shape[-1])
    if causal:
        s = jnp.where(jnp.tril(jnp.ones(s.shape[-2:], bool)), s, -jnp.inf)
    return s


def _sum_groups(x, kv_heads):
    """Sums [B, N, S, D] over the query heads of each key head."""
    b, n, s, d = x.shape
    return x.reshape(b, kv_heads, n // kv_heads, s, d).sum(axis=2)


# The decompositions compute what the kernels do, with JAX's own precision:
# products of BF16 operands accumulated in FP32, and an FP32 softmax.
def _forward_decomposition(q, k, v, *, mask_type, dropout_probability):
    del dropout_probability
    s = _scores(q, k, mask_type == _CAUSAL)
    lse = jax.nn.logsumexp(s, axis=-1, keepdims=True)
    p = jnp.exp(s - lse).astype(q.dtype)
    out = _dot("bnts,bnsd->bntd", p, jnp.repeat(v, q.shape[1] // v.shape[1], axis=1))
    return out.astype(q.dtype), jnp.broadcast_to(lse, (*lse.shape[:-1], _LSE_WIDTH))


def _backward_decomposition(do, out, q, k, v, lse, *, mask_type, dropout_probability):
    del dropout_probability
    groups = q.shape[1] // k.shape[1]
    p = jnp.exp(_scores(q, k, mask_type == _CAUSAL) - lse[..., :1])
    dv = _dot("bnts,bntd->bnsd", p.astype(do.dtype), do)
    dp = _dot("bntd,bnsd->bnts", do, jnp.repeat(v, groups, axis=1))
    delta = jnp.sum(do.astype(jnp.float32) * out.astype(jnp.float32), axis=-1, keepdims=True)
    ds = (p * (dp - delta) / math.sqrt(q.shape[-1])).astype(q.dtype)
    dq = _dot("bnts,bnsd->bntd", ds, jnp.repeat(k, groups, axis=1))
    dk = _dot("bnts,bntd->bnsd", ds, q)
    return (
        dq.astype(q.dtype),
        _sum_groups(dk, k.shape[1]).astype(k.dtype),
        _sum_groups(dv, v.shape[1]).astype(v.dtype),
    )


def _fold_vmap_into_batch(fn):
    """`fn` over [B, ...] arrays, with any vmapped axis folded into B.

    Composites have no batching rule, and the kernels take one batch dim.
    """
    fn = custom_batching.custom_vmap(fn)

    @fn.def_vmap
    def _rule(axis_size, in_batched, *args):
        args = [x if batched else jnp.broadcast_to(x, (axis_size, *x.shape)) for x, batched in zip(args, in_batched)]
        outs = fn(*(x.reshape(axis_size * x.shape[1], *x.shape[2:]) for x in args))
        outs = tuple(y.reshape(axis_size, y.shape[0] // axis_size, *y.shape[1:]) for y in outs)
        return outs, (True,) * len(outs)

    return fn


@functools.cache
def _kernels(mask_type):
    forward = jax.lax.composite(_forward_decomposition, name="tenstorrent.sdpa_fw")
    backward = jax.lax.composite(_backward_decomposition, name="tenstorrent.sdpa_bw")
    attrs = {"mask_type": mask_type, "dropout_probability": 0.0}
    return (
        _fold_vmap_into_batch(lambda q, k, v: forward(q, k, v, **attrs)),
        _fold_vmap_into_batch(lambda *args: backward(*args, **attrs)),
    )


@functools.partial(jax.custom_vjp, nondiff_argnums=(3,))
def _attention(q, k, v, mask_type):
    return _kernels(mask_type)[0](q, k, v)[0]


def _attention_fwd(q, k, v, mask_type):
    out, lse = _kernels(mask_type)[0](q, k, v)
    return out, (out, q, k, v, lse)


def _attention_bwd(mask_type, residuals, do):
    out, q, k, v, lse = residuals
    return tuple(_kernels(mask_type)[1](do, out, q, k, v, lse))


_attention.defvjp(_attention_fwd, _attention_bwd)


def _fused(query, key, value, bias, mask, options):
    """Whether the kernels compute this call: bf16 [B, T, N, D] self-attention
    with T and D multiples of a tile, no mask other than a causal one and every
    other option at its default but an equivalent scale, on the tt backend."""
    options = dict(options)
    scale = options.pop("scale", None)
    if bias is not None or mask is not None or any(
        option is not None and option is not False for option in options.values()
    ):
        return False
    if query.ndim not in (3, 4) or key.ndim != query.ndim or value.shape != key.shape:
        return False
    if any(x.dtype != jnp.bfloat16 for x in (query, key, value)):
        return False
    *_, t, n, d = query.shape
    *_, s, kv_heads, kd = key.shape
    if s != t or kd != d or t % _TILE or d % _TILE or n % kv_heads:
        return False
    if scale is not None and scale != 1 / math.sqrt(d):
        return False
    return jax.default_backend() == "tt"


def install():
    """Routes jax.nn.dot_product_attention's supported calls to the kernels."""
    if os.environ.get("LIBTT_FUSED_ATTENTION", "1") == "0":
        return
    jax_attention = jax.nn.dot_product_attention
    if getattr(jax_attention, "_libtt_fused", False):
        return

    @functools.wraps(jax_attention)
    def dot_product_attention(query, key, value, bias=None, mask=None, *, is_causal=False, **options):
        query, key, value = (jnp.asarray(x) for x in (query, key, value))
        if not _fused(query, key, value, bias, mask, options):
            return jax_attention(query, key, value, bias, mask, is_causal=is_causal, **options)
        unbatched = query.ndim == 3
        if unbatched:
            query, key, value = (x[None] for x in (query, key, value))
        # The kernels take [B, N, T, D].
        q, k, v = (x.transpose(0, 2, 1, 3) for x in (query, key, value))
        out = _attention(q, k, v, _CAUSAL if is_causal else _NO_MASK).transpose(0, 2, 1, 3)
        return out[0] if unbatched else out

    dot_product_attention._libtt_fused = True
    jax.nn.dot_product_attention = dot_product_attention
