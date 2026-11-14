"""TorchTitan's complex RoPE without [..., D/2, 2] tensors (`torchtitan_tpu_kernels.apply_complex_rope`).

Llama 3's rotary embedding rotates adjacent pairs of each head's columns, which
TorchTitan writes on a [..., D/2, 2] view of the queries and keys; in 32 x 32
tiles that view takes 16 times its size. The replacement splits the columns
into even and odd ones with BF16 permutation matmuls: its outputs and the
gradients of its inputs must equal those of TorchTitan's real-arithmetic
version, run here on CPU, bit for bit. Other types than BF16 keep TorchTitan's
version: FP32 values aren't moved exactly by the matmuls, and Tenstorrent tiles
have no FP16.
"""

import pytest
import torch
import torch_tpu  # noqa: F401
from jax_plugins.libtt import torchtitan_tpu_kernels


def titan_rope(query, key, rope_cache, *, inverse=False):
    """TorchTitan's `TpuRealArithmeticComplexRoPE.apply_rotary_emb` (experiments/tpu/overrides/real_rope.py).
    Its `view_as_real` of the conjugate view fails on CPU, so the conjugate is resolved here."""
    if inverse:
        rope_cache = rope_cache.conj().resolve_conj()
    cache = torch.view_as_real(rope_cache)
    cos, sin = cache[..., 0:1], cache[..., 1:2]

    def rotate(x):
        pairs = x.float().reshape(*x.shape[:-1], -1, 2)
        real, imag = pairs[..., 0:1], pairs[..., 1:2]
        rotated = torch.cat([real * cos - imag * sin, real * sin + imag * cos], dim=-1)
        return rotated.flatten(-2).type_as(x)

    if key is None:
        return rotate(query)
    return rotate(query), rotate(key)


apply_complex_rope = torchtitan_tpu_kernels._complex_rope(titan_rope).__func__


def rope_cache(tokens, dim, theta=500000.0):
    freqs = torch.outer(torch.arange(tokens, dtype=torch.float32), theta ** -(torch.arange(0, dim, 2).float() / dim))
    return torch.polar(torch.ones_like(freqs), freqs).view(tokens, 1, dim // 2)


@pytest.mark.parametrize("inverse", [False, True])
def test_rope_matches_torchtitan(inverse):
    tokens, heads, kv_heads, dim = 256, 4, 2, 128
    generator = torch.Generator().manual_seed(0)
    query = torch.randn(tokens, heads, dim, generator=generator).bfloat16()
    key = torch.randn(tokens, kv_heads, dim, generator=generator).bfloat16()
    query_weights = torch.randn(tokens, heads, dim, generator=generator)
    key_weights = torch.randn(tokens, kv_heads, dim, generator=generator)
    cache = rope_cache(tokens, dim)

    def loss(rope, query, key, cache, query_weights, key_weights):
        q, k = rope(query, key, cache, inverse=inverse)
        return (q.float() * query_weights).sum() + (k.float() * key_weights).sum(), q, k

    reference_query, reference_key = query.clone().requires_grad_(), key.clone().requires_grad_()
    reference_loss, want_q, want_k = loss(titan_rope, reference_query, reference_key, cache, query_weights, key_weights)
    reference_loss.backward()

    device_query, device_key = query.to("tpu").requires_grad_(), key.to("tpu").requires_grad_()
    compiled = torch.compile(loss, backend="tpu")
    device_loss, got_q, got_k = compiled(
        apply_complex_rope, device_query, device_key, cache.to("tpu"), query_weights.to("tpu"), key_weights.to("tpu")
    )
    device_loss.backward()

    assert torch.equal(got_q.cpu(), want_q)
    assert torch.equal(got_k.cpu(), want_k)
    assert torch.equal(device_query.grad.cpu(), reference_query.grad)
    assert torch.equal(device_key.grad.cpu(), reference_key.grad)


def test_queries_only():
    tokens, heads, dim = 64, 2, 64
    query = torch.randn(tokens, heads, dim, generator=torch.Generator().manual_seed(1)).bfloat16()
    cache = rope_cache(tokens, dim)
    got = apply_complex_rope(query.to("tpu"), None, cache.to("tpu"))
    assert torch.equal(got.cpu(), titan_rope(query, None, cache))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_other_types_keep_torchtitan(dtype):
    calls = []

    def original(query, key, rope_cache, *, inverse=False):
        calls.append(query.dtype)
        return titan_rope(query, key, rope_cache, inverse=inverse)

    rope = torchtitan_tpu_kernels._complex_rope(original).__func__
    query = torch.randn(8, 2, 64).to(dtype)
    rope(query, None, rope_cache(8, 64))
    assert calls == [dtype]
