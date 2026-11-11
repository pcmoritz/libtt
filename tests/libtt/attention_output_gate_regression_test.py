"""Gated attention's output gate runs as one program, bit for bit.

Qwen3.5 and Qwen3.8 gate each attention head's output by the sigmoid of a gate
head that follows its q head in the QKV projection. SGLang-TorchTPU traces it
as slices of the projection and of the paged SDPA decode output (whose rows
hold k and v side by side), a typecast to FP32, sigmoid, a typecast back to
BF16 and a multiply: nine programs per attention layer, fused into one
ttnn.attention_output_gate.
"""

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

OPTIONS = {"optimization_level": "O0"}


def _gate(qkv, heads, head_dim):
    """torch.sigmoid of the gate heads (each after its q head), as torch_tpu
    traces it on BF16: in FP32, cast back to BF16, as [T, heads * head_dim]."""
    tokens = qkv.shape[0]
    q_gate = qkv[:, : heads * 2 * head_dim].reshape(tokens, heads, 2 * head_dim)
    gate = q_gate[..., head_dim:].astype(jnp.float32)
    return jax.nn.sigmoid(gate).astype(jnp.bfloat16).reshape(tokens, heads * head_dim)


def _values(attention, head_dim):
    """The value half of each head of the SDPA output [1, T, heads, 2 head_dim]."""
    _, tokens, heads, _ = attention.shape
    return attention[0, :, :, head_dim:].reshape(tokens, heads * head_dim)


@pytest.mark.parametrize(
    "tokens,heads,kv_heads",
    [
        # Qwen3.8-27B on four chips, one token and SGLang's batch of four; on one chip.
        (1, 6, 1),
        (4, 6, 1),
        (1, 24, 4),
    ],
)
def test_gated_attention_output(tokens, heads, kv_heads, tmp_path):
    head_dim = 256
    width = heads * 2 * head_dim + 2 * kv_heads * head_dim
    device = jax.devices("tt")[0]
    rng = np.random.default_rng(tokens * 100 + heads)
    attention = jax.device_put(jnp.asarray(rng.normal(0, 1, (1, tokens, heads, 2 * head_dim)), jnp.bfloat16), device)
    # Gate logits of all sizes, so the sigmoid saturates and rounds both ways.
    qkv = jax.device_put(jnp.asarray(rng.normal(0, 4, (tokens, width)), jnp.bfloat16), device)

    fused = jax.jit(
        lambda a, q: _values(a, head_dim) * _gate(q, heads, head_dim),
        compiler_options={**OPTIONS, "export_path": str(tmp_path)},
    )
    # The gate and the multiply in separate programs, which nothing fuses.
    gate = jax.jit(lambda q: _gate(q, heads, head_dim), compiler_options=OPTIONS)(qkv)
    want = jax.jit(lambda a, g: _values(a, head_dim) * g, compiler_options=OPTIONS)(attention, gate)
    np.testing.assert_array_equal(np.asarray(fused(attention, qkv), np.float32), np.asarray(want, np.float32))

    irs = [path.read_text() for path in (tmp_path / "irs").glob("ttnn_[0-9]*.mlir")]
    assert any(re.search(r'"ttnn\.attention_output_gate"', ir) for ir in irs), "the gate was not fused"
    assert not any('"ttnn.sigmoid"' in ir for ir in irs), "a sigmoid was left"
