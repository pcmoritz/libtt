"""Decode rotary embedding must preserve heads smaller than a tile."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize("trace", [False, True])
@pytest.mark.parametrize("heads", [4, 16, 32])
def test_rotary_decode_logical_heads(trace, heads):
    def decode(q, k, v, cos, sin, table, positions):
        rotated = jnp.concatenate((-q[..., 64:], q[..., :64]), axis=-1)
        q = (q * cos + rotated * sin).transpose(0, 2, 1, 3)
        out = jax.ffi.ffi_call(
            "tt.paged_scaled_dot_product_attention_decode",
            jax.ShapeDtypeStruct(q.shape, q.dtype),
            vmap_method="sequential",
        )(
            q,
            k,
            v,
            table,
            positions,
            is_causal=True,
            has_attention_mask=False,
            has_cur_pos_tensor=True,
            has_attention_sink=False,
        )
        # The fused rotary must not expose its head padding to this reshape.
        return out.reshape(1, heads * 128)

    run = jax.jit(
        decode,
        compiler_options={"optimization_level": "O1", "enable_trace": str(trace).lower()},
    )
    device = jax.devices("tt")[0]
    rng = np.random.default_rng(7)
    table = np.arange(4, dtype=np.int32).reshape(1, 4)
    positions = np.array([31], dtype=np.int32)
    for _ in range(3):  # Warmup, capture, and replay with changed inputs.
        q = rng.normal(0, 0.2, (1, heads, 1, 128)).astype(jnp.bfloat16)
        k, v = (rng.normal(0, 0.2, (4, 4, 32, 128)).astype(jnp.bfloat16) for _ in range(2))
        angles = np.tile(rng.uniform(-3, 3, (1, 1, 1, 64)), 2)
        cos, sin = (f(angles).astype(jnp.bfloat16) for f in (np.cos, np.sin))
        actual = run(*(jax.device_put(x, device) for x in (q, k, v, cos, sin, table, positions)))

        q, k, v, cos, sin = (x.astype(np.float32) for x in (q, k, v, cos, sin))
        q = q * cos + np.concatenate((-q[..., 64:], q[..., :64]), axis=-1) * sin
        q = q.astype(jnp.bfloat16).astype(np.float32).reshape(heads, 128)
        keys, values = (np.repeat(x[0], heads // 4, axis=0) for x in (k, v))
        scores = np.einsum("hd,hsd->hs", q, keys) / np.sqrt(128)
        scores = np.exp(scores - scores.max(axis=-1, keepdims=True))
        scores /= scores.sum(axis=-1, keepdims=True)
        expected = np.einsum("hs,hsd->hd", scores, values).reshape(1, -1)
        np.testing.assert_allclose(np.asarray(actual).astype(np.float32), expected, atol=0.004, rtol=0.02)
