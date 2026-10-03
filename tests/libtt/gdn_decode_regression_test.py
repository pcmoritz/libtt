"""GDN decode from the flat convolution output, with grouped query and key heads."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def gated_delta_decode(state, qkv, b, a, A_log, dt_bias, indices, initial):
    return jax.ffi.ffi_call(
        "tt.gated_delta_decode",
        (
            jax.ShapeDtypeStruct(state.shape, state.dtype),
            jax.ShapeDtypeStruct((*b.shape, state.shape[-1]), jnp.float32),
        ),
        input_output_aliases={0: 0},
        vmap_method="sequential",
    )(state, qkv, b, a, A_log, dt_bias, indices, initial)


@pytest.mark.parametrize("trace", [False, True])
@pytest.mark.parametrize(
    "batch,key_heads,groups,qk_scale,qkv_dtype,gate_dtype",
    [
        (1, 4, 3, 1.0, np.float32, np.float32),
        (2, 2, 2, 1.0, np.float32, np.float32),
        (1, 4, 1, 1.0, np.float32, np.float32),
        # 48 value heads: scalars and outputs past head 32, and more workers.
        (2, 16, 3, 1.0, np.float32, np.float32),
        # Tiny and zero q and k, whose norms the epsilon dominates.
        (1, 4, 3, 1e-4, np.float32, np.float32),
        (1, 4, 3, 0.0, np.float32, np.float32),
        # The serving combination: the FP32 convolution output with the
        # model's BFLOAT16 gate logits, with odd rows and gate columns past
        # the first face.
        (2, 16, 3, 1.0, np.float32, jnp.bfloat16),
        # A BFLOAT16 convolution output is widened before the kernel.
        (1, 4, 3, 1.0, jnp.bfloat16, jnp.bfloat16),
    ],
)
def test_gdn_decode_grouped_heads(trace, batch, key_heads, groups, qk_scale, qkv_dtype, gate_dtype):
    # Like Qwen3.5: the kernel reads q, k and v as heads of one
    # [B, (2 * key_heads + heads) * D] tensor, L2-normalizes q and k, scales q,
    # and each q/k head serves a group of value heads.
    heads, dim = key_heads * groups, 128
    eps, scale = 1e-6, dim**-0.5

    def decode(state, qk, v, b, a, A_log, dt_bias, indices, initial):
        mixed = jnp.concatenate([qk.reshape(batch, -1), v.reshape(batch, -1)], axis=1).astype(qkv_dtype)
        return gated_delta_decode(
            state, mixed, b.astype(gate_dtype), a.astype(gate_dtype), A_log, dt_bias, indices, initial
        )

    device = jax.devices("tt")[0]
    run = jax.jit(
        decode,
        compiler_options={"optimization_level": "O1", "enable_trace": str(trace).lower()},
    )
    rng = np.random.default_rng(71)
    slots = batch + 2
    for _ in range(3):
        # The inputs as the kernel sees them, rounded to dtype.
        qk = (rng.normal(0, 1, (batch, 2 * key_heads, dim)) * qk_scale).astype(qkv_dtype).astype(np.float32)
        v = rng.normal(0, 0.5, (batch, heads, dim)).astype(qkv_dtype).astype(np.float32)
        b = rng.normal(0, 1, (batch, heads)).astype(gate_dtype).astype(np.float32)
        a = rng.normal(0, 1, (batch, heads)).astype(gate_dtype).astype(np.float32)
        A_log = rng.uniform(-1, 1, heads).astype(np.float32)
        dt_bias = rng.uniform(-1, 1, heads).astype(np.float32)
        state = rng.normal(0, 0.05, (slots, heads, dim, dim)).astype(np.float32)
        # Slot 0 is the padding slot, whose state is never written.
        indices = (1 + rng.permutation(slots - 1)[:batch]).astype(np.int32)
        initial = (np.arange(batch) % 2 == 0).astype(jnp.bfloat16)

        norm = qk / np.sqrt(np.sum(qk * qk, axis=-1, keepdims=True) + eps)
        q = np.repeat(norm[:, :key_heads], groups, axis=1) * scale
        k = np.repeat(norm[:, key_heads:], groups, axis=1)
        beta = 1 / (1 + np.exp(-b))
        gate = -np.exp(A_log) * np.log1p(np.exp(a + dt_bias))
        expected_state = state.copy()
        expected_output = np.empty_like(v)
        for i, slot in enumerate(indices):
            s = state[slot] if initial[i] else np.zeros_like(state[slot])
            s = s * np.exp(gate[i])[:, None, None]
            residual = v[i] - np.einsum("hk,hkv->hv", k[i], s)
            s = s + k[i][:, :, None] * (beta[i][:, None] * residual)[:, None, :]
            expected_state[slot] = s
            expected_output[i] = np.einsum("hk,hkv->hv", q[i], s)

        actual_state, actual_output = run(
            *(jax.device_put(x, device) for x in (state, qk, v, b, a, A_log, dt_bias, indices, initial))
        )
        np.testing.assert_allclose(np.asarray(actual_state), expected_state, atol=2e-3, rtol=0.05)
        np.testing.assert_allclose(np.asarray(actual_output), expected_output, atol=2e-3, rtol=0.05)


@pytest.mark.parametrize("output_dtype", [jnp.bfloat16, jnp.float32])
def test_conv_update_output_dtype(output_dtype):
    """The convolution update writes silu(conv) in the dtype the graph
    declares: BFLOAT16, or FP32 for the decode kernel without a cast."""
    batch, channels, slots, taps = 2, 256, 4, 4

    def conv_update(state, value, weight, indices, initial):
        return jax.ffi.ffi_call(
            "tt.causal_conv1d_update",
            (
                jax.ShapeDtypeStruct(state.shape, state.dtype),
                jax.ShapeDtypeStruct(value.shape, output_dtype),
            ),
            input_output_aliases={0: 0},
            vmap_method="sequential",
        )(state, value, weight, indices, initial)

    device = jax.devices("tt")[0]
    run = jax.jit(conv_update, compiler_options={"optimization_level": "O1"})
    rng = np.random.default_rng(5)
    state = rng.normal(0, 1, (slots, channels, taps - 1)).astype(jnp.bfloat16)
    value = rng.normal(0, 1, (batch, channels)).astype(jnp.bfloat16)
    weight = rng.normal(0, 0.5, (channels, taps)).astype(jnp.bfloat16)
    indices = np.array([2, 1], np.int32)
    initial = np.array([1, 0], np.float32).astype(jnp.bfloat16)
    new_state, out = run(*(jax.device_put(x, device) for x in (state, value, weight, indices, initial)))

    s = state.astype(np.float32)
    history = np.stack([np.where(initial[i], s[slot], 0) for i, slot in enumerate(indices)])  # [B, C, 3]
    history = np.concatenate([history, value.astype(np.float32)[:, :, None]], axis=-1)  # [B, C, 4]
    conv = np.sum(history * weight.astype(np.float32)[None], axis=-1)
    expected = conv / (1 + np.exp(-conv))
    actual = np.asarray(out).astype(np.float32)
    assert out.dtype == output_dtype
    # The kernel computes in BF16 either way; the FP32 output only spares the
    # decode kernel a cast.
    np.testing.assert_allclose(actual, expected, atol=1 / 64, rtol=1 / 64)
    expected_state = s.copy()
    for i, slot in enumerate(indices):
        expected_state[slot] = history[i, :, 1:]
    np.testing.assert_allclose(np.asarray(new_state).astype(np.float32), expected_state, atol=0, rtol=0)
