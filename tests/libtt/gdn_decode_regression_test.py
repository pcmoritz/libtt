"""GDN decode with query and key heads shared by groups of value heads."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def gated_delta_decode(state, q, k, v, b, a, A_log, dt_bias, indices, initial):
    return jax.ffi.ffi_call(
        "tt.gated_delta_decode",
        (jax.ShapeDtypeStruct(state.shape, state.dtype), jax.ShapeDtypeStruct(v.shape, jnp.float32)),
        input_output_aliases={0: 0},
        vmap_method="sequential",
    )(state, q, k, v, b, a, A_log, dt_bias, indices, initial)


@pytest.mark.parametrize("trace", [False, True])
@pytest.mark.parametrize(
    "batch,key_heads,groups,flat", [(1, 4, 3, False), (1, 4, 3, True), (2, 2, 2, True), (1, 4, 1, False)]
)
def test_gdn_decode_grouped_heads(trace, batch, key_heads, groups, flat):
    # Like Qwen3.5: normalize q and k together, repeat each head for its group
    # of value heads, then split and scale q. With flat, q, k and v are slices
    # of one BF16 [B, (2 * key_heads + heads) * D] convolution output.
    heads, dim = key_heads * groups, 128

    def decode(state, qk, v, b, a, A_log, dt_bias, indices, initial):
        if flat:
            mixed = jnp.concatenate(
                [qk.reshape(batch, -1), v.reshape(batch, -1)], axis=1
            ).astype(jnp.bfloat16)
            mixed = mixed @ jnp.eye(mixed.shape[1], dtype=mixed.dtype)
            qk = mixed[:, : 2 * key_heads * dim].reshape(batch, 2 * key_heads, dim).astype(jnp.float32)
            v = mixed[:, 2 * key_heads * dim :].reshape(batch, heads, dim).astype(jnp.float32)
        qk = qk / jnp.sqrt(jnp.sum(qk * qk, axis=-1, keepdims=True) + 1e-6)
        qk = jnp.repeat(qk, groups, axis=-2)
        q, k = qk[:, :heads], qk[:, heads:]
        return gated_delta_decode(state, q * dim**-0.5, k, v, b, a, A_log, dt_bias, indices, initial)

    device = jax.devices("tt")[0]
    run = jax.jit(
        decode,
        compiler_options={"optimization_level": "O1", "enable_trace": str(trace).lower()},
    )
    rng = np.random.default_rng(71)
    slots = batch + 2
    for _ in range(3):
        qk = rng.normal(0, 1, (batch, 2 * key_heads, dim)).astype(np.float32)
        v = rng.normal(0, 0.5, (batch, heads, dim)).astype(np.float32)
        if flat:
            # The program rounds the flat input to BF16.
            qk, v = (x.astype(jnp.bfloat16).astype(np.float32) for x in (qk, v))
        b = rng.normal(0, 1, (batch, heads)).astype(np.float32)
        a = rng.normal(0, 1, (batch, heads)).astype(np.float32)
        A_log = rng.uniform(-1, 1, heads).astype(np.float32)
        dt_bias = rng.uniform(-1, 1, heads).astype(np.float32)
        state = rng.normal(0, 0.05, (slots, heads, dim, dim)).astype(np.float32)
        # Slot 0 is the padding slot, whose state is never written.
        indices = (1 + rng.permutation(slots - 1)[:batch]).astype(np.int32)
        initial = (np.arange(batch) % 2 == 0).astype(jnp.bfloat16)

        norm = qk / np.sqrt(np.sum(qk * qk, axis=-1, keepdims=True) + 1e-6)
        q = np.repeat(norm[:, :key_heads], groups, axis=1) * dim**-0.5
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
