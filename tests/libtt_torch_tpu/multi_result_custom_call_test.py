"""A JAX function that torch_tpu runs may call a custom call with several results.

XLA gives such a call a single tuple result, so it reaches libtt as a
stablehlo.custom_call returning a tuple and get_tuple_element ops that take it
apart, which Shardy rejects ("Shardy propagation doesn't support tuples").
SGLang-TorchTPU's gated DeltaNet layers call TTNN's gated delta decode this
way: it updates the state pool in place and returns the layer's output.
"""

import jax
import jax.numpy as jnp
import numpy as np
import torch
from torch_tpu._internal import pallas

KEY_HEADS, HEADS, DIM, BATCH, SLOTS = 1, 2, 128, 2, 4


def gated_delta_decode(
    state: jax.Array,
    qkv: jax.Array,
    b: jax.Array,
    a: jax.Array,
    A_log: jax.Array,
    dt_bias: jax.Array,
    indices: jax.Array,
    initial: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    return jax.ffi.ffi_call(
        "tt.gated_delta_decode",
        (
            jax.ShapeDtypeStruct(state.shape, state.dtype),
            jax.ShapeDtypeStruct((BATCH, HEADS, DIM), jnp.float32),
        ),
        input_output_aliases={0: 0},
        vmap_method="sequential",
    )(state, qkv, b, a, A_log, dt_bias, indices, initial.astype(jnp.bfloat16))


def test_custom_call_with_two_results(monkeypatch):
    # torch_tpu exports with DisabledSafetyCheck.custom_call("ALL"), which JAX
    # reads as every target from 0.10 on; on older JAX, allow this one.
    from jax._src.export import _export

    monkeypatch.setattr(
        _export,
        "_CUSTOM_CALL_TARGETS_GUARANTEED_STABLE",
        _export._CUSTOM_CALL_TARGETS_GUARANTEED_STABLE | {"tt.gated_delta_decode"},
    )
    op = pallas.jax_op("libtt_test::gated_delta_decode", gated_delta_decode)
    rng = np.random.default_rng(5)
    state = rng.normal(0, 0.05, (SLOTS, HEADS, DIM, DIM)).astype(np.float32)
    qk = rng.normal(0, 1, (BATCH, 2 * KEY_HEADS, DIM)).astype(np.float32)
    v = rng.normal(0, 0.5, (BATCH, HEADS, DIM)).astype(np.float32)
    b = rng.normal(0, 1, (BATCH, HEADS)).astype(np.float32)
    a = rng.normal(0, 1, (BATCH, HEADS)).astype(np.float32)
    A_log = rng.uniform(-1, 1, HEADS).astype(np.float32)
    dt_bias = rng.uniform(-1, 1, HEADS).astype(np.float32)
    indices = np.array([2, 1], np.int32)
    initial = np.array([1.0, 0.0], np.float32)
    qkv = np.concatenate([qk.reshape(BATCH, -1), v.reshape(BATCH, -1)], axis=1)

    new_state, output = op(
        *(torch.from_numpy(x).to("tpu") for x in (state, qkv, b, a, A_log, dt_bias, indices, initial))
    )

    norm = qk / np.sqrt(np.sum(qk * qk, axis=-1, keepdims=True) + 1e-6)
    q = np.repeat(norm[:, :KEY_HEADS], HEADS // KEY_HEADS, axis=1) * DIM**-0.5
    k = np.repeat(norm[:, KEY_HEADS:], HEADS // KEY_HEADS, axis=1)
    beta = 1 / (1 + np.exp(-b))
    gate = -np.exp(A_log) * np.log1p(np.exp(a + dt_bias))
    expected_state, expected_output = state.copy(), np.empty_like(v)
    for i, slot in enumerate(indices):
        s = state[slot] if initial[i] else np.zeros_like(state[slot])
        s = s * np.exp(gate[i])[:, None, None]
        residual = v[i] - np.einsum("hk,hkv->hv", k[i], s)
        s = s + k[i][:, :, None] * (beta[i][:, None] * residual)[:, None, :]
        expected_state[slot] = s
        expected_output[i] = np.einsum("hk,hkv->hv", q[i], s)
    np.testing.assert_allclose(output.cpu().numpy(), expected_output, atol=2e-3, rtol=0.05)
    np.testing.assert_allclose(new_state.cpu().numpy()[1:], expected_state[1:], atol=2e-3, rtol=0.05)
