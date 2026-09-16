"""GDN prefill owns its kernel constants, including during trace capture."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def gated_delta_rule(q, k, v, gate, beta, state):
    return jax.ffi.ffi_call(
        "tt.gated_delta_rule",
        (jax.ShapeDtypeStruct(state.shape, state.dtype), jax.ShapeDtypeStruct(v.shape, v.dtype)),
        vmap_method="sequential",
    )(q, k, v, gate, beta, state)


@pytest.mark.parametrize("trace", [False, True])
def test_gdn_prefill_without_kernel_constants(trace):
    device = jax.devices("tt")[0]
    run = jax.jit(
        gated_delta_rule,
        compiler_options={"optimization_level": "1", "enable_trace": str(trace).lower()},
    )
    rng = np.random.default_rng(35)
    shape = (1, 64, 2, 128)

    # Exercise warmup, capture and replay with different inputs and initial states.
    for _ in range(3):
        q, k, v = (rng.normal(0, 0.1, shape).astype(np.float32) for _ in range(3))
        q /= np.linalg.norm(q, axis=-1, keepdims=True) * np.sqrt(np.float32(128))
        k /= np.linalg.norm(k, axis=-1, keepdims=True)
        # TTNN uses BF16 q/k/v and FP32 gates and recurrent arithmetic.
        q, k, v = (x.astype(jnp.bfloat16).astype(np.float32) for x in (q, k, v))
        gate = rng.uniform(-0.2, -0.01, shape[:-1]).astype(np.float32)
        beta = rng.uniform(0.2, 0.8, shape[:-1]).astype(np.float32)
        state = rng.normal(0, 0.01, (1, 2, 128, 128)).astype(np.float32)
        expected_state = state[0].copy()
        expected_output = np.empty_like(v)
        for t in range(shape[1]):
            expected_state *= np.exp(gate[0, t, :, None, None])
            residual = v[0, t] - np.einsum("hk,hkv->hv", k[0, t], expected_state)
            expected_state += k[0, t, :, :, None] * (beta[0, t, :, None] * residual)[:, None, :]
            expected_output[0, t] = np.einsum("hk,hkv->hv", q[0, t], expected_state)

        actual_state, actual_output = run(
            *(jax.device_put(x, device) for x in (q, k, v, gate, beta, state))
        )
        np.testing.assert_allclose(
            np.asarray(actual_state), expected_state[None], atol=1e-3, rtol=0.06
        )
        np.testing.assert_allclose(np.asarray(actual_output), expected_output, atol=3e-5, rtol=0.06)
