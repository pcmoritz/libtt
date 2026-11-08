"""SGLang-TorchTPU's gated DeltaNet kernel on libtt's recurrent kernels (sglang_tpu_gdn).

`fused_conv1d_gdn` against a NumPy recurrence, with the pools in the layout
`patch_mamba_pool` gives them: a decode batch with a fresh request and a
padding row, ragged extend batches with fresh and continuing requests, and a
speculative verify batch.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax_plugins.libtt import sglang_tpu_gdn

N_KQ, N_V, DIM, TAPS = 2, 4, 128, 4
CHANNELS = (2 * N_KQ + N_V) * DIM
SLOTS = 12


def reference(qkv, b, a, conv, recurrent, weight, a_log, dt_bias, requests):
    """NumPy fused_conv1d_gdn. requests: (tokens, slot, initial), or (tokens,
    slot, initial, writes) to write the states after each token to its slot in
    writes instead of the last ones to slot."""
    conv, recurrent = conv.astype(np.float64).copy(), recurrent.astype(np.float64).copy()
    qkv, b, a = (x.astype(np.float64) for x in (qkv, b, a))
    out = np.zeros((qkv.shape[0], N_V, DIM))
    beta = 1 / (1 + np.exp(-b))
    gate = -np.exp(a_log) * np.log1p(np.exp(a + dt_bias))
    groups = N_V // N_KQ
    for slot_tokens, slot, initial, *writes in requests:
        window = conv[slot].T.copy() if initial else np.zeros((TAPS - 1, CHANNELS))
        state = recurrent[slot].copy() if initial else np.zeros((N_V, DIM, DIM))
        for i, t in enumerate(slot_tokens):
            taps = np.concatenate([window, qkv[t][None]])
            x = np.sum(taps * weight.T, axis=0)
            x = x / (1 + np.exp(-x))
            window = taps[1:]
            qk = x[: 2 * N_KQ * DIM].reshape(2 * N_KQ, DIM)
            qk = qk / np.sqrt(np.sum(qk * qk, axis=-1, keepdims=True) + 1e-6)
            q = np.repeat(qk[:N_KQ], groups, axis=0) * DIM**-0.5
            k = np.repeat(qk[N_KQ:], groups, axis=0)
            v = x[2 * N_KQ * DIM :].reshape(N_V, DIM)
            state = state * np.exp(gate[t])[:, None, None]
            residual = v - np.einsum("hk,hkv->hv", k, state)
            state = state + k[:, :, None] * (beta[t][:, None] * residual)[:, None, :]
            out[t] = np.einsum("hk,hkv->hv", q, state)
            if writes:
                conv[writes[0][i]], recurrent[writes[0][i]] = window.T, state
        if not writes:
            conv[slot], recurrent[slot] = window.T, state
    return conv, recurrent, out.reshape(out.shape[0], -1)


def run(qkv, b, a, conv, recurrent, weight, a_log, dt_bias, starts, slots, distribution, seq_lens,
        *verify, **options):
    def gdn(conv, recurrent, qkv, b, a, weight, a_log, dt_bias, starts, slots, distribution, seq_lens, *verify):
        (conv, recurrent), out = sglang_tpu_gdn.fused_conv1d_gdn(
            qkv, b, a, conv, recurrent, weight, None, a_log, dt_bias, starts, slots, distribution,
            seq_lens, *verify, n_kq=N_KQ, n_v=N_V, d_k=DIM, d_v=DIM, kernel_size=TAPS, **options,
        )
        return conv, recurrent, out

    args = [conv, recurrent, qkv, b, a, weight[:, None, :], a_log, dt_bias, starts, slots, distribution, seq_lens,
            *verify]
    device = jax.devices("tt")[0]
    conv, recurrent, out = jax.jit(gdn, donate_argnums=(0, 1), compiler_options={"optimization_level": "O0"})(
        *[jax.device_put(x, device) for x in args]
    )
    return np.asarray(conv).astype(np.float64), np.asarray(recurrent), np.asarray(out).astype(np.float64)


def inputs(tokens, seed=0):
    rng = np.random.default_rng(seed)
    qkv = np.asarray(rng.normal(0, 1, (tokens, CHANNELS)), jnp.bfloat16)
    b = np.asarray(rng.normal(0, 1, (tokens, N_V)), jnp.bfloat16)
    a = np.asarray(rng.normal(0, 1, (tokens, N_V)), jnp.bfloat16)
    conv = np.asarray(rng.normal(0, 1, (SLOTS, CHANNELS, TAPS - 1)), jnp.bfloat16)
    recurrent = rng.normal(0, 0.05, (SLOTS, N_V, DIM, DIM)).astype(np.float32)
    weight = np.asarray(rng.normal(0, 0.3, (CHANNELS, TAPS)), jnp.bfloat16)
    a_log = rng.uniform(-1, 1, N_V).astype(np.float32)
    dt_bias = rng.uniform(-1, 1, N_V).astype(np.float32)
    return qkv, b, a, conv, recurrent, weight, a_log, dt_bias


def check(got, expected, atol, rtol):
    for g, e in zip(got, expected):
        np.testing.assert_allclose(g, e, atol=atol, rtol=rtol)


def test_decode():
    # Slots 3 and 5 continue, slot 1 starts fresh (seq_len 1), the last row pads.
    qkv, b, a, conv, recurrent, weight, a_log, dt_bias = args = inputs(4)
    slots = np.array([3, 1, 5, 7], np.int32)
    seq_lens = np.array([10, 1, 4, 0], np.int32)
    got = run(*args, np.arange(5, dtype=np.int32), slots, np.array([3, 3, 3], np.int32), seq_lens,
              batched_only=True)
    expected = reference(*args, [([0], 3, True), ([1], 1, False), ([2], 5, True)])
    expected[2][3] = got[2][3] = 0  # not compared: a padding row's output
    check(got, expected, atol=2e-3, rtol=0.05)


@pytest.mark.parametrize(
    "lens,slots,seq_lens,tokens",
    [
        # One continuing request of 40 tokens in a bucket of 64.
        ([40], [5], [52], 64),
        # A fresh and a continuing request, and a padding row.
        ([20, 30, 0], [3, 8, 0], [20, 41, 0], 64),
    ],
)
def test_extend(lens, slots, seq_lens, tokens):
    qkv, b, a, conv, recurrent, weight, a_log, dt_bias = args = inputs(tokens, seed=2)
    starts = np.concatenate([[0], np.cumsum(lens)]).astype(np.int32)
    live = sum(1 for n in lens if n)
    got = run(*args, starts, np.array(slots, np.int32), np.array([0, 0, live], np.int32),
              np.array(seq_lens, np.int32), prefill_only=True)
    requests = [
        (range(starts[i], starts[i + 1]), slots[i], seq_lens[i] > lens[i])
        for i in range(live)
    ]
    expected = reference(*args, requests)
    # The chunked kernel takes q, k and v in BFLOAT16.
    check(got, expected, atol=1e-2, rtol=0.05)
    assert not np.any(got[2][starts[live]:])


def test_verify():
    # Windows of four tokens (three draft tokens). The first request reads the
    # state its last verify accepted, one past its base slot; the second reads
    # its base slot; the last row pads. Each writes the states after each of
    # its tokens to its base slot and the next three.
    qkv, b, a, conv, recurrent, weight, a_log, dt_bias = args = inputs(12, seed=4)
    starts = np.array([0, 4, 8, 8], np.int32)
    slots = np.array([2, 7, 0], np.int32)
    read_offsets = np.array([1, 0, 0], np.int32)
    got = run(*args, starts, slots, np.array([2, 2, 2], np.int32), np.array([20, 9, 0], np.int32), read_offsets,
              batched_only=True, num_spec_tokens=3)
    expected = reference(*args, [(range(0, 4), 3, True, [2, 3, 4, 5]), (range(4, 8), 7, True, [7, 8, 9, 10])])
    check(got, expected, atol=2e-3, rtol=0.05)
    assert not np.any(got[2][8:])
