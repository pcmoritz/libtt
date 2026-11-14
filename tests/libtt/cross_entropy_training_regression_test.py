"""tt-train's fused cross-entropy, as JAX custom calls and as TorchTitan's fused loss.

`tt.cross_entropy_fw` and `tt.cross_entropy_bw` run ttml::metal::cross_entropy_fw
and cross_entropy_bw on BF16 logits [N, 1, H, W]: per-row losses, and
(softmax - one_hot) * scaler * grad. In TPU compatibility mode they replace the
Pallas linear softmax cross-entropy behind TorchTitan's fused loss
(`jax_plugins.libtt.torchtitan_tpu_kernels`), a chunk of tokens at a time.
"""

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest

from jax_plugins.libtt import torchtitan_tpu_kernels

TOKENS, HIDDEN, VOCAB = 256, 512, 4096


def _relative_error(got, want):
    got = np.asarray(got).astype(np.float64)
    return np.max(np.abs(got - want)) / np.max(np.abs(want))


def _device(*arrays):
    device = jax.devices("tt")[0]
    return [jax.device_put(a, device) for a in arrays]


def _softmax_and_losses(logits, labels):
    logits = logits.astype(np.float64)
    lse = np.log(np.sum(np.exp(logits - logits.max(-1, keepdims=True)), -1)) + logits.max(-1)
    p = np.exp(logits - lse[:, None])
    return p, lse, lse - logits[np.arange(len(labels)), labels]


def test_custom_calls_match_reference():
    rng = np.random.default_rng(0)
    logits = (rng.standard_normal((TOKENS, VOCAB)) * 3).astype(ml_dtypes.bfloat16)
    labels = rng.integers(0, VOCAB, TOKENS).astype(np.int32)
    p, _, losses_want = _softmax_and_losses(logits, labels)
    p[np.arange(TOKENS), labels] -= 1
    dlogits_want = p * 0.5 * 2.0

    def forward(logits, labels):
        return jax.ffi.ffi_call("tt.cross_entropy_fw", jax.ShapeDtypeStruct((1, 1, TOKENS, 1), logits.dtype))(
            logits[None, None], labels[None]
        )

    def backward(logits, labels, grad):
        return jax.ffi.ffi_call(
            "tt.cross_entropy_bw", jax.ShapeDtypeStruct((1, 1, TOKENS, VOCAB), logits.dtype)
        )(logits[None, None], labels[None], grad, scaler=0.5)

    tl, tlab = _device(logits, labels)
    losses = jax.jit(forward)(tl, tlab)
    assert _relative_error(np.asarray(losses).reshape(TOKENS), losses_want) < 5e-3
    grad = jax.device_put(jnp.full((1, 1, 1, 1), 2.0, jnp.bfloat16), jax.devices("tt")[0])
    dlogits = jax.jit(backward)(tl, tlab, grad)
    assert _relative_error(np.asarray(dlogits)[0, 0], dlogits_want) < 1e-2


@pytest.mark.parametrize("reduction", ["sum", "mean", "none"])
def test_fused_loss_functions(monkeypatch, reduction):
    # TorchTitan's fused output layer and loss, two chunks of tokens and four
    # weight-gradient accumulators.
    monkeypatch.setattr(torchtitan_tpu_kernels, "LOSS_TOKEN_CHUNK", TOKENS // 2)
    monkeypatch.setattr(torchtitan_tpu_kernels, "LOSS_WEIGHT_COLUMNS", VOCAB // 4)
    rng = np.random.default_rng(1)
    x = (rng.standard_normal((TOKENS, HIDDEN)) * 0.5).astype(ml_dtypes.bfloat16)
    w = (rng.standard_normal((HIDDEN, VOCAB)) * 0.1).astype(ml_dtypes.bfloat16)
    labels = rng.integers(0, VOCAB, TOKENS).astype(np.int32)
    dout = rng.uniform(0.5, 1.5, TOKENS).astype(np.float32) if reduction == "none" else np.float32(1.5)

    xf, wf = x.astype(np.float64), w.astype(np.float64)
    p, lse_want, losses = _softmax_and_losses(xf @ wf, labels)
    p[np.arange(TOKENS), labels] -= 1
    scale = dout[:, None] if reduction == "none" else dout / TOKENS if reduction == "mean" else dout
    p *= scale
    loss_want = {"sum": losses.sum(), "mean": losses.mean(), "none": losses}[reduction]

    tx, tw, tl, td = _device(x, w, labels, dout)
    loss, lse = jax.jit(lambda x, l, w: torchtitan_tpu_kernels.linear_cross_entropy_fwd(x, l, w, reduction=reduction))(
        tx, tl, tw
    )
    assert lse.shape == (TOKENS,) and lse.dtype == jnp.float32
    assert _relative_error(loss, loss_want) < 5e-3
    assert _relative_error(lse, lse_want) < 5e-3
    dx, dw = jax.jit(
        lambda d, s, x, l, w: torchtitan_tpu_kernels.linear_cross_entropy_bwd(d, s, x, l, w, reduction=reduction)
    )(td, lse, tx, tl, tw)
    assert dx.dtype == x.dtype and dw.dtype == w.dtype
    assert _relative_error(dx, p @ wf.T) < 3e-2
    assert _relative_error(dw, xf.T @ p) < 3e-2


def test_weight_gradient_adds_up_in_float32(monkeypatch):
    # 32 chunks of tokens that all take label 0, with w = 0: every softmax
    # probability is 1 / 4096, exactly, and so is the gradient of a column other
    # than the label's. The first chunk's tokens are 2048 times larger, so it
    # adds 16 to that column's weight gradient and each later chunk 1 / 128,
    # which a BF16 sum of 16 rounds away (16 instead of 16.2421875); in FP32
    # only the final rounding remains.
    monkeypatch.setattr(torchtitan_tpu_kernels, "LOSS_TOKEN_CHUNK", 32)
    monkeypatch.setattr(torchtitan_tpu_kernels, "LOSS_WEIGHT_COLUMNS", VOCAB // 4)
    tokens = 32 * 32
    x = np.ones((tokens, HIDDEN), ml_dtypes.bfloat16)
    x[:32] = 2048
    w = np.zeros((HIDDEN, VOCAB), ml_dtypes.bfloat16)
    labels = np.zeros(tokens, np.int32)
    tx, tw, tl = _device(x, w, labels)
    _, lse = jax.jit(torchtitan_tpu_kernels.linear_cross_entropy_fwd)(tx, tl, tw)
    _, dw = jax.jit(torchtitan_tpu_kernels.linear_cross_entropy_bwd)(jnp.ones((), jnp.float32), lse, tx, tl, tw)
    want = (32 * 2048 + (tokens - 32)) / VOCAB
    assert np.max(np.abs(np.asarray(dw)[:, 1].astype(np.float64) - want)) / want < 5e-3
