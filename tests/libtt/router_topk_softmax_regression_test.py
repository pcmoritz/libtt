"""A mixture-of-experts router's top-k of softmax, renormalized, is a softmax
over the top k of the logits.

Routers such as sglang-jax's take softmax over all experts in FP32, the top k
of the probabilities and divide them by their sum. tt-mlir rewrites that to a
top-k of the logits and a softmax over the k picks, which skips the full
softmax and the renormalization's reduction and division. The experts and
their weights must match float64. Without the renormalization the top-k values
are probabilities over all experts and keep the full softmax.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

HIDDEN = 512


def route(hidden, gate, k, renormalize):
    logits = jnp.dot(hidden.astype(jnp.float32), gate, precision=lax.Precision.HIGHEST)
    probs = jax.nn.softmax(logits, axis=-1)
    weights, ids = lax.top_k(probs, k)
    if renormalize:
        weights = weights / jnp.sum(weights, axis=-1, keepdims=True)
    return weights, ids


@pytest.mark.parametrize("renormalize", [True, False])
@pytest.mark.parametrize("tokens,experts,k", [(1, 128, 8), (4, 128, 8), (33, 128, 8), (1, 256, 8), (2, 64, 4)])
def test_router(tokens, experts, k, renormalize, tmp_path):
    rng = np.random.default_rng(tokens * experts + k)
    hidden = rng.standard_normal((tokens, HIDDEN)).astype(jnp.bfloat16)
    gate = (rng.standard_normal((HIDDEN, experts)) / np.sqrt(HIDDEN)).astype(np.float32)
    run = jax.jit(lambda h, g: route(h, g, k, renormalize), compiler_options={"export_path": str(tmp_path)})
    device = jax.devices("tt")[0]
    weights, ids = (np.asarray(a) for a in run(jax.device_put(hidden, device), jax.device_put(gate, device)))

    logits = np.asarray(hidden).astype(np.float64) @ gate.astype(np.float64)
    probs = np.exp(logits - logits.max(-1, keepdims=True))
    probs /= probs.sum(-1, keepdims=True)
    # The device's FP32 matmul reads the gate at the matrix unit's 19 bits, so
    # logits closer than that may swap: in order within the top k, and at the
    # cut, where any expert within the tolerance of the k-th logit is a valid
    # pick.
    tolerance = 5e-3
    want_ids = np.argsort(-probs, axis=-1, kind="stable")[:, :k]
    ordered = np.sort(logits, axis=-1)[:, ::-1]
    near_tie = ordered[:, k - 1] - ordered[:, k] < tolerance
    np.testing.assert_array_equal(np.sort(ids[~near_tie], -1), np.sort(want_ids[~near_tie], -1))
    picked = np.take_along_axis(logits, ids, -1)
    assert (picked >= ordered[:, k - 1 : k] - tolerance).all(), "an expert outside the top k"
    assert (np.diff(picked, axis=-1) <= tolerance).all(), "the experts are out of order"
    assert all(len(set(row)) == k for row in ids), "an expert picked twice"
    want = np.take_along_axis(probs, ids, -1)
    if renormalize:
        want /= want.sum(-1, keepdims=True)
    np.testing.assert_allclose(weights.astype(np.float64), want, rtol=1e-2, atol=1e-4)

    irs = [path.read_text() for path in (tmp_path / "irs").glob("ttnn_[0-9]*.mlir")]
    assert irs, "no IR was exported"
    ir = "\n".join(irs)
    if renormalize:
        assert '"ttnn.sum"' not in ir and '"ttnn.divide"' not in ir, "the renormalization was left over"
        softmaxes = [line for line in ir.splitlines() if '"ttnn.softmax"' in line]
        assert softmaxes and all(f"x{experts}xf32" not in line for line in softmaxes), "the softmax runs over all experts"
