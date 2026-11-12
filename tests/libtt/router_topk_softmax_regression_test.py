"""A mixture-of-experts router's top-k of softmax, renormalized, is a softmax
over the top k of the logits.

Routers such as sglang-jax's take softmax over all experts in FP32, the top k
of the probabilities and divide them by their sum. tt-mlir rewrites that to a
top-k of the logits and a softmax over the k picks, which skips the full
softmax and the renormalization's reduction and division. The experts and
their weights must match float64. Without the renormalization the top-k values
are probabilities over all experts and keep the full softmax.

SGLang-TorchTPU's router reaches tt-mlir as PyTorch lowers it: the softmax as
exp times the reciprocal of its sum, and an epsilon of 1e-20 added to the
renormalization's sum, which can't change an FP32 sum of the top k
probabilities. It is rewritten the same way.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

HIDDEN = 512


def route(logits, k, renormalize):
    weights, ids = lax.top_k(jax.nn.softmax(logits, axis=-1), k)
    if renormalize:
        weights /= weights.sum(axis=-1, keepdims=True)
    return weights, ids


def distinct_logits(rows, experts, seed):
    # Logits 1/16 to 3/16 apart, which the matrix unit's 19 bits hold exactly,
    # so the top k is unambiguous. The spread differs between rows so their
    # weights do too.
    rng = np.random.default_rng(seed)
    return np.stack([rng.permutation(experts) * (row % 3 + 1) for row in range(rows)]).astype(np.float32) / 16


def check_route(logits, k, renormalize, weights, ids):
    logits = logits.astype(np.float64)
    probs = np.exp(logits - logits.max(-1, keepdims=True))
    probs /= probs.sum(-1, keepdims=True)
    want_ids = np.argsort(-logits, axis=-1)[:, :k]
    np.testing.assert_array_equal(ids, want_ids)
    want = np.take_along_axis(probs, want_ids, -1)
    if renormalize:
        want /= want.sum(-1, keepdims=True)
    # The device's FP32 exp is accurate to about 1e-3.
    np.testing.assert_allclose(weights.astype(np.float64), want, rtol=5e-3, atol=1e-6)


def exported_ir(tmp_path):
    irs = [path.read_text() for path in (tmp_path / "irs").glob("ttnn_[0-9]*.mlir")]
    assert irs, "no IR was exported"
    return "\n".join(irs)


def check_rewritten(ir, experts):
    assert '"ttnn.sum"' not in ir and '"ttnn.divide"' not in ir, "the renormalization was left over"
    softmaxes = [line for line in ir.splitlines() if '"ttnn.softmax"' in line]
    assert softmaxes and all(f"x{experts}xf32" not in line for line in softmaxes), "the softmax runs over all experts"


@pytest.mark.parametrize("renormalize", [True, False])
@pytest.mark.parametrize("tokens,experts,k", [(1, 128, 8), (4, 128, 8), (33, 128, 8), (1, 256, 8), (2, 64, 4)])
def test_router(tokens, experts, k, renormalize, tmp_path):
    logits = distinct_logits(tokens, experts, tokens * experts + k)
    run = jax.jit(lambda x: route(x, k, renormalize), compiler_options={"export_path": str(tmp_path)})
    weights, ids = (np.asarray(a) for a in run(jax.device_put(logits, jax.devices("tt")[0])))

    check_route(logits, k, renormalize, weights, ids)
    if renormalize:
        check_rewritten(exported_ir(tmp_path), experts)


def test_router_in_a_model(tmp_path):
    # As in a model: the logits come from an FP32 matmul of the BF16 hidden
    # state with the gate (one-hot hidden states pick the gate's rows), and the
    # experts are gathered before the weights are renormalized, so the
    # rewritten top-k must come before the gather.
    tokens, experts, k, width = 4, 128, 8, 16
    hidden = np.eye(tokens, HIDDEN).astype(jnp.bfloat16)
    gate = np.zeros((HIDDEN, experts), np.float32)
    gate[:tokens] = distinct_logits(tokens, experts, 0)
    table = np.random.default_rng(1).standard_normal((experts, width)).astype(np.float32)

    def run(hidden, gate, table):
        logits = jnp.dot(hidden.astype(jnp.float32), gate, precision=lax.Precision.HIGHEST)
        weights, ids = lax.top_k(jax.nn.softmax(logits, axis=-1), k)
        selected = jnp.take(table, ids, axis=0)
        return weights / weights.sum(axis=-1, keepdims=True), ids, selected

    device = jax.devices("tt")[0]
    weights, ids, selected = (
        np.asarray(a)
        for a in jax.jit(run, compiler_options={"export_path": str(tmp_path)})(
            *(jax.device_put(a, device) for a in (hidden, gate, table))
        )
    )

    check_route(gate[:tokens], k, True, weights, ids)
    np.testing.assert_array_equal(selected, table[ids])
    check_rewritten(exported_ir(tmp_path), experts)


def test_denominator_broadcast_across_rows(tmp_path):
    # Two rows of two picks divided by the row sums laid out as a row, so each
    # row is divided elementwise by both sums: not a renormalization.
    rows = k = 2
    experts = 64

    def run(logits):
        weights, _ = lax.top_k(jax.nn.softmax(logits, axis=-1), k)
        return weights / jnp.sum(weights, axis=-1).reshape(1, rows)

    logits = distinct_logits(rows, experts, 0)
    got = jax.jit(run, compiler_options={"export_path": str(tmp_path)})(jax.device_put(logits, jax.devices("tt")[0]))

    probs = np.exp(logits.astype(np.float64))
    probs = -np.sort(-probs / probs.sum(-1, keepdims=True), axis=-1)[:, :k]
    want = probs / probs.sum(-1).reshape(1, rows)
    np.testing.assert_allclose(np.asarray(got).astype(np.float64), want, rtol=5e-3, atol=1e-6)
    softmaxes = [line for line in exported_ir(tmp_path).splitlines() if '"ttnn.softmax"' in line]
    assert any(f"x{experts}xf32" in line for line in softmaxes), "the division was rewritten to a softmax"



def torch_route(logits, k, epsilon):
    # As PyTorch lowers SGLang-TorchTPU's router (fused_topk_torch_native).
    exp = jnp.exp(logits - logits.max(axis=-1, keepdims=True))
    probs = exp * (1.0 / exp.sum(axis=-1, keepdims=True))
    weights, ids = lax.top_k(probs, k)
    return weights / (weights.sum(axis=-1, keepdims=True) + epsilon), ids


@pytest.mark.parametrize("epsilon", [1e-20, 1e-3])
@pytest.mark.parametrize("tokens,experts,k", [(1, 128, 8), (33, 128, 8), (2, 256, 8)])
def test_torch_router(tokens, experts, k, epsilon, tmp_path):
    logits = distinct_logits(tokens, experts, tokens * experts + k)
    run = jax.jit(lambda x: torch_route(x, k, epsilon), compiler_options={"export_path": str(tmp_path)})
    weights, ids = (np.asarray(a) for a in run(jax.device_put(logits, jax.devices("tt")[0])))

    want_ids = np.argsort(-logits.astype(np.float64), axis=-1)[:, :k]
    np.testing.assert_array_equal(ids, want_ids)
    probs = np.exp(logits.astype(np.float64) - logits.max(-1, keepdims=True))
    picked = np.take_along_axis(probs / probs.sum(-1, keepdims=True), want_ids, -1)
    want = picked / (picked.sum(-1, keepdims=True) + epsilon)
    np.testing.assert_allclose(weights.astype(np.float64), want, rtol=5e-3, atol=1e-6)
    # An epsilon of 1e-3 changes the weights by up to 1e-3 / (k / experts), which a rewrite would drop.
    ir = exported_ir(tmp_path)
    if epsilon < 1e-12:
        check_rewritten(ir, experts)
    else:
        softmaxes = [line for line in ir.splitlines() if '"ttnn.softmax"' in line]
        assert any(f"x{experts}xf32" in line for line in softmaxes), "the softmax wasn't fused"
        assert '"ttnn.divide"' in ir or '"ttnn.add"' in ir, "the renormalization was dropped"
