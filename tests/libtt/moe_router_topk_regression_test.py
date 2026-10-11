"""Check the fused top k, softmax and index masking of a mixture-of-experts router."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax


def _router(k, masked):
    def route(logits, valid_rows):
        # sglang-jax's router: the softmax over every expert, the top k,
        # renormalized, and the padding tokens' experts masked.
        probs = jax.nn.softmax(logits, axis=-1)
        weights, ids = lax.top_k(probs, k)
        weights = weights / weights.sum(axis=-1, keepdims=True)
        if masked:
            valid = jnp.arange(logits.shape[0]) < valid_rows
            ids = jnp.where(valid[:, None], ids, -1)
        return weights, ids

    return route


def _reference(logits, k, valid_rows, masked):
    ids = np.argsort(-logits, axis=-1, kind="stable")[:, :k]
    top = np.take_along_axis(logits, ids, axis=-1).astype(np.float64)
    weights = np.exp(top - top[:, :1])
    weights /= weights.sum(axis=-1, keepdims=True)
    if masked:
        ids = np.where(np.arange(logits.shape[0])[:, None] < valid_rows, ids, -1)
    return weights, ids


@pytest.mark.parametrize("trace", [False, True])
@pytest.mark.parametrize(
    "rows,experts,k,masked",
    [
        # Qwen3-30B-A3B decode: one token, 128 experts, the top 8.
        (1, 128, 8, True),
        (1, 128, 8, False),
        # A few tokens, a full tile row of them, other widths and k.
        (4, 128, 8, True),
        (32, 128, 8, True),
        (2, 64, 4, True),
        (3, 256, 8, False),
    ],
)
def test_moe_router_topk(trace, rows, experts, k, masked, tmp_path):
    run = jax.jit(
        _router(k, masked),
        compiler_options={
            "optimization_level": "O1",
            "enable_trace": str(trace).lower(),
            "export_path": str(tmp_path),
        },
    )
    device = jax.devices("tt")[0]
    rng = np.random.default_rng(17)
    for valid_rows in (rows, max(rows - 1, 0), 0):
        # Distinct logits, exact in FP32, so the top k is unambiguous.
        logits = np.stack([rng.permutation(experts) for _ in range(rows)]).astype(np.float32) / 16
        weights, ids = run(jax.device_put(logits, device), jax.device_put(np.int32(valid_rows), device))
        expected_weights, expected_ids = _reference(logits, k, valid_rows, masked)
        np.testing.assert_array_equal(np.asarray(ids), expected_ids)
        np.testing.assert_allclose(np.asarray(weights), expected_weights, rtol=2e-3, atol=1e-5)

    assert any(
        "ttnn.moe_router_topk" in path.read_text() for path in (tmp_path / "irs").glob("ttnn*.mlir")
    ), "the router was not fused"
