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


def _gate_router(k, weight_dtype):
    route = _router(k, masked=True)

    def gate_route(x, w, valid_rows):
        # sglang-jax's gate: FP32 logits of the BF16 hidden states.
        if weight_dtype:
            w = jax.ffi.ffi_call("tt.weight_dtype_override", jax.ShapeDtypeStruct(w.shape, w.dtype))(
                w, **{"ttcore.weight_dtype": weight_dtype}
            )
        return route(jnp.dot(x.astype(jnp.float32), w.astype(jnp.float32)), valid_rows)

    return gate_route


def _gate_problem(rng, rows, hidden, experts):
    # Values of at most 5 significant bits, exact in BF16, in BFP8 (whose
    # 16-value blocks share an exponent) and in a matmul at any fidelity, so
    # every sum is exact in FP32 in any order. The last two input columns, 1/2,
    # times two weight rows that tell the experts apart, block e // 16 and
    # e % 16 within it, by less than 1/64 together, separate equal sums of the
    # rest: the top k is unambiguous. The logits stay within about +-6, like a
    # router's.
    x = rng.integers(-4, 5, (rows, hidden)).astype(np.float32)
    w = rng.integers(-1, 2, (hidden, experts)).astype(np.float32) / 64
    x[:, -2:] = 0.5
    expert = np.arange(experts)
    w[-2] = (expert // 16) / 2**9
    w[-1] = (16 + expert % 16) / 2**14
    return x.astype(jnp.bfloat16), w


@pytest.mark.parametrize("trace", [False, True])
@pytest.mark.parametrize(
    "rows,hidden,experts,k,weight_dtype",
    [
        # Qwen3-30B-A3B decode: 2048 hidden, 128 experts, BFP8 gate weights.
        (1, 2048, 128, 8, "bfp_bf8"),
        (4, 2048, 128, 8, "bfp_bf8"),
        (32, 2048, 128, 8, "bfp_bf8"),
        # Qwen3.5-35B-A3B: 256 experts; other widths, k and weight dtypes.
        (1, 2048, 256, 8, "bfp_bf8"),
        (3, 1024, 64, 4, "bf16"),
        (2, 4096, 128, 8, None),
    ],
)
def test_moe_router_gate(trace, rows, hidden, experts, k, weight_dtype, tmp_path):
    run = jax.jit(
        _gate_router(k, weight_dtype),
        compiler_options={
            "optimization_level": "O1",
            "enable_trace": str(trace).lower(),
            "export_path": str(tmp_path),
        },
    )
    device = jax.devices("tt")[0]
    rng = np.random.default_rng(23)
    x, w = _gate_problem(rng, rows, hidden, experts)
    logits = x.astype(np.float64) @ w.astype(np.float64)
    for valid_rows in (rows, max(rows - 1, 0)):
        weights, ids = run(*jax.device_put((x, w, np.int32(valid_rows)), device))
        expected_weights, expected_ids = _reference(logits, k, valid_rows, True)
        # The ids check that the logits are exact, to well within the 2^-15
        # that separates experts. The router's softmax reads the k values
        # through the matrix unit's 19-bit source registers, which these
        # logits, unlike the other tests', do not fit exactly.
        np.testing.assert_array_equal(np.asarray(ids), expected_ids)
        np.testing.assert_allclose(np.asarray(weights), expected_weights, rtol=1e-2, atol=1e-4)

    ir = "\n".join(path.read_text() for path in (tmp_path / "irs").glob("ttnn_*.mlir"))
    assert "ttnn.matmul" not in ir, "the gate was not fused into the router"
    assert "operandSegmentSizes = array<i32: 1, 1, 1, 1>" in ir, "the router has no gate weight"
