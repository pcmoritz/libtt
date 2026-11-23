"""tt.moe_route: a router's top-k selection (sglang_tpu_kernels.patch_inkling2_route's call) runs as one
ttnn.moe_route: ids = top k of sigmoid(logits) + bias (best first, the lower expert first among equal keys), weights
the selected sigmoid scores renormalized times a scale, rows from num_valid on weight 0 and id -1; ids exactly and
weights within 1e-5 of a float64 reference."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def route(logits, bias, num_valid, *, k, scale):
    tokens = logits.shape[0]
    operands = [logits, bias, *([num_valid] if num_valid is not None else [])]
    return jax.ffi.ffi_call(
        "tt.moe_route",
        (jax.ShapeDtypeStruct((tokens, k), jnp.float32), jax.ShapeDtypeStruct((tokens, k), jnp.int32)),
    )(*operands, k=np.int64(k), scale=np.float32(scale), has_num_valid=num_valid is not None)


@pytest.mark.parametrize("trace", [False, True])
@pytest.mark.parametrize("tokens,valid", [(1, None), (4, None), (4, 3), (1, 1)])
def test_moe_route(trace, tokens, valid, tmp_path):
    experts, k, scale = 128, 8, 2.5
    run = jax.jit(
        lambda l, b, n: route(l, b, n, k=k, scale=scale),
        compiler_options={"enable_trace": str(trace).lower(), "export_path": str(tmp_path)},
    )
    device = jax.devices("tt")[0]
    rng = np.random.default_rng(tokens * 10 + (valid or 0))
    logits = rng.normal(0, 3, (tokens, experts)).astype(np.float32)
    bias = rng.normal(0, 0.1, experts).astype(np.float32)
    num_valid = None if valid is None else jax.device_put(jnp.asarray([valid], jnp.int32), device)
    weights, ids = run(jax.device_put(jnp.asarray(logits), device), jax.device_put(jnp.asarray(bias), device), num_valid)
    weights, ids = np.asarray(weights, np.float64), np.asarray(ids)
    s = 1 / (1 + np.exp(-logits.astype(np.float64)))
    keys = (s.astype(np.float32) + bias).astype(np.float64)
    for t in range(tokens):
        if valid is not None and t >= valid:
            assert (ids[t] == -1).all() and (weights[t] == 0).all()
            continue
        order = np.argsort(-keys[t], kind="stable")[:k]
        np.testing.assert_array_equal(ids[t], order)
        np.testing.assert_allclose(weights[t], s[t, order] / s[t, order].sum() * scale, rtol=1e-5)
    irs = [path.read_text() for path in (tmp_path / "irs").glob("ttnn_*.mlir")]
    assert irs and any('"ttnn.moe_route"' in ir for ir in irs)
