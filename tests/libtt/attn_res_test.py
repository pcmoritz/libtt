"""tt.attn_res: Inkling 2's block attention residuals (sglang_tpu_kernels.patch_inkling2_attn_res's call) runs as
one ttnn.attn_res and matches the math in float64: for each query row q and token t, softmax over the sources of
<s_i, q> / rms(s_i), the weighted sum of the sources (normalized or raw), then with a weight the RMS norm times
the weight (BF16, within a BF16 rounding) or without one the FP32 mixture."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

EPS_RES, EPS_NORM = 1e-6, 1e-6


def attn_res(sources, addend, queries, weight, *, normalize):
    tokens, width = sources[0].shape
    dtype = jnp.bfloat16 if weight is not None else jnp.float32
    operands = [*sources, *([addend] if addend is not None else []), queries, *([weight] if weight is not None else [])]
    return jax.ffi.ffi_call("tt.attn_res", jax.ShapeDtypeStruct((queries.shape[0], tokens, width), dtype))(
        *operands,
        num_sources=np.int64(len(sources)),
        has_addend=addend is not None,
        has_weight=weight is not None,
        normalize=normalize,
        eps_res=np.float32(EPS_RES),
        eps_norm=np.float32(EPS_NORM),
    )


def reference(sources, addend, queries, weight, normalize):
    v = np.stack([np.asarray(s, np.float64) for s in sources])  # [n, T, W]
    if addend is not None:
        v[-1] += np.asarray(addend, np.float64)
    r = 1 / np.sqrt(np.mean(v * v, axis=-1, keepdims=True) + EPS_RES)
    logits = np.einsum("ntw,qw->qtn", v, np.asarray(queries, np.float64)) * r[..., 0].T[None]
    p = np.exp(logits - logits.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    mix = np.einsum("qtn,ntw->qtw", p, v * r if normalize else v)
    if weight is None:
        return mix
    return mix / np.sqrt(np.mean(mix * mix, axis=-1, keepdims=True) + EPS_NORM) * np.asarray(weight, np.float64)[:, None]


@pytest.mark.parametrize("trace", [False, True])
@pytest.mark.parametrize(
    "sources,queries,tokens,addend,weight,normalize",
    [
        # Inkling-2-Mini decode: a local layer's attention input, an MLP input with the attention output added, a
        # global layer's Q, K and V inputs; four tokens.
        (3, 1, 1, False, True, True),
        (4, 1, 1, True, True, True),
        (5, 3, 1, False, True, True),
        (8, 3, 4, True, True, True),
        # The mixtures alone (later loop passes, loop messages), raw values.
        (4, 3, 2, True, False, True),
        (3, 1, 1, False, True, False),
    ],
)
def test_attn_res(trace, sources, queries, tokens, addend, weight, normalize, tmp_path):
    width = 2304
    run = jax.jit(
        lambda s, a, q, w: attn_res(s, a, q, w, normalize=normalize),
        compiler_options={"enable_trace": str(trace).lower(), "export_path": str(tmp_path)},
    )
    device = jax.devices("tt")[0]
    rng = np.random.default_rng(sources * 100 + queries * 10 + tokens)
    s = [jnp.asarray(rng.normal(0, 2.0 ** (j % 4 - 1), (tokens, width)), jnp.bfloat16) for j in range(sources)]
    a = jnp.asarray(rng.normal(0, 0.5, (tokens, width)), jnp.bfloat16) if addend else None
    # Each query leans towards a source's first token (a logit of about 3), so the softmax is far from uniform.
    q = np.stack([np.asarray(s[k % sources][0], np.float64) for k in range(queries)])
    q = 3 * q / np.linalg.norm(q, axis=-1, keepdims=True) / np.sqrt(width)
    q = jnp.asarray(q + rng.normal(0, 0.01 / width, q.shape), jnp.float32)
    w = jnp.asarray(rng.normal(1, 0.2, (queries, width)), jnp.float32) if weight else None
    put = lambda x: None if x is None else jax.device_put(x, device)  # noqa: E731
    got = np.asarray(run([put(x) for x in s], put(a), put(q), put(w)), np.float64)
    want = reference(s, a, q, w, normalize)
    if weight:
        np.testing.assert_allclose(got, want, rtol=2**-7, atol=1e-6)
    else:
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6 * np.abs(want).max())
    irs = [path.read_text() for path in (tmp_path / "irs").glob("ttnn_*.mlir")]
    assert irs and any('"ttnn.attn_res"' in ir for ir in irs)
