"""tt.moe_compute runs TTNN's fused MoE kernel: each token's top-k SiLU-gated
experts, with the experts stored in BFP4.

Without a cluster axis every chip runs the experts its row of the expert
mapping places on it, for all of its tokens, and writes zeros for the slots of
experts held elsewhere; the layer's output is the sum over chips. With rings,
a chip runs its experts on several matmul rings side by side.

On a mesh, chips finish their experts at different times. The all-reduce that
sums them must not write into a chip that is still running the kernel, which
the uneven routing below provokes under trace.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.xla_metadata import set_xla_metadata
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

HIDDEN, INTERMEDIATE, K = 2048, 768, 8


def weight(w):
    return jax.ffi.ffi_call("tt.weight_dtype_override", jax.ShapeDtypeStruct(w.shape, w.dtype))(
        w, **{"ttcore.weight_dtype": "bf16"}
    )


def moe_compute(x, scores, ids, mapping, w0, w1, w2, *, rings):
    """x [T, H], scores and ids [T, k], mapping [chips, E], this chip's experts
    w0 and w1 [E_local, H, I] and w2 [E_local, I, H] (marked as weights by the
    caller, outside any shard_map): this chip's partial of each token's
    weighted sum of its experts, [T, H]."""
    tokens = x.shape[0]
    out = jax.ShapeDtypeStruct((K * tokens, HIDDEN), jnp.bfloat16)
    with set_xla_metadata(layer_id="0", output_height_shard_dim="4", rings=str(rings)):
        slots = jax.ffi.ffi_call("tt.moe_compute", out)(
            x[None], ids, scores, mapping, w0[None], w1[None], w2[None]
        )
    # Slot j of token t is row j * T + t.
    place = jnp.asarray(np.eye(tokens)[:, None, :], jnp.bfloat16)
    return (scores[:, :, None] * place).reshape(tokens, K * tokens) @ slots


def reference(x, scores, ids, w0, w1, w2, first_expert):
    out = np.zeros((x.shape[0], HIDDEN))
    for t in range(x.shape[0]):
        for j in range(K):
            e = int(ids[t, j]) - first_expert
            if 0 <= e < w0.shape[0]:
                g, u = x[t] @ w0[e], x[t] @ w1[e]
                out[t] += scores[t, j] * ((g / (1 + np.exp(-g))) * u) @ w2[e]
    return out


def experts(rng, count):
    w0, w1 = (rng.standard_normal((count, HIDDEN, INTERMEDIATE)) * HIDDEN**-0.5 for _ in range(2))
    w2 = rng.standard_normal((count, INTERMEDIATE, HIDDEN)) * INTERMEDIATE**-0.5
    return w0, w1, w2


def routing(rng, tokens, num_experts, *, favored=None):
    """Each token's k distinct experts and normalized scores; with `favored`,
    most of them from that range of experts."""
    ids = []
    for _ in range(tokens):
        if favored is None:
            ids.append(rng.choice(num_experts, K, replace=False))
        else:
            lo, hi = favored
            near = rng.choice(np.arange(lo, hi), K - 1, replace=False)
            far = rng.choice(np.setdiff1d(np.arange(num_experts), near), 1)
            ids.append(np.concatenate([near, far]))
    scores = rng.random((tokens, K))
    return np.stack(ids).astype(np.uint16), scores / scores.sum(1, keepdims=True)


def pcc(a, b):
    return np.corrcoef(a.ravel(), b.ravel())[0, 1]


@pytest.mark.parametrize("rings", [0, 3])
@pytest.mark.parametrize("tokens", [1, 3, 32])
def test_single_chip(tokens, rings):
    """32 of 128 experts on this chip; the token's others are elsewhere."""
    rng = np.random.default_rng(tokens)
    local = 32
    w0, w1, w2 = experts(rng, local)
    x = rng.standard_normal((tokens, HIDDEN))
    ids, scores = routing(rng, tokens, 4 * local)
    mapping = (np.arange(4 * local) // local).astype(np.uint16)[None]
    device = jax.devices("tt")[0]
    args = jax.device_put(
        tuple(np.asarray(v).astype(jnp.bfloat16) for v in (x, scores))
        + (ids, mapping)
        + tuple(w.astype(jnp.bfloat16) for w in (w0, w1, w2)),
        device,
    )
    out = jax.jit(
        lambda x, s, i, m, a, b, c: moe_compute(x, s, i, m, weight(a), weight(b), weight(c), rings=rings)
    )(*args)
    want = reference(x, scores, ids, w0, w1, w2, 0)
    # BFP4 weights: the outputs track a float64 reference to ~0.98.
    assert pcc(np.asarray(out).astype(np.float64), want) > 0.97


@pytest.mark.parametrize("rings", [0, 3])
@pytest.mark.parametrize("tokens", [1, 3, 8])
def test_mesh_uneven_experts_traced(tokens, rings):
    devices = jax.local_devices(backend="tt")
    if len(devices) != 4:
        pytest.skip("Requires four local TT devices")
    chips, local = 4, 32
    mesh = Mesh(np.array(devices).reshape(1, chips), ("data", "tensor"))
    rng = np.random.default_rng(100 + tokens)
    w0, w1, w2 = experts(rng, chips * local)
    x = rng.standard_normal((tokens, HIDDEN))
    # Seven of every token's experts on chip 0, which finishes last.
    ids, scores = routing(rng, tokens, chips * local, favored=(0, local))
    owner = (np.arange(chips * local) // local).astype(np.uint16)
    mapping = np.broadcast_to(owner, (chips, chips * local)).copy()

    def layer(x, scores, ids, mapping, w0, w1, w2):
        y = moe_compute(x, scores, ids, mapping, w0, w1, w2, rings=rings)
        return jax.lax.psum(y, "tensor")

    experts_spec = P("tensor", None, None)
    sharded = jax.shard_map(
        layer,
        mesh=mesh,
        in_specs=(P(), P(), P(), P()) + (experts_spec,) * 3,
        out_specs=P(),
        check_vma=False,
    )
    run = jax.jit(
        lambda x, s, i, m, a, b, c: sharded(x, s, i, m, weight(a), weight(b), weight(c)),
        compiler_options={"optimization_level": "O1", "enable_trace": "true"},
    )
    replicated = NamedSharding(mesh, P())
    args = tuple(
        jax.device_put(v, replicated)
        for v in (x.astype(jnp.bfloat16), scores.astype(jnp.bfloat16), ids, mapping)
    ) + tuple(jax.device_put(w.astype(jnp.bfloat16), NamedSharding(mesh, experts_spec)) for w in (w0, w1, w2))
    want = reference(x, scores, ids, w0, w1, w2, 0)
    for _ in range(3):
        out = np.asarray(run(*args)).astype(np.float64)
        assert pcc(out, want) > 0.97
