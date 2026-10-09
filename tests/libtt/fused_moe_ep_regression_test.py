"""tt.fused_moe_ep runs a mixture-of-experts layer's experts on each device.

Called per device with the tokens, their top-k routing weights and expert ids
(negative for padding, which is skipped) and the device's experts, it returns
each token's routing-weighted sum of the SiLU-gated experts that device holds,
computed by TTNN's fused MoE kernel with the experts in BFP4. The experts are
split evenly over the mesh in device order, so on a mesh the layer is the sum
over devices. Batches of more than 512 tokens take several kernel calls.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

HIDDEN, INTERMEDIATE, K = 2048, 768, 8


def fused_moe_ep(x, weights, ids, w1, w3, w2):
    out = jax.ShapeDtypeStruct(x.shape, x.dtype, manual_axis_type=jax.typeof(x).manual_axis_type)
    return jax.ffi.ffi_call("tt.fused_moe_ep", out)(x, weights, ids, w1, w3, w2)


def problem(rng, tokens, experts, padding):
    w1, w3 = ((rng.standard_normal((experts, HIDDEN, INTERMEDIATE)) * HIDDEN**-0.5) for _ in range(2))
    w2 = rng.standard_normal((experts, INTERMEDIATE, HIDDEN)) * INTERMEDIATE**-0.5
    x = rng.standard_normal((tokens, HIDDEN))
    ids = np.stack([rng.choice(experts, K, replace=False) for _ in range(tokens)]).astype(np.int32)
    ids[tokens - padding :] = -1
    weights = rng.random((tokens, K)).astype(np.float32)
    weights /= weights.sum(1, keepdims=True)
    bf16 = lambda a: np.asarray(a).astype(jnp.bfloat16)
    return bf16(x), weights, ids, bf16(w1), bf16(w3), bf16(w2)


def reference(x, weights, ids, w1, w3, w2):
    x, w1, w3, w2 = (np.asarray(a).astype(np.float64) for a in (x, w1, w3, w2))
    out = np.zeros(x.shape)
    for t in range(x.shape[0]):
        for weight, e in zip(weights[t], ids[t]):
            if e >= 0:
                g, u = x[t] @ w1[e], x[t] @ w3[e]
                out[t] += weight * ((g / (1 + np.exp(-g))) * u) @ w2[e]
    return out


def check(out, want, padding):
    # Padding is skipped, so padding tokens' outputs are zero. BFP4 experts
    # track a float64 reference to a correlation of about 0.98.
    rows = want.shape[0] - padding
    got = np.asarray(out).astype(np.float64)
    assert not got[rows:].any()
    assert np.corrcoef(got[:rows].ravel(), want[:rows].ravel())[0, 1] > 0.97


@pytest.mark.parametrize("tokens,padding", [(1, 0), (3, 1), (600, 7)])
def test_single_device(tokens, padding):
    rng = np.random.default_rng(tokens)
    args = problem(rng, tokens, 16, padding)
    out = jax.jit(fused_moe_ep)(*jax.device_put(args, jax.devices("tt")[0]))
    assert out.shape == (tokens, HIDDEN) and out.dtype == jnp.bfloat16
    check(out, reference(*args), padding)


@pytest.mark.parametrize("tokens,padding", [(1, 0), (5, 2), (600, 7)])
def test_mesh(tokens, padding):
    devices = jax.local_devices(backend="tt")
    if len(devices) != 4:
        pytest.skip("Requires four local TT devices")
    mesh = Mesh(np.array(devices).reshape(1, 4), ("data", "tensor"))
    axes = ("data", "tensor")
    rng = np.random.default_rng(100 + tokens)
    args = problem(rng, tokens, 4 * 8, padding)

    def layer(x, weights, ids, w1, w3, w2):
        return jax.lax.psum(fused_moe_ep(x, weights, ids, w1, w3, w2), axes)

    split = P(axes, None, None)
    run = jax.jit(
        jax.shard_map(
            layer, mesh=mesh, in_specs=(P(),) * 3 + (split,) * 3, out_specs=P(), check_vma=False
        )
    )
    shardings = [NamedSharding(mesh, P())] * 3 + [NamedSharding(mesh, split)] * 3
    out = run(*(jax.device_put(a, s) for a, s in zip(args, shardings)))
    check(out, reference(*args), padding)
