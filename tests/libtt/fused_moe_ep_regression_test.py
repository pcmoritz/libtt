"""tt.fused_moe_ep runs a mixture-of-experts layer's experts on each device.

Called per device in a shard_map over the mesh, with the tokens, their top-k
routing weights and expert ids (negative for padding, which is skipped) and the
device's experts, it returns each token's routing-weighted sum of the
SiLU-gated experts that device holds, computed by TTNN's fused MoE kernel with
the experts in BFP4, or in BF16 with weight_dtype="bf16". The experts are split
evenly over the mesh in device order, so the layer is the sum over devices.
Batches of more than 512 tokens take several kernel calls.

With weight_dtype="fp8" the kernel reads FP8 experts as they are and applies
their scales, one per expert and output column, as SGLang-TorchTPU passes them.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax_plugins.libtt import sglang_tpu_kernels

HIDDEN, INTERMEDIATE, K = 2048, 768, 8


def fused_moe_ep(x, weights, ids, w1, w3, w2, **attributes):
    out = jax.ShapeDtypeStruct(x.shape, x.dtype, manual_axis_type=jax.typeof(x).manual_axis_type)
    return jax.ffi.ffi_call("tt.fused_moe_ep", out)(x, weights, ids, w1, w3, w2, **attributes)


def weight(w):
    """Marks w as a weight, prepared once rather than per call."""
    return jax.ffi.ffi_call("tt.weight_dtype_override", jax.ShapeDtypeStruct(w.shape, w.dtype))(
        w, **{"ttcore.weight_dtype": "bf16"}
    )


def problem(rng, tokens, experts, padding):
    w1, w3 = ((rng.standard_normal((experts, HIDDEN, INTERMEDIATE)) * HIDDEN**-0.5) for _ in range(2))
    w2 = rng.standard_normal((experts, INTERMEDIATE, HIDDEN)) * INTERMEDIATE**-0.5
    x = rng.standard_normal((tokens, HIDDEN))
    ids = np.stack([rng.choice(experts, K, replace=False) for _ in range(tokens)]).astype(np.int32)
    # Any negative id is padding, also one whose low 16 bits are an expert's.
    ids[tokens - padding :] = rng.choice([-1, -65536, np.iinfo(np.int32).min], (padding, K))
    weights = rng.random((tokens, K)).astype(np.float32)
    weights /= weights.sum(1, keepdims=True)
    bf16 = lambda a: np.asarray(a).astype(jnp.bfloat16)
    return bf16(x), weights, ids, bf16(w1), bf16(w3), bf16(w2)


def bfp4(w):
    """w rounded as the kernel stores it: blocks of 16 along the last axis share
    an exponent, and each value keeps a sign and 3 magnitude bits."""
    w = np.asarray(w).astype(np.float64)
    blocks = w.reshape(*w.shape[:-1], -1, 16)
    top = np.abs(blocks).max(-1, keepdims=True)
    step = 2.0 ** (np.floor(np.log2(np.where(top > 0, top, 1))) - 2)
    return (np.sign(blocks) * np.clip(np.round(np.abs(blocks) / step), 0, 7) * step).reshape(w.shape)


def reference(x, weights, ids, w1, w3, w2, stored=bfp4):
    x = np.asarray(x).astype(np.float64)
    w1, w3, w2 = (stored(w) for w in (w1, w3, w2))
    out = np.zeros(x.shape)
    for e in range(w1.shape[0]):
        t, j = np.nonzero(ids == e)
        g, u = x[t] @ w1[e], x[t] @ w3[e]
        np.add.at(out, t, weights[t, j, None] * ((g / (1 + np.exp(-g))) * u) @ w2[e])
    return out


def check(out, want, padding, tolerance):
    # Padding is skipped, so padding tokens' outputs are zero. Against the
    # experts as stored the outputs agree to a relative error of about 0.012
    # with BFP4 experts, the rest of the kernel accumulating in FP32.
    rows = want.shape[0] - padding
    got = np.asarray(out).astype(np.float64)
    assert not got[rows:].any()
    error = np.linalg.norm(got[:rows] - want[:rows]) / np.linalg.norm(want[:rows])
    assert error < tolerance, error


# 513 tokens end in a one-token kernel call, 600 in a call of 88.
@pytest.mark.parametrize("weight_dtype", ["bfp4", "bf16"])
@pytest.mark.parametrize("tokens,padding", [(1, 0), (3, 1), (5, 2), (513, 0), (600, 7)])
def test_mesh(tokens, padding, weight_dtype):
    devices = jax.local_devices(backend="tt")
    if len(devices) not in (1, 4):
        pytest.skip("Requires one or four local TT devices")
    mesh = Mesh(np.array(devices).reshape(1, len(devices)), ("data", "tensor"))
    axes = ("data", "tensor")
    rng = np.random.default_rng(100 + tokens)
    args = problem(rng, tokens, len(devices) * 8, padding)

    attributes = {} if weight_dtype == "bfp4" else {"weight_dtype": weight_dtype}

    def layer(x, weights, ids, w1, w3, w2):
        return jax.lax.psum(fused_moe_ep(x, weights, ids, w1, w3, w2, **attributes), axes)

    split = P(axes, None, None)
    sharded = jax.shard_map(
        layer, mesh=mesh, in_specs=(P(),) * 3 + (split,) * 3, out_specs=P(), check_vma=False
    )
    run = jax.jit(lambda x, w, i, a, b, c: sharded(x, w, i, weight(a), weight(b), weight(c)))
    shardings = [NamedSharding(mesh, P())] * 3 + [NamedSharding(mesh, split)] * 3
    out = run(*(jax.device_put(a, s) for a, s in zip(args, shardings)))
    assert out.shape == (tokens, HIDDEN) and out.dtype == jnp.bfloat16
    if weight_dtype == "bfp4":
        check(out, reference(*args), padding, 0.03)
    else:
        # BF16 experts as they are: the intermediate activations and the
        # outputs are rounded to BF16.
        check(out, reference(*args, stored=lambda w: np.asarray(w).astype(np.float64)), padding, 0.01)


def fp8_channelwise(w):
    """w ([experts, k, n]) as FP8 values and one FP32 scale per expert and
    output column, [experts, 1, 1, n], as SGLang-TorchTPU prepares FP8 experts."""
    w = np.asarray(w).astype(np.float32)
    scale = np.abs(w).max(1, keepdims=True) / float(jnp.finfo(jnp.float8_e4m3fn).max)
    return (w / scale).astype(jnp.float8_e4m3fn), scale[:, None]


@pytest.mark.parametrize("tokens,padding", [(1, 0), (5, 2), (600, 7)])
def test_fp8_experts(tokens, padding):
    devices = jax.local_devices(backend="tt")
    if len(devices) not in (1, 4):
        pytest.skip("Requires one or four local TT devices")
    mesh = Mesh(np.array(devices).reshape(1, len(devices)), ("data", "tensor"))
    axes = ("data", "tensor")
    rng = np.random.default_rng(200 + tokens)
    x, weights, ids, w1, w3, w2 = problem(rng, tokens, len(devices) * 8, padding)
    # SGLang-TorchTPU's layout: gate and up projections side by side.
    (w13, s13), (w2, s2) = fp8_channelwise(np.concatenate([w1, w3], -1)), fp8_channelwise(w2)

    def layer(x, weights, ids, w13, s13, w2, s2):
        experts = sglang_tpu_kernels._fp8_experts(w13, s13, w2, s2)
        return jax.lax.psum(fused_moe_ep(x, weights, ids, *experts, weight_dtype="fp8"), axes)

    split3, split4 = P(axes, None, None), P(axes, None, None, None)
    specs = (split3, split4, split3, split4)
    sharded = jax.shard_map(layer, mesh=mesh, in_specs=(P(),) * 3 + specs, out_specs=P(), check_vma=False)
    run = jax.jit(lambda x, w, i, *experts: sharded(x, w, i, *(weight(e) for e in experts)))
    shardings = [NamedSharding(mesh, P())] * 3 + [NamedSharding(mesh, spec) for spec in specs]
    out = run(*(jax.device_put(a, s) for a, s in zip((x, weights, ids, w13, s13, w2, s2), shardings)))
    assert out.shape == (tokens, HIDDEN) and out.dtype == jnp.bfloat16

    # The experts as the checkpoint means them: the FP8 values times their scales.
    stored13 = w13.astype(np.float64) * s13[:, 0]
    stored2 = w2.astype(np.float64) * s2[:, 0]
    stored = (x, weights, ids, stored13[..., :INTERMEDIATE], stored13[..., INTERMEDIATE:], stored2)
    check(out, reference(*stored, stored=lambda w: w), padding, 0.01)
