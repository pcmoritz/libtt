"""tt.loop_mix: Inkling 2's loop-message controller (sglang_tpu_kernels.patch_inkling2_attn_res's loop_mix_tpu call)
runs as one ttnn.loop_mix and matches the model's math in float64 with its BF16 roundings (the normalized inputs,
hidden and the gate pre-activations): within 1e-2 of each row's scale everywhere and 2e-3 for all but 0.5% of the
elements, where an FP32 sum lands on the other side of a BF16 rounding boundary. With a norm weight it returns each
destination's next sub-layer input instead, bf16(rms(output) * weight), one [T, W] BF16 tensor per destination."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

EPS = 1e-6


def loop_mix(dest, messages, down, gain, forget):
    return jax.ffi.ffi_call("tt.loop_mix", jax.ShapeDtypeStruct(messages.shape, jnp.float32))(
        dest, messages, down, gain, forget, epsilon=np.float32(EPS)
    )


def loop_mix_normed(dest, messages, down, gain, forget, weight):
    shapes = tuple(jax.ShapeDtypeStruct(messages.shape[1:], jnp.bfloat16) for _ in range(messages.shape[0]))
    return jax.ffi.ffi_call("tt.loop_mix", shapes)(
        dest, messages, down, gain, forget, weight, epsilon=np.float32(EPS), has_norm=True, eps_norm=np.float32(EPS)
    )


def bf16(x):
    return np.asarray(jnp.asarray(x, jnp.float32).astype(jnp.bfloat16), np.float64)


def reference(dest, messages, down, gain, forget):
    d = np.asarray(dest, np.float64)
    m = np.asarray(messages, np.float64)
    d = np.broadcast_to(d, m.shape)
    rms = lambda x: bf16(x / np.sqrt(np.mean(x * x, -1, keepdims=True) + EPS))  # noqa: E731
    normed = np.concatenate([rms(d), rms(m)], -1)  # [Q, T, 2W]
    hidden = bf16(np.einsum("qtk,qkr->qtr", normed, np.asarray(down, np.float64)))
    g = np.tanh(bf16(np.einsum("qtr,qrw->qtw", hidden, np.asarray(gain, np.float64))))
    f = np.tanh(bf16(np.einsum("qtr,qrw->qtw", hidden, np.asarray(forget, np.float64))))
    return d + (g * m - f * d)


@pytest.mark.parametrize("trace", [False, True])
@pytest.mark.parametrize("queries,dests,tokens", [(3, 1, 1), (3, 3, 1), (1, 1, 1), (3, 1, 4)])
def test_loop_mix(trace, queries, dests, tokens, tmp_path):
    width, rank = 2304, 64
    run = jax.jit(loop_mix, compiler_options={"enable_trace": str(trace).lower(), "export_path": str(tmp_path)})
    device = jax.devices("tt")[0]
    rng = np.random.default_rng(queries * 10 + dests + tokens)
    dest = jnp.asarray(rng.normal(0, 3, (dests, tokens, width)), jnp.float32)
    messages = jnp.asarray(rng.normal(0, 1, (queries, tokens, width)), jnp.float32)
    down = jnp.asarray(rng.normal(0, 1 / np.sqrt(2 * width), (queries, 2 * width, rank)), jnp.bfloat16)
    gain = jnp.asarray(rng.normal(0, 1 / np.sqrt(rank), (queries, rank, width)), jnp.bfloat16)
    forget = jnp.asarray(rng.normal(0, 1 / np.sqrt(rank), (queries, rank, width)), jnp.bfloat16)
    args = [jax.device_put(a, device) for a in (dest, messages, down, gain, forget)]
    got = np.asarray(run(*args), np.float64)
    want = reference(dest, messages, down, gain, forget)
    scale = (np.abs(np.broadcast_to(np.asarray(dest), want.shape)) + np.abs(np.asarray(messages))).max(-1, keepdims=True)
    error = np.abs(got - want) / scale
    assert error.max() <= 1e-2, error.max()
    assert (error > 2e-3).mean() <= 5e-3, (error > 2e-3).mean()
    irs = [path.read_text() for path in (tmp_path / "irs").glob("ttnn_*.mlir")]
    assert irs and any('"ttnn.loop_mix"' in ir for ir in irs)


@pytest.mark.parametrize("trace", [False, True])
@pytest.mark.parametrize("queries,dests,tokens", [(3, 1, 1), (1, 1, 1), (3, 3, 2)])
def test_loop_mix_normed(trace, queries, dests, tokens, tmp_path):
    width, rank = 2304, 64
    run = jax.jit(
        loop_mix_normed, compiler_options={"enable_trace": str(trace).lower(), "export_path": str(tmp_path)}
    )
    device = jax.devices("tt")[0]
    rng = np.random.default_rng(queries * 10 + dests + tokens + 100)
    dest = jnp.asarray(rng.normal(0, 3, (dests, tokens, width)), jnp.float32)
    messages = jnp.asarray(rng.normal(0, 1, (queries, tokens, width)), jnp.float32)
    down = jnp.asarray(rng.normal(0, 1 / np.sqrt(2 * width), (queries, 2 * width, rank)), jnp.bfloat16)
    gain = jnp.asarray(rng.normal(0, 1 / np.sqrt(rank), (queries, rank, width)), jnp.bfloat16)
    forget = jnp.asarray(rng.normal(0, 1 / np.sqrt(rank), (queries, rank, width)), jnp.bfloat16)
    weight = jnp.asarray(1 + 0.2 * rng.normal(0, 1, (queries, width)), jnp.float32)
    args = [jax.device_put(a, device) for a in (dest, messages, down, gain, forget, weight)]
    outs = run(*args)
    assert len(outs) == queries and all(o.shape == (tokens, width) and o.dtype == jnp.bfloat16 for o in outs)
    got = np.stack([np.asarray(o, np.float64) for o in outs])
    mixed = reference(dest, messages, down, gain, forget)
    want = mixed / np.sqrt(np.mean(mixed * mixed, -1, keepdims=True) + EPS) * np.asarray(weight, np.float64)[:, None]
    scale = np.abs(want).max(-1, keepdims=True)
    # The mix's tolerance plus the output's BF16 rounding (2^-8 of each value).
    error = np.maximum(np.abs(got - want) - np.abs(want) * 2.0**-8, 0) / scale
    assert error.max() <= 1e-2, error.max()
    assert (error > 2e-3).mean() <= 5e-3, (error > 2e-3).mean()
    irs = [path.read_text() for path in (tmp_path / "irs").glob("ttnn_*.mlir")]
    assert irs and any('"ttnn.loop_mix"' in ir for ir in irs)
