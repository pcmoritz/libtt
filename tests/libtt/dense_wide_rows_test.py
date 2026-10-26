"""A few vocabulary-sized rows are computed as [..., H, W / 32, 32].

A decode batch of logits, [2, 151936], fills 2 of each 32 x 32 tile's rows, so
every pass over it moves 16 times its data. Sampling at temperature > 0 makes
some 25 such passes. The dense-rows pass (DenseWideRows.cpp) computes
connected [..., H, W] work on the dense shape instead, including the random
bits, the argmax and the branches of conditionals, and leaves it when the
reshapes in and out would cost more than they save. Each program here runs
with the pass on and off (LIBTT_DENSE_ROWS=0); results agree exactly, except
sums, whose order changes.
"""

import os
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

B, V = 2, 151936


def dense_ops(ir, rows=B, width=V):
    """The ops, traced ones included, that compute a [rows, width / 32, 32] result."""
    shape = f"tensor<{rows}x{width // 32}x32x"
    return [
        op
        for op, result in re.findall(r'"ttnn\.([a-z_0-9]+)"\(.*?->\s*(.*)', ir)
        if shape in result and op not in ("reshape", "to_layout", "copy")
    ]


def compile_and_run(fn, args, tmp_path, dense):
    previous = os.environ.get("LIBTT_DENSE_ROWS")
    os.environ["LIBTT_DENSE_ROWS"] = "1" if dense else "0"
    try:
        export = tmp_path / ("dense" if dense else "wide")
        run = jax.jit(fn, compiler_options={"enable_trace": "true", "export_path": str(export)})
        device = jax.devices("tt")[0]
        out = run(*(jax.device_put(x, device) for x in args))
        out = jax.tree.map(np.asarray, out)
    finally:
        if previous is None:
            del os.environ["LIBTT_DENSE_ROWS"]
        else:
            os.environ["LIBTT_DENSE_ROWS"] = previous
    (ir,) = [path.read_text() for path in (export / "irs").glob("ttnn_[0-9]*.mlir")]
    return out, ir


def both(fn, args, tmp_path):
    dense, dense_ir = compile_and_run(fn, args, tmp_path, dense=True)
    wide, wide_ir = compile_and_run(fn, args, tmp_path, dense=False)
    assert dense_ops(wide_ir) == []
    return dense, wide, dense_ir


def logits(dtype=jnp.bfloat16, seed=0):
    return (np.random.default_rng(seed).standard_normal((B, V), np.float32) * 3).astype(dtype)


def sample(l, t, key):
    # SGLang-JAX's temperature-only sampling: scaled logits, the Gumbel argmax
    # and the log-probabilities of the softmax.
    probs = jax.nn.softmax((l / t).astype(l.dtype), axis=-1)
    ids = jax.random.categorical(key, l.astype(jnp.float32) / t)
    return ids, jnp.log(probs).clip(min=jnp.finfo(probs.dtype).min)


def test_sampling_draws_the_same_tokens(tmp_path):
    key = jax.random.key_data(jax.random.key(7, impl="unsafe_rbg"))
    temperature = np.full((B, 1), 0.7, np.float32)

    def fn(l, t, k):
        return sample(l, t, jax.random.wrap_key_data(k, impl="unsafe_rbg"))

    (ids, logprobs), (wide_ids, wide_logprobs), ir = both(fn, (logits(), temperature, key), tmp_path)
    assert "rng_bit_generator" in dense_ops(ir)
    np.testing.assert_array_equal(ids, wide_ids)
    np.testing.assert_allclose(logprobs.astype(np.float32), wide_logprobs.astype(np.float32), atol=0.07, rtol=0.02)


def test_random_bits_are_the_same(tmp_path):
    key = jax.random.key_data(jax.random.key(3, impl="unsafe_rbg"))

    def fn(k):
        bits = jax.random.bits(jax.random.wrap_key_data(k, impl="unsafe_rbg"), (B, V))
        return jnp.argmax(bits >> 8, axis=-1), jnp.max(bits & 0xFFFF, axis=-1), jnp.min(bits >> 16, axis=-1)

    dense, wide, ir = both(fn, (key,), tmp_path)
    assert "rng_bit_generator" in dense_ops(ir)
    for a, w in zip(dense, wide):
        np.testing.assert_array_equal(a, w)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16], ids=["f32", "bf16"])
def test_argmax_and_argmin_pick_the_first_index(dtype, tmp_path):
    x = logits(dtype, seed=1)
    # Ties at the maximum and the minimum, an all -inf row, a NaN.
    x[0, [5, 70000, 150000]] = 100
    x[0, [9, 80000]] = -100
    x[1, :] = -np.inf
    x[1, 120000] = np.nan
    x[1, 130000] = np.nan

    def fn(v):
        v = v * 2
        return jnp.argmax(v, axis=-1), jnp.argmin(v, axis=-1), jnp.argmax(jnp.where(jnp.isnan(v), 0, v), axis=-1)

    dense, wide, ir = both(fn, (x,), tmp_path)
    want = jax.jit(fn, device=jax.devices("cpu")[0])(x)
    for a, w, c in zip(dense, wide, want):
        np.testing.assert_array_equal(a, c)
        np.testing.assert_array_equal(w, c)


def test_values_flow_through_conditionals_and_loops(tmp_path):
    # A conditional passes its operand through or computes on it, a loop
    # reads a value the conditional produced, and the result leaves densely
    # computed work as [B, V].
    def fn(l, flag, scale):
        x = l.astype(jnp.float32) * scale
        y = lax.cond(flag, lambda v: v * 2 + 1, lambda v: v, x)
        total = lax.fori_loop(0, 3, lambda i, acc: acc + jnp.sum(y[:, :1024]), 0.0)
        z = jnp.exp(y - jnp.max(y, axis=-1, keepdims=True))
        return z / jnp.sum(z, axis=-1, keepdims=True), total

    for flag in (False, True):
        args = (logits(), np.bool_(flag), np.float32(0.5))
        (probs, total), (wide_probs, wide_total), ir = both(fn, args, tmp_path / str(flag))
        assert dense_ops(ir)
        np.testing.assert_allclose(probs, wide_probs, rtol=1e-3, atol=1e-6)
        np.testing.assert_allclose(total, wide_total, rtol=1e-3)


def test_argmax_inside_a_conditional(tmp_path):
    def fn(l, flag):
        x = l.astype(jnp.float32) * 0.5 - 1.0
        return lax.cond(flag, lambda v: jnp.argmax(v, axis=-1), lambda v: jnp.argmin(v, axis=-1), x)

    for flag in (False, True):
        dense, wide, ir = both(fn, (logits(), np.bool_(flag)), tmp_path / str(flag))
        np.testing.assert_array_equal(dense, wide)


@pytest.mark.parametrize(
    "shape", [(64, V), (B, 8192), (B, 151937)], ids=["many_rows", "narrow", "unaligned"]
)
def test_other_shapes_stay_as_they_are(shape, tmp_path):
    x = np.random.default_rng(2).standard_normal(shape, np.float32)

    def fn(v):
        z = jnp.exp(v * 0.5 - jnp.max(v, axis=-1, keepdims=True))
        return jnp.log(z / jnp.sum(z, axis=-1, keepdims=True))

    out, ir = compile_and_run(fn, (x,), tmp_path, dense=True)
    assert dense_ops(ir, *shape) == []
    want = jax.nn.log_softmax(x * 0.5, axis=-1)
    np.testing.assert_allclose(out, want, atol=0.05)


def test_wide_mlps_keep_their_shape(tmp_path):
    # SwiGLU of Qwen3-14B's MLP: the TTIR fusion matches it in its own shape.
    rng = np.random.default_rng(4)
    x = (rng.standard_normal((1, 256)) * 0.1).astype(jnp.bfloat16)
    w = (rng.standard_normal((256, 2 * 17408)) * 0.1).astype(jnp.bfloat16)

    def fn(x, w):
        up, gate = jnp.split(x @ w, 2, axis=-1)
        return up * jax.nn.silu(gate)

    out, ir = compile_and_run(fn, (x, w), tmp_path, dense=True)
    assert dense_ops(ir, 1, 17408) == []


def test_one_pass_is_not_worth_the_reshapes(tmp_path):
    # Packing the input and unpacking the result would cost more than the one
    # elementwise op saves.
    out, ir = compile_and_run(lambda v: v * 3, (logits(),), tmp_path, dense=True)
    assert dense_ops(ir) == []
    np.testing.assert_array_equal(out, logits() * 3)
