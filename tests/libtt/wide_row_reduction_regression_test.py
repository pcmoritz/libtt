"""Reductions over long rows split into groups.

A sum, max or min over a last dimension of at least 16384 with at most one
tile row of rows (a decode batch) used one core (264 us to sum two rows of a
37984-entry vocab shard). tt-mlir now reshapes such rows to [..., W / 32, 32],
reduces the groups of 32 and then the groups. More rows already reduce on a
core per tile row and stay a single reduction (splitting prefill logprob
reductions made serving hang intermittently).

Rows of at most 16 and at least 32768 columns go further: the reduction runs on
the view [..., W / C, C] (C a multiple of 32 dividing W), whose tiles are full,
and so do the elementwise ops that compute its input. A sampler's masked sums
over the vocabulary, sum(where(probs >= cutoff, probs, 0)), then move and
compute 1/32 of the data: SGLang-TorchTPU's 30-round top-p cutoff search over
a [1, 151936] row went from 8.3 to 2.1 ms.
"""

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

REDUCTIONS = {
    "sum": (jnp.sum, np.sum),
    "max": (jnp.max, np.max),
    "min": (jnp.min, np.min),
}


def reduction_inputs(tmp_path, name):
    """The input shapes of the `name` reductions in the final runtime IR."""
    irs = list((tmp_path / "irs").glob("ttnn_runtime_*.mlir"))
    assert len(irs) == 1, irs
    return [
        tuple(int(d) for d in shape.split("x"))
        for shape in re.findall(rf'"ttnn\.{name}"\(%\w+\).*?: \(tensor<([0-9x]+)x[a-z]', irs[0].read_text())
    ]


def integer_rows(shape, dtype=np.float32, seed=0):
    # Small integers: every sum is exact in FP32, so a missing group or mixed
    # rows changes the result.
    return np.random.default_rng(seed).integers(-2, 3, shape).astype(dtype)


def view_width(width):
    """The widest multiple of 32 up to 1024 dividing `width` into at least 32 groups."""
    return next(c for c in (1024, 512, 256, 128, 64, 32) if width % c == 0 and width // c >= 32)


@pytest.mark.parametrize("keepdims", [False, True])
@pytest.mark.parametrize("name", list(REDUCTIONS))
@pytest.mark.parametrize(
    "shape, split",
    [
        ((2, 37984), "view"),  # Two rows of a Qwen3 vocab shard at TP4: groups of 32.
        ((1, 151936), "view"),  # Groups of 128.
        ((1, 248320), "view"),  # Groups of 512.
        ((2, 32768), "view"),  # The narrowest view: 32 groups of 1024.
        ((16, 37984), "view"),
        ((17, 37984), "split"),  # More than half a tile row.
        ((32, 37984), "split"),  # One full tile row.
        ((33, 37984), "single"),  # Two tile rows.
        ((3, 2, 16384), "split"),  # Six rows over two leading dimensions.
        ((2, 32736), "split"),  # Below 32768 columns.
        ((2, 16384), "split"),  # 512 groups: the narrowest split.
        ((2, 16352), "single"),  # 511 groups.
        ((2, 16400), "single"),  # Not a multiple of 32.
        ((2, 5120), "single"),  # A hidden size: RMSNorm fusion sees a single reduction.
        ((512, 37984), "single"),  # Prefill logprobs.
    ],
)
def test_split_eligibility(shape, split, name, keepdims, tmp_path):
    reduce, reference = REDUCTIONS[name]
    run = jax.jit(lambda a: reduce(a, axis=-1, keepdims=keepdims), compiler_options={"export_path": str(tmp_path)})
    x = integer_rows(shape)
    out = np.asarray(run(jax.device_put(x, jax.devices("tt")[0])))
    np.testing.assert_array_equal(out, reference(x, axis=-1, keepdims=keepdims))

    *rows, width = shape
    if split == "view":
        expected = [(*rows, width // view_width(width), view_width(width))]
    elif split == "split":
        expected = [(*rows, width // 32, 32), (*rows, width // 32)]
    else:
        expected = [shape]
    assert reduction_inputs(tmp_path, name) == expected


@pytest.mark.parametrize("name", list(REDUCTIONS))
@pytest.mark.parametrize("shape", [(2, 37984), (1, 151936), (32, 37984), (3, 2, 16384)])
def test_split_accuracy(shape, name):
    # Random values: the split must keep FP32 sums as accurate as one
    # reduction. The tolerance scales with each row's own magnitude.
    reduce, reference = REDUCTIONS[name]
    x = np.random.default_rng(1).standard_normal(shape).astype(np.float32)
    out = np.asarray(jax.jit(lambda a: reduce(a, axis=-1))(jax.device_put(x, jax.devices("tt")[0])))
    expected = reference(x.astype(np.float64), axis=-1)
    if name == "sum":
        error = np.abs(out - expected)
        tolerance = 1e-6 * np.abs(x).sum(axis=-1)
        assert (error <= tolerance).all(), (error, tolerance)
    else:
        np.testing.assert_array_equal(out, expected)


@pytest.mark.parametrize("name", ["sum", "max"])
def test_split_reduction_traced(name):
    # A traced decode step replays the split program with new inputs.
    reduce, reference = REDUCTIONS[name]
    run = jax.jit(lambda a: reduce(a, axis=-1), compiler_options={"enable_trace": "true"})
    device = jax.devices("tt")[0]
    for i in range(20):
        x = integer_rows((2, 37984), seed=i)
        np.testing.assert_array_equal(np.asarray(run(jax.device_put(x, device))), reference(x, axis=-1), err_msg=f"run {i}")


def op_inputs(tmp_path, name):
    """The first operand shapes of the `name` ops in the final runtime IR."""
    irs = list((tmp_path / "irs").glob("ttnn_runtime_*.mlir"))
    assert len(irs) == 1, irs
    return [
        tuple(int(d) for d in shape.split("x"))
        for shape in re.findall(rf'"ttnn\.{name}"\(%\w+[^:]*: \(tensor<([0-9x]+)x[a-z]', irs[0].read_text())
    ]


@pytest.mark.parametrize("shape", [(1, 151936), (4, 248320), (2, 37984)])
def test_view_computes_the_reduced_values(shape, tmp_path):
    # The elementwise ops feeding the reductions run on the view too, with a
    # cutoff per row, a row of offsets and a scalar broadcast into it.
    def fn(x, cutoff, offset):
        kept = x >= cutoff[:, None]
        mass = jnp.sum(jnp.where(kept, x, 0.0), axis=-1)
        count = jnp.sum(kept, axis=-1, dtype=jnp.int32)
        peak = jnp.max(x * 2.0 + offset, axis=-1, keepdims=True)
        return mass, count, peak

    *rows, width = shape
    x = integer_rows(shape)
    cutoff = np.arange(len(x.reshape(-1, width)), dtype=np.float32).reshape(rows) % 3 - 1
    offset = integer_rows((1, width), seed=1)
    device = jax.devices("tt")[0]
    got = jax.jit(fn, compiler_options={"export_path": str(tmp_path)})(
        *[jax.device_put(a, device) for a in (x, cutoff, offset)])
    for g, w in zip(got, fn(x, cutoff, offset)):
        np.testing.assert_array_equal(np.asarray(g), np.asarray(w))
    view = (*rows, width // view_width(width), view_width(width))
    for name in ("ge", "where", "multiply", "add"):
        assert op_inputs(tmp_path, name) and all(s == view for s in op_inputs(tmp_path, name)), name


def filter_probs(probs, top_ks, top_ps, need_top_k, need_top_p):
    """sglang_tpu.sampling_mask's sort-free top-k / top-p filter (SGLang-TorchTPU)."""
    vocab = probs.shape[-1]

    def keeps_enough(threshold):
        kept = probs >= threshold[:, None]
        enough = None
        if need_top_k:
            enough = jnp.sum(kept, axis=-1, dtype=jnp.int32) >= jnp.clip(top_ks, 1, vocab)
        if need_top_p:
            mass = jnp.sum(jnp.where(kept, probs, 0.0), axis=-1) >= top_ps
            enough = mass if enough is None else enough | mass
        return enough

    def step(i, bits):
        candidate = bits | jnp.left_shift(1, 29 - i)
        return jnp.where(keeps_enough(jax.lax.bitcast_convert_type(candidate, jnp.float32)), candidate, bits)

    bits = jax.lax.fori_loop(0, 30, step, jnp.zeros(probs.shape[0], jnp.int32))
    cutoff = jax.lax.bitcast_convert_type(bits, jnp.float32)
    return jnp.where(probs >= cutoff[:, None], probs, 0.0), cutoff


@pytest.mark.parametrize("shape", [(1, 151936), (4, 248320)])
@pytest.mark.parametrize("need_top_k, need_top_p", [(False, True), (True, False), (True, True)])
def test_sampling_filter_matches_cpu(shape, need_top_k, need_top_p):
    # The unrolled search's 30 masked sums and counts run on the view and keep
    # exactly the tokens the CPU keeps, with near-ties at the top.
    rng = np.random.default_rng(2)
    logits = rng.normal(0, 2.5, shape).astype(np.float32)
    for i in range(shape[0]):
        top = np.argmax(logits[i])
        logits[i, (top + 7) % shape[1]] = logits[i, top] - 10.0 ** -(i + 3)
    probs = np.asarray(jax.nn.softmax(jnp.asarray(logits) / 0.7, axis=-1))
    top_ks = np.full(shape[0], 20, np.int32)
    top_ps = np.linspace(0.5, 0.95, shape[0]).astype(np.float32)
    fn = lambda p, k, t: filter_probs(p, k, t, need_top_k, need_top_p)  # noqa: E731
    device = jax.devices("tt")[0]
    got = jax.jit(fn, compiler_options={"enable_trace": "true"})(
        *[jax.device_put(a, device) for a in (probs, top_ks, top_ps)])
    want = jax.jit(fn, backend="cpu")(probs, top_ks, top_ps)
    np.testing.assert_array_equal(np.asarray(got[1]), np.asarray(want[1]))
    np.testing.assert_array_equal(np.asarray(got[0]) > 0, np.asarray(want[0]) > 0)
