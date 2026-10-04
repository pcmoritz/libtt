"""Reductions over long rows split into groups of 32.

A sum, max or min over a last dimension of at least 16384 with at most one
tile row of rows (a decode batch) used one core (264 us to sum two rows of a
37984-entry vocab shard). tt-mlir now reshapes such rows to [..., W / 32, 32],
reduces the groups of 32 and then the groups. More rows already reduce on a
core per tile row and stay a single reduction (splitting prefill logprob
reductions made serving hang intermittently).
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


@pytest.mark.parametrize("keepdims", [False, True])
@pytest.mark.parametrize("name", list(REDUCTIONS))
@pytest.mark.parametrize(
    "shape, split",
    [
        ((2, 37984), True),  # Two rows of a Qwen3 vocab shard at TP4.
        ((1, 151936), True),
        ((32, 37984), True),  # One full tile row.
        ((33, 37984), False),  # Two tile rows.
        ((3, 2, 16384), True),  # Six rows over two leading dimensions.
        ((2, 16384), True),  # 512 groups: the narrowest split.
        ((2, 16352), False),  # 511 groups.
        ((2, 16400), False),  # Not a multiple of 32.
        ((2, 5120), False),  # A hidden size: RMSNorm fusion sees a single reduction.
        ((512, 37984), False),  # Prefill logprobs.
    ],
)
def test_split_eligibility(shape, split, name, keepdims, tmp_path):
    reduce, reference = REDUCTIONS[name]
    run = jax.jit(lambda a: reduce(a, axis=-1, keepdims=keepdims), compiler_options={"export_path": str(tmp_path)})
    x = integer_rows(shape)
    out = np.asarray(run(jax.device_put(x, jax.devices("tt")[0])))
    np.testing.assert_array_equal(out, reference(x, axis=-1, keepdims=keepdims))

    *rows, width = shape
    groups = width // 32
    expected = [(*rows, groups, 32), (*rows, groups)] if split else [shape]
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
