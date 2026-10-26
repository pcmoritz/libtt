"""Conditionals nested in a branch select on host, without waiting for the device.

The runtime reads a case index on host. A conditional nested in a branch
received its predicate on device, and a lax.switch computes its index from its
predicates on device, so reading the index waited for all the work queued
before it and the host stopped running ahead of the device. SGLang-JAX's
sampler at temperature > 0 has both inside its sampling branch: a lax.switch
over its top-p and top-k flags and a min-p conditional. Serving Qwen3-8B on one
chip spent about 2 ms per token on the wait.

A case whose index is computed from entry predicates now becomes nested
two-way cases on the predicates (tt_mlir_control_flow.cpp), and a branch
program keeps on host the arguments that only reach a nested case's index
(tt_mlir_case_selector_in_outlined_program.patch).
"""

import itertools
import re
import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

SOURCE = (np.arange(64 * 32, dtype=np.float32).reshape(64, 32) % 7) - 3


def _put(value):
    return jax.device_put(np.asarray(value), jax.devices("tt")[0])


def sampler_like(x, greedy, use_p, use_k, use_min_p):
    def sample(v):
        index = jnp.asarray(use_p, jnp.int32) + 2 * jnp.asarray(use_k, jnp.int32)
        v = lax.switch(index, [lambda u: u, lambda u: u * 2, lambda u: u + 3, lambda u: u * 2 + 3], v)
        v = lax.cond(use_min_p, lambda u: u - 1, lambda u: u, v)
        return v * 0.5

    return lax.cond(greedy, lambda v: v + 100, sample, x)


def expected(x, greedy, use_p, use_k, use_min_p):
    if greedy:
        return x + 100
    v = [x, x * 2, x + 3, x * 2 + 3][int(use_p) + 2 * int(use_k)]
    return (v - 1 if use_min_p else v) * 0.5


@pytest.mark.parametrize("trace", ["true", "false"])
def test_every_combination_of_nested_predicates(trace):
    run = jax.jit(sampler_like, compiler_options={"optimization_level": "O1", "enable_trace": trace})
    bools = {value: _put(np.bool_(value)) for value in (False, True)}
    # Twice over, so each combination also replays traces the others captured.
    combinations = list(itertools.product((False, True), repeat=4)) * 2
    for k, flags in enumerate(combinations):
        x = SOURCE + k
        result = run(_put(x), *(bools[f] for f in flags))
        np.testing.assert_array_equal(np.asarray(result), expected(x, *flags), err_msg=str(flags))


def test_every_case_index_is_on_host(tmp_path):
    run = jax.jit(sampler_like, compiler_options={"enable_trace": "true", "export_path": str(tmp_path)})
    flags = [_put(np.bool_(f)) for f in (False, True, False, True)]
    np.testing.assert_array_equal(np.asarray(run(_put(SOURCE), *flags)), expected(SOURCE, False, True, False, True))

    (ir,) = [path.read_text() for path in (tmp_path / "irs").glob("ttnn_runtime_[0-9]*.mlir")]
    layouts = dict(re.findall(r"^(#ttnn_layout\d*) = (.*)$", ir, re.M))
    indices = re.findall(r'"ttcore\.case"\(.*?:\s*\(tensor<[^,]*, (#ttnn_layout\d*)>', ir)
    # The outer conditional, the two that replace the switch, and min-p's.
    assert len(indices) >= 4
    for layout in indices:
        assert "system_memory" in layouts[layout], layouts[layout]


def test_host_runs_ahead_of_the_device():
    # Device work that takes a while, then the nested conditionals. Reading an
    # index on device would hold the call until that work is done.
    def heavy(x, w):
        for _ in range(24):
            x = jnp.tanh(x @ w)
        return x[:64, :32]

    rng = np.random.default_rng(0)
    w = _put((rng.standard_normal((2048, 2048)) * 0.02).astype(np.float32))
    x = _put(rng.standard_normal((2048, 2048)).astype(np.float32))
    options = {"optimization_level": "O1", "enable_trace": "true"}
    heavy = jax.jit(heavy, compiler_options=options)
    run = jax.jit(sampler_like, compiler_options=options)
    flags = [_put(np.bool_(f)) for f in (False, True, True, True)]
    for _ in range(3):
        np.asarray(run(heavy(x, w), *flags))

    start = time.perf_counter()
    result = run(heavy(x, w), *flags)
    dispatched = time.perf_counter() - start
    result = np.asarray(result)
    finished = time.perf_counter() - start
    assert dispatched < 0.5 * finished, f"the call returned after {dispatched * 1e3:.1f} of {finished * 1e3:.1f} ms"
