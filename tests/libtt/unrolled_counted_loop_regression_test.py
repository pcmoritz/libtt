"""Counted loops with constant bounds and short bodies run unrolled.

A loop runs on the host, one body program after another, outside the trace
around it. SGLang-TorchTPU's top-p and top-k filters search their cutoff bit by
bit in a 30-iteration fori_loop, which on a 64-wide block took 0.86 ms per call
traced and 0.59 ms unrolled. Loops of more than 64 iterations, or of more ops
than an unrolled budget, and loops with runtime bounds stay loops.
"""

import glob
import re

import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
import pytest


def runtime_ir(tmp_path):
    irs = glob.glob(str(tmp_path / "irs" / "ttnn_runtime_*.mlir"))
    assert len(irs) == 1, irs
    return open(irs[0]).read()


def run(fn, *args, tmp_path=None, **options):
    if tmp_path is not None:
        options["export_path"] = str(tmp_path)
    device = jax.devices("tt")[0]
    return np.asarray(jax.jit(fn, compiler_options=options or None)(*[jax.device_put(a, device) for a in args]))


@pytest.mark.parametrize("trips, unrolled", [(30, True), (64, True), (65, False)])
def test_short_counted_loops_unroll(trips, unrolled, tmp_path):
    def fn(x):
        return lax.fori_loop(0, trips, lambda i, a: a * 0.5 + i.astype(jnp.float32), x)

    x = np.arange(64, dtype=np.float32).reshape(1, 64)
    got = run(fn, x, tmp_path=tmp_path)
    np.testing.assert_allclose(got, np.asarray(jax.jit(fn, backend="cpu")(x)), rtol=1e-6)
    ir = runtime_ir(tmp_path)
    assert ("while_loop" not in ir) == unrolled
    assert len(re.findall(r'"ttnn\.multiply"', ir)) == (trips if unrolled else 1)


def test_runtime_bound_stays_a_loop(tmp_path):
    def fn(x, n):
        return lax.fori_loop(0, n, lambda i, a: a + 1.0, x)

    x = np.zeros((1, 32), np.float32)
    np.testing.assert_array_equal(run(fn, x, np.int32(7), tmp_path=tmp_path), x + 7)
    assert "while_loop" in runtime_ir(tmp_path)


def test_unrolled_loop_carries_and_strides():
    # A start and step other than 0 and 1, several carried values, one passed
    # through unchanged, and a nested short loop.
    def fn(x, y):
        def inner(j, a):
            return a + j.astype(jnp.float32)

        def body(i, carry):
            a, b, c = carry
            a = lax.fori_loop(0, 3, inner, a) * 0.5
            return a, b + a, c

        return lax.fori_loop(2, 20, body, (x, y, x * 2.0))

    def strided(x):
        return lax.while_loop(lambda s: s[0] < 17, lambda s: (s[0] + 5, s[1] * 2.0 + s[0].astype(jnp.float32)),
                              (jnp.int32(1), x))[1]

    x = np.linspace(-1, 1, 64, dtype=np.float32).reshape(1, 64)
    y = np.ones((1, 64), np.float32)
    for got, want in zip(run(fn, x, y), jax.jit(fn, backend="cpu")(x, y)):
        np.testing.assert_allclose(got, np.asarray(want), rtol=1e-5)
    np.testing.assert_allclose(run(strided, x), np.asarray(jax.jit(strided, backend="cpu")(x)), rtol=1e-6)


def test_threshold_search_traced():
    # sglang_tpu.sampling_mask's top-p cutoff search: the largest float32 whose
    # kept mass reaches top_p, found bit by bit, traced and replayed.
    def search(probs, top_p):
        def step(i, bits):
            candidate = bits | jnp.left_shift(1, 29 - i)
            threshold = lax.bitcast_convert_type(candidate, jnp.float32)
            mass = jnp.sum(jnp.where(probs >= threshold[:, None], probs, 0.0), axis=-1)
            return jnp.where(mass >= top_p, candidate, bits)

        bits = lax.fori_loop(0, 30, step, jnp.zeros(probs.shape[0], jnp.int32))
        return lax.bitcast_convert_type(bits, jnp.float32)

    rng = np.random.default_rng(0)
    device = jax.devices("tt")[0]
    f = jax.jit(search, compiler_options={"enable_trace": "true"})
    for i in range(3):
        probs = np.asarray(jax.nn.softmax(rng.normal(0, 2, (2, 64)).astype(np.float32), axis=-1))
        top_p = np.array([0.5, 0.9], np.float32)
        got = np.asarray(f(jax.device_put(probs, device), jax.device_put(top_p, device)))
        np.testing.assert_array_equal(got, np.asarray(jax.jit(search, backend="cpu")(probs, top_p)), err_msg=f"run {i}")
