"""Values crossing a trace boundary are copied once, and stay correct.

An eager conditional splits a traced program into a trace before it and a
trace after it. The runtime published every trace output as a copy of its slot
and copied every staged trace input into a slot of its own, so a value crossing
a boundary cost two passes over it; the sampler's vocabulary-sized logits cross
several per decode step (2-4% of decode). Traces now publish their output slots
as they are, so only the copy into the next trace's input slot remains. A value
that outlives the invocation is copied where it escapes: a program output, and
a loop carry (tt_mlir_trace_publish_slots.patch, tt_mlir_control_flow_runtime.cpp).
"""

import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

SOURCE = (np.arange(64 * 32, dtype=np.float32).reshape(64, 32) % 7) - 3


def _jit(fn):
    return jax.jit(fn, compiler_options={"optimization_level": "O1", "enable_trace": "true"})


def _put(value):
    return jax.device_put(np.asarray(value), jax.devices("tt")[0])


def test_values_cross_conditionals_between_traces():
    # A trace before the conditional, whose result is also a program output,
    # an eager conditional whose false branch passes its operand through, and
    # a trace after it.
    def step(x, pred):
        y = x * 2 + 1
        z = lax.cond(pred, lambda v: -v, lambda v: v, y)
        return y, z + 3

    step = _jit(step)
    bools = {value: _put(np.bool_(value)) for value in (False, True)}
    calls = []
    for k, pred in enumerate([True, False, False, True, True, False]):
        x = SOURCE + k
        calls.append((x, pred, step(_put(x), bools[pred])))
    # Checked only after the last call: every result must have outlived the
    # later replays that rewrote the slots it was published from.
    for x, pred, (y, z) in calls:
        expected = x * 2 + 1
        np.testing.assert_array_equal(np.asarray(y), expected)
        np.testing.assert_array_equal(np.asarray(z), (-expected if pred else expected) + 3)


def test_loop_body_with_a_conditional_carries_its_values():
    # A conditional makes the loop run eagerly, with traces inside its body
    # program. The body forwards a value it computed the previous iteration,
    # so the carry must not refer to a trace slot the next replay rewrites.
    def run(count, x):
        def body(i, state):
            a, b = state
            a = lax.cond(i % 2 == 0, lambda v: v * 2.0, lambda v: v + 1.0, a)
            return a + b, a

        return lax.fori_loop(0, count, body, (x, x * 2))

    run = _jit(run)
    x = _put(SOURCE)
    for count in (3, 1, 4, 0, 2):
        a, b = SOURCE, SOURCE * 2
        for i in range(count):
            a = a * 2.0 if i % 2 == 0 else a + 1.0
            a, b = a + b, a
        result_a, result_b = run(_put(np.int32(count)), x)
        np.testing.assert_array_equal(np.asarray(result_a), a)
        np.testing.assert_array_equal(np.asarray(result_b), b)


def test_conditionals_cost_one_pass_over_the_logits():
    # The sampler's shape: traced model work, then conditionals over the
    # vocabulary-sized logits (the first passes them through unchanged when
    # off), then traced work after them. Per step, two such conditionals cost
    # about 300 us when each boundary copied the logits twice, and about
    # 200 us with the one copy into the next trace's input slot that remains.
    vocab = 151936
    rng = np.random.default_rng(0)
    weight = (rng.standard_normal((1024, 1024)) * 0.03).astype(np.float32)
    scale = rng.standard_normal((1, vocab)).astype(np.float32)

    def model(x, w):
        for _ in range(4):
            x = jnp.tanh(x @ w)
        return x

    def with_conds(x, w, scale, pred, penalize):
        h = model(x, w)
        logits = lax.cond(penalize, lambda l: l - 1.0, lambda l: l, h[:, :1] * scale)
        token = lax.cond(
            pred, lambda l: jnp.argmax(l, axis=-1), lambda l: jnp.argmin(l, axis=-1), logits
        )
        return h, token

    def without(x, w, scale, pred, penalize):
        h = model(x, w)
        return h, jnp.argmax(h[:, :1] * scale, axis=-1)

    x = _put(rng.standard_normal((1, 1024)).astype(np.float32))
    w, scale = _put(weight), _put(scale)
    true, false = _put(np.bool_(True)), _put(np.bool_(False))

    def per_step(fn, steps=50):
        fn = _jit(fn)
        h = x
        for _ in range(5):
            h, token = fn(h, w, scale, true, false)
        np.asarray(token)
        h = x
        start = time.perf_counter()
        for _ in range(steps):
            h, token = fn(h, w, scale, true, false)
        np.asarray(token)
        return (time.perf_counter() - start) / steps

    overhead = per_step(with_conds) - per_step(without)
    assert (
        overhead < 255e-6
    ), f"two conditionals over the logits cost {overhead * 1e6:.0f} us per step"
