"""Values crossing a trace boundary are not copied, and stay correct.

An eager conditional splits a traced program into a trace before it and a
trace after it. The runtime published every trace output as a copy of its slot
and copied every staged trace input into a slot of its own, so a value crossing
a boundary cost two passes over it; the sampler's vocabulary-sized logits cross
several per decode step (2-4% of decode). Traces now publish their output slots
as they are, and a trace fed from another trace's slot reads it in place. A
value that outlives the invocation is copied where it escapes: a program output,
and a loop carry (tt_mlir_trace_slot_aliasing.patch, tt_mlir_control_flow_runtime.cpp).
"""

import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

VOCAB = 151936
rng = np.random.default_rng(0)
WEIGHT = (rng.standard_normal((1024, 1024)) * 0.03).astype(np.float32)
SCALE = rng.standard_normal((1, VOCAB)).astype(np.float32)


def _jit(fn):
    return jax.jit(fn, compiler_options={"optimization_level": "O1", "enable_trace": "true"})


def _put(value):
    return jax.device_put(np.asarray(value), jax.devices("tt")[0])


def _model(x, w):
    h = x
    for _ in range(4):
        h = jnp.tanh(h @ w)
    return h


def _logits(h, scale):
    return h[:, :1] * scale


def _reference(x, pred, penalize):
    h = x
    for _ in range(4):
        h = np.tanh(h @ WEIGHT)
    logits = h[:, :1] * SCALE
    if penalize:
        logits = logits - 1.0
    token = np.argmax(logits, axis=-1) if pred else np.argmin(logits, axis=-1)
    return h, token


def _step(x, w, scale, pred, penalize):
    # The sampler's shape: traced model work, then conditionals over the
    # vocabulary-sized logits (the first passes them through unchanged when
    # off), then traced work after them.
    h = _model(x, w)
    logits = lax.cond(penalize, lambda l: l - 1.0, lambda l: l, _logits(h, scale))
    token = lax.cond(
        pred, lambda l: jnp.argmax(l, axis=-1), lambda l: jnp.argmin(l, axis=-1), logits
    )
    return h, token


def test_conditionals_between_traces_take_either_branch():
    step = _jit(_step)
    x, w, scale = _put(rng.standard_normal((1, 1024)).astype(np.float32)), _put(WEIGHT), _put(SCALE)
    bools = {value: _put(np.bool_(value)) for value in (False, True)}
    # Alternating predicates exercise both branch programs and both of the
    # following trace's variants: one fed from the pass-through branch's
    # aliased slot, one from the penalty branch's fresh output.
    for pred, penalize in [
        (True, False),
        (False, False),
        (True, True),
        (False, True),
        (True, False),
        (False, True),
    ]:
        h, token = step(x, w, scale, bools[pred], bools[penalize])
        h_ref, token_ref = _reference(np.asarray(x), pred, penalize)
        np.testing.assert_allclose(np.asarray(h), h_ref, rtol=1e-2, atol=5e-3)
        assert int(np.asarray(token)[0]) == int(token_ref[0])


def test_outputs_outlive_the_next_call():
    # A program output that is a trace's output slot is copied on the way out,
    # so the next invocation's replay (which rewrites the slot) leaves it alone.
    step = _jit(_step)
    w, scale, true = _put(WEIGHT), _put(SCALE), _put(np.bool_(True))
    false = _put(np.bool_(False))
    xs = [rng.standard_normal((1, 1024)).astype(np.float32) for _ in range(3)]
    results = [step(_put(x), w, scale, true, false) for x in xs]
    for x, (h, token) in zip(xs, results):
        h_ref, token_ref = _reference(x, True, False)
        np.testing.assert_allclose(np.asarray(h), h_ref, rtol=1e-2, atol=5e-3)
        assert int(np.asarray(token)[0]) == int(token_ref[0])


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
    source = (np.arange(64 * 32, dtype=np.float32).reshape(64, 32) % 7) - 3
    x = _put(source)
    for count in (3, 1, 4, 0, 2):
        a, b = source, source * 2
        for i in range(count):
            a = a * 2.0 if i % 2 == 0 else a + 1.0
            a, b = a + b, a
        result_a, result_b = run(_put(np.int32(count)), x)
        np.testing.assert_array_equal(np.asarray(result_a), a)
        np.testing.assert_array_equal(np.asarray(result_b), b)


def test_conditionals_cost_no_pass_over_the_logits():
    # Per step, a conditional over the logits cost two copies of [1, 151936]
    # (about 300 us) before; now it costs a trace launch (tens of us).
    def with_conds(x, w, scale, pred, penalize):
        return _step(x, w, scale, pred, penalize)

    def without(x, w, scale, pred, penalize):
        h = _model(x, w)
        return h, jnp.argmax(_logits(h, scale), axis=-1)

    x, w, scale = _put(rng.standard_normal((1, 1024)).astype(np.float32)), _put(WEIGHT), _put(SCALE)
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
        overhead < 150e-6
    ), f"two conditionals over the logits cost {overhead * 1e6:.0f} us per step"
