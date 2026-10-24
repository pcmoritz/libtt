"""Counted loops keep device counters only when the body or caller needs them."""

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize("trace", [False, True])
@pytest.mark.parametrize("dynamic_limit", [False, True])
@pytest.mark.parametrize("use", ["unused", "body", "result", "shared_increment"])
def test_counted_loop_counter_liveness(trace, dynamic_limit, use, tmp_path):
    def calculate(x, limit):
        limit = limit if dynamic_limit else jnp.int32(5)

        def body(state):
            counter, value, previous = state
            next_counter = counter + 1
            value = value * 2 + counter if use == "body" else value * 2
            previous = next_counter if use == "shared_increment" else previous
            return next_counter, value, previous

        counter, value, previous = jax.lax.while_loop(
            lambda state: state[0] < limit,
            body,
            (jnp.int32(0), x, jnp.int32(0)),
        )
        if use == "result":
            return value, counter
        if use == "shared_increment":
            return value, previous
        return value

    run = jax.jit(
        calculate,
        compiler_options={
            "optimization_level": "O1",
            "enable_trace": str(trace).lower(),
            "export_path": str(tmp_path),
        },
    )
    device = jax.devices("tt")[0]
    # Reuse the executable with different inputs, including an empty loop
    # and a single iteration when the limit is a runtime argument.
    for limit, x in [(5, 3), (0, 7), (1, -2), (5, 4)]:
        value = x
        for counter in range(limit if dynamic_limit else 5):
            value = value * 2 + counter if use == "body" else value * 2
        expected = np.int32(value)
        if use in ("result", "shared_increment"):
            expected = (expected, np.int32(limit if dynamic_limit else 5))
        got = run(
            jax.device_put(np.int32(x), device),
            jax.device_put(np.int32(limit), device),
        )
        jax.tree.map(np.testing.assert_array_equal, got, expected)

    if use == "unused":
        (ir,) = (tmp_path / "irs").glob("ttnn_runtime_*.mlir")
        text = ir.read_text()
        body = re.search(
            r"func\.func private @main_counted_loop\(.*?\n      \}", text, re.S
        )
        assert body is not None
        assert '"ttnn.add"' not in body.group(0)
