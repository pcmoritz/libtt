"""Reads of a cache before and after an in-place update stay apart under CSE,
and a value converted for several ops is converted once."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

SHAPE = (3, 128, 3)


def update(state, indices, values):
    return jax.ffi.ffi_call(
        "tt.state_pool_update",
        jax.ShapeDtypeStruct(state.shape, state.dtype),
        input_output_aliases={0: 0},
        vmap_method="sequential",
    )(state, indices, values)


def test_shared_indices_converted_once(tmp_path):
    """Two state updates with the same indices tilize them once: the layout
    pass converts every use on its own, and CSE after the layout
    decomposition keeps one conversion."""

    def step(first, second, indices, values):
        return update(first, indices, values), update(second, indices, values)

    run = jax.jit(step, donate_argnums=(0, 1), compiler_options={"export_path": str(tmp_path)})
    spec = lambda shape, dtype: jax.ShapeDtypeStruct(shape, dtype)
    run.lower(
        spec(SHAPE, jnp.bfloat16), spec(SHAPE, jnp.bfloat16), spec((1,), jnp.int32), spec((1, *SHAPE[1:]), jnp.bfloat16)
    ).compile()
    (ir,) = [path.read_text() for path in (tmp_path / "irs").glob("ttnn_[0-9]*.mlir")]
    # The indices are the third argument; both updates read the one tilized copy.
    assert ir.count('"ttnn.to_layout"(%arg2)') == 1
    assert ir.count("ttnn.state_pool_update") == 2


@pytest.mark.parametrize("trace", [False, True])
@pytest.mark.parametrize("read", ["cast", "row"])
def test_read_after_in_place_update_sees_new_state(trace, read):
    """The updated state is a new value: a cast of it is not the cast of the
    state before the update (TTIR-level CSE), and a row read of it does not
    reuse the layout conversion made for a row read before (TTNN-level CSE)."""

    def step(state, indices, values):
        if read == "cast":
            before = state.astype(jnp.float32)
            state = update(state, indices, values)
            return before, state.astype(jnp.float32), state
        before = state[1, 3]
        state = update(state, indices, values)
        return before, state[1, 4], state

    device = jax.devices("tt")[0]
    run = jax.jit(step, donate_argnums=(0,), compiler_options={"enable_trace": str(trace).lower()})
    rng = np.random.default_rng(3)
    state = rng.normal(0, 1, SHAPE).astype(jnp.bfloat16)
    indices = np.array([1], np.int32)
    expected = state.astype(np.float32)
    for value in (1, 2):
        values = np.full((1, *SHAPE[1:]), value, np.float32).astype(jnp.bfloat16)
        before, after, state = run(*jax.device_put((state, indices, values), device))
        expected_before = expected if read == "cast" else expected[1, 3]
        np.testing.assert_array_equal(np.asarray(before).astype(np.float32), expected_before)
        expected[1] = value
        expected_after = expected if read == "cast" else expected[1, 4]
        np.testing.assert_array_equal(np.asarray(after).astype(np.float32), expected_after)
        np.testing.assert_array_equal(np.asarray(state).astype(np.float32), expected)


@pytest.mark.parametrize("trace", [False, True])
def test_two_updates_of_one_state(trace):
    """A second update writes the first one's result; both rows land, and a
    read between them sees only the first."""

    def step(state, indices, values):
        state = update(state, indices, values)
        middle = state.astype(jnp.float32)
        state = update(state, indices + 1, values + 1)
        return middle, state

    device = jax.devices("tt")[0]
    run = jax.jit(step, donate_argnums=(0,), compiler_options={"enable_trace": str(trace).lower()})
    # Slot 0 is the padding slot, which updates never write.
    indices = np.array([1], np.int32)
    values = np.full((1, *SHAPE[1:]), 3, np.float32).astype(jnp.bfloat16)
    for _ in range(2):
        state = np.zeros(SHAPE, np.float32).astype(jnp.bfloat16)
        middle, state = run(*jax.device_put((state, indices, values), device))
        expected = np.zeros(SHAPE, np.float32)
        expected[1] = 3
        np.testing.assert_array_equal(np.asarray(middle), expected)
        expected[2] = 4
        np.testing.assert_array_equal(np.asarray(state).astype(np.float32), expected)
