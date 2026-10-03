"""A value converted for several ops is converted once, unless an in-place
write intervenes."""

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
    pass converts every use on its own, the decomposition folds the copies."""

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
def test_conversion_after_in_place_write_reads_new_state(trace):
    """A row read before an in-place update and another row read after it
    each untilize the state (unaligned slices go through row-major); the
    second must not reuse the first, the update changed the state behind its
    value."""

    def step(state, indices, values):
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
        np.testing.assert_array_equal(np.asarray(before).astype(np.float32), expected[1, 3])
        expected[1] = value
        np.testing.assert_array_equal(np.asarray(after).astype(np.float32), expected[1, 4])
        np.testing.assert_array_equal(np.asarray(state).astype(np.float32), expected)


@pytest.mark.xfail(
    strict=True,
    reason="An in-place op has no result, so the lowering replaces the aliased "
    "result by the operand and the TTIR-level CSE merges a cast after the update "
    "into the cast before it (upstream convention, shared by update_cache).",
)
def test_cast_after_in_place_write_reads_new_state():
    def step(state, indices, values):
        before = state.astype(jnp.float32)
        state = update(state, indices, values)
        return before, state.astype(jnp.float32), state

    device = jax.devices("tt")[0]
    run = jax.jit(step, donate_argnums=(0,))
    rng = np.random.default_rng(3)
    state = rng.normal(0, 1, SHAPE).astype(jnp.bfloat16)
    indices = np.array([1], np.int32)
    expected = state.astype(np.float32)
    values = np.full((1, *SHAPE[1:]), 7, np.float32).astype(jnp.bfloat16)
    before, after, _ = run(*jax.device_put((state, indices, values), device))
    np.testing.assert_array_equal(np.asarray(before), expected)
    expected[1] = 7
    np.testing.assert_array_equal(np.asarray(after), expected)
