"""Returned constants must not expose shared storage to in-place state updates."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize(
    "dtype,shape", [(jnp.float32, (2, 2, 128, 128)), (jnp.bfloat16, (2, 128, 3))]
)
@pytest.mark.parametrize("trace", [False, True])
def test_constant_outputs_are_independent(dtype, shape, trace):
    device = jax.devices("tt")[0]
    options = {"enable_trace": str(trace).lower()}

    def zeros():
        state = jnp.zeros(shape, dtype)
        return state, state

    allocate = jax.jit(zeros, compiler_options=options)

    def update(state, indices, values):
        return jax.ffi.ffi_call(
            "tt.state_pool_update",
            jax.ShapeDtypeStruct(state.shape, state.dtype),
            input_output_aliases={0: 0},
            vmap_method="sequential",
        )(state, indices, values)

    update = jax.jit(update, donate_argnums=(0,), compiler_options=options)
    indices = jax.device_put(np.array([1], np.int32), device)
    first, sibling = allocate()
    second, _ = allocate()
    # Exercise warmup, capture and replay without allowing writes to escape.
    for value in (1, 2, 3):
        values = jax.device_put(np.full((1, *shape[1:]), value, np.dtype(dtype)), device)
        first = update(first, indices, values)
        expected = np.zeros(shape, np.dtype(dtype))
        expected[1] = value
        np.testing.assert_array_equal(np.asarray(first), expected)
        np.testing.assert_array_equal(np.asarray(sibling), 0)
        np.testing.assert_array_equal(np.asarray(second), 0)
        for fresh in allocate():
            np.testing.assert_array_equal(np.asarray(fresh), 0)
