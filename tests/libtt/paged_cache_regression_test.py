"""Shared paged-cache writes must preserve every row, including trace replay."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, NamedSharding, PartitionSpec as P


@pytest.mark.parametrize("repeat", [False, True])
@pytest.mark.parametrize("trace", [False, True])
@pytest.mark.parametrize("tokens", [16, 32])
def test_shared_page_updates(trace, tokens, repeat):
    device = jax.devices("tt")[0]
    dtype = jnp.bfloat16
    rng = np.random.default_rng(0)
    expected = np.asarray(rng.normal(size=(17, 8, 32, 128)), dtype=dtype)
    cache = jax.device_put(expected.copy(), device)
    pages = np.tile(np.arange(16, dtype=np.int32)[::-1], (tokens, 1))
    positions = np.arange(tokens, dtype=np.int32) + 24
    if repeat:
        # A later row written to the same position replaces the earlier one.
        positions[1::2] = positions[::2]
    # Exercise skipped writes before, between and after valid updates.
    positions[[0, tokens // 2, -1]] = -1

    def update(cache, values, positions, pages):
        return jax.ffi.ffi_call(
            "tt.paged_update_cache",
            jax.ShapeDtypeStruct(cache.shape, cache.dtype),
            input_output_aliases={0: 0},
        )(cache, values, positions, pages, share_cache=True)

    update = jax.jit(
        update,
        donate_argnums=(0,),
        compiler_options={"optimization_level": "O1", "enable_trace": str(trace).lower()},
    )
    for offset in [0, 5, 1]:
        indices = np.where(positions < 0, -1, positions + offset).astype(np.int32)
        values = np.asarray(rng.normal(size=(1, tokens, 8, 128)), dtype=dtype)
        cache = update(cache, *[jax.device_put(x, device) for x in (values, indices, pages)])
        for row, index in enumerate(indices):
            if index >= 0:
                expected[pages[row, index // 32], :, index % 32, :] = values[0, row]
        np.testing.assert_array_equal(np.asarray(cache), expected)


def test_chained_updates_of_a_sharded_cache():
    """Two updates of one cache in one program, its heads split over the chips.
    Sharding propagation leaves the first update's result, which only feeds the
    second, without a sharding; it must still get the cache's local shape."""
    devices = jax.devices("tt")
    mesh = jax.make_mesh(
        (len(devices),), ("tensor",), axis_types=(AxisType.Explicit,), devices=devices
    )
    dtype, heads, users = jnp.bfloat16, 2 * len(devices), 4
    rng = np.random.default_rng(1)
    expected = np.asarray(rng.normal(size=(17, heads, 32, 128)), dtype=dtype)
    values = np.asarray(rng.normal(size=(1, users, heads, 128)), dtype=dtype)
    positions = np.array([3, 40, 77, 300], np.int32)
    pages = np.tile(np.arange(16, dtype=np.int32), (users, 1))

    def update(cache, values, positions, pages):
        return jax.ffi.ffi_call(
            "tt.paged_update_cache",
            jax.ShapeDtypeStruct(cache.shape, cache.dtype, sharding=jax.typeof(cache).sharding),
            input_output_aliases={0: 0},
        )(cache, values, positions, pages, share_cache=True)

    def update_twice(cache, values, positions, pages):
        cache = update(cache, values, positions, pages)
        return update(cache, values * 2, positions + 1, pages)

    with jax.set_mesh(mesh):
        heads_sharded = NamedSharding(mesh, P(None, "tensor"))
        cache = jax.jit(update_twice, donate_argnums=0)(
            jax.device_put(expected.copy(), heads_sharded),
            jax.device_put(values, NamedSharding(mesh, P(None, None, "tensor"))),
            jax.device_put(positions, NamedSharding(mesh, P())),
            jax.device_put(pages, NamedSharding(mesh, P())),
        )
        for step in range(2):
            for row, position in enumerate(positions + step):
                expected[pages[row, position // 32], :, position % 32] = values[0, row] * (step + 1)
        np.testing.assert_array_equal(np.asarray(cache), expected)
