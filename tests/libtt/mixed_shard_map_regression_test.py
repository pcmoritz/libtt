"""shard_map regions inside automatically sharded programs.

The automatic parts are partitioned around the shard_map region, whose body
already computes on device-local values.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, NamedSharding, PartitionSpec as P


@pytest.mark.parametrize("axis_type", [AxisType.Auto, AxisType.Explicit])
def test_shard_map_between_sharded_ops(axis_type):
    # Like SGLang-JAX's logits processor: a sharded projection, a per-shard row
    # selection in shard_map, and more automatic ops on the result.
    devices = jax.devices("tt")
    if len(devices) < 2:
        pytest.skip("needs at least two devices")
    n = len(devices)
    mesh = jax.make_mesh((1, n), ("data", "tensor"), (axis_type, axis_type), devices=devices)
    rng = np.random.default_rng(0)
    x = (rng.integers(-4, 5, (8, 64)) / 4).astype(np.float32)
    w = (rng.integers(-4, 5, (64, 32 * n)) / 4).astype(np.float32)
    rows = np.array([3, 1], np.int32)

    def select(hidden, rows):
        return hidden[rows]

    def body(x, w, rows):
        hidden = x @ w
        if axis_type == AxisType.Explicit:
            hidden = jax.sharding.reshard(hidden, P("data", None))
        selected = jax.shard_map(
            select, mesh=mesh, in_specs=(P("data", None), P()), out_specs=P("data", None)
        )(hidden, rows)
        return jnp.tanh(selected) * 2 + 1

    with jax.set_mesh(mesh):
        out = jax.jit(body)(
            jax.device_put(x, NamedSharding(mesh, P())),
            jax.device_put(w, NamedSharding(mesh, P(None, "tensor"))),
            jax.device_put(rows, NamedSharding(mesh, P())),
        )
    np.testing.assert_allclose(np.asarray(out), np.tanh((x @ w)[rows]) * 2 + 1, rtol=1e-5, atol=1e-6)


def test_shard_map_with_collective_in_sharded_program():
    # A psum inside the region, sharded inputs flowing in, and automatic ops on
    # the region's sharded output.
    devices = jax.devices("tt")
    if len(devices) < 2:
        pytest.skip("needs at least two devices")
    n = len(devices)
    mesh = jax.make_mesh((n,), ("x",), (AxisType.Auto,), devices=devices)
    x = np.arange(n * 8 * 32, dtype=np.float32).reshape(n * 8, 32) / 64

    def body(x):
        doubled = x * 2
        total = jax.shard_map(
            lambda b: b - jax.lax.psum(b, "x") / n, mesh=mesh, in_specs=P("x"), out_specs=P("x")
        )(doubled)
        return total.sum(axis=1)

    with jax.set_mesh(mesh):
        out = jax.jit(body)(jax.device_put(x, NamedSharding(mesh, P("x"))))
    doubled = (x * 2).reshape(n, 8, 32)
    expected = (doubled - doubled.sum(axis=0) / n).reshape(n * 8, 32).sum(axis=1)
    np.testing.assert_allclose(np.asarray(out), expected, rtol=1e-5, atol=1e-4)


def test_automatic_op_after_fully_manual_shard_map():
    # The arguments feed only the shard_map, but its result is still global to
    # the automatic tanh.
    devices = jax.devices("tt")
    if len(devices) < 2:
        pytest.skip("needs at least two devices")
    n = len(devices)
    mesh = jax.make_mesh((n,), ("x",), (AxisType.Auto,), devices=devices)
    x = np.arange(n * 8, dtype=np.float32) / (n * 8)

    def body(x):
        doubled = jax.shard_map(lambda b: b * 2, mesh=mesh, in_specs=P("x"), out_specs=P("x"))(x)
        return jnp.tanh(doubled)

    with jax.set_mesh(mesh):
        out = jax.jit(body)(jax.device_put(x, NamedSharding(mesh, P("x"))))
    np.testing.assert_allclose(np.asarray(out), np.tanh(x * 2), rtol=1e-5, atol=1e-6)


def test_shard_map_manual_over_one_of_two_axes():
    devices = jax.devices("tt")
    if len(devices) < 4:
        pytest.skip("needs four devices")
    mesh = jax.make_mesh((2, 2), ("x", "y"), (AxisType.Auto, AxisType.Auto), devices=devices[:4])
    x = np.arange(8 * 64, dtype=np.float32).reshape(8, 64) / 512

    def body(x):
        centered = jax.shard_map(
            lambda b: b - jax.lax.psum(b, "x"), mesh=mesh, in_specs=P("x"), out_specs=P("x"), axis_names={"x"}
        )(x)
        return jnp.tanh(centered)

    with jax.set_mesh(mesh):
        out = jax.jit(body)(jax.device_put(x, NamedSharding(mesh, P("x", "y"))))
    halves = x.reshape(2, 4, 64)
    np.testing.assert_allclose(
        np.asarray(out), np.tanh((halves - halves.sum(axis=0)).reshape(8, 64)), rtol=1e-5, atol=1e-6
    )


@pytest.mark.parametrize("updates", [1, 2])
def test_shard_map_manual_over_one_axis_updates_a_cache(updates):
    # The cache's heads are split over x by the shard_map and over y inside
    # it. Its region argument is local over x only until the region is
    # inlined, while the updates' results are local over x and y.
    devices = jax.devices("tt")
    if len(devices) < 4:
        pytest.skip("needs four devices")
    mesh = jax.make_mesh((2, 2), ("x", "y"), (AxisType.Auto, AxisType.Auto), devices=devices[:4])
    dtype, heads, users = jnp.bfloat16, 8, 2
    rng = np.random.default_rng(2)
    expected = np.asarray(rng.normal(size=(17, heads, 32, 128)), dtype=dtype)
    values = np.asarray(rng.normal(size=(1, users, heads, 128)), dtype=dtype)
    positions = np.array([3, 40], np.int32)
    pages = np.tile(np.arange(16, dtype=np.int32), (users, 1))

    def update(cache, values, positions, pages):
        return jax.ffi.ffi_call(
            "tt.paged_update_cache",
            jax.ShapeDtypeStruct(cache.shape, cache.dtype),
            input_output_aliases={0: 0},
        )(cache, values, positions, pages, share_cache=True)

    def region(cache, values, positions, pages):
        cache = jax.lax.with_sharding_constraint(cache, P(None, "y"))
        values = jax.lax.with_sharding_constraint(values, P(None, None, "y"))
        for step in range(updates):
            cache = update(cache, values * (step + 1), positions + step, pages)
        return jax.lax.with_sharding_constraint(cache, P(None, "y"))

    def body(cache, values, positions, pages):
        return jax.shard_map(
            region,
            mesh=mesh,
            in_specs=(P(None, "x"), P(None, None, "x"), P(), P()),
            out_specs=P(None, "x"),
            axis_names={"x"},
        )(cache, values, positions, pages)

    heads_sharded = NamedSharding(mesh, P(None, ("x", "y")))
    with jax.set_mesh(mesh):
        out = jax.jit(body, donate_argnums=0, out_shardings=heads_sharded)(
            jax.device_put(expected.copy(), heads_sharded),
            jax.device_put(values, NamedSharding(mesh, P(None, None, ("x", "y")))),
            jax.device_put(positions, NamedSharding(mesh, P())),
            jax.device_put(pages, NamedSharding(mesh, P())),
        )
    for step in range(updates):
        for row, position in enumerate(positions + step):
            expected[pages[row, position // 32], :, position % 32] = values[0, row] * (step + 1)
    np.testing.assert_array_equal(np.asarray(out), expected)
