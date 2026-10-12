"""Gathers, scatters and dynamic updates lower to the indexed-rows kernels.

Every StableHLO gather, scatter and dynamic_update_slice compiles to
ttir.rows_gather or ttir.rows_scatter: rows of a [B, N, R] table moved by int32
index components. These tests cover the shapes the lowering normalizes: token
and sampler lookups, windows, batching dims, multi-component indices, element
types moved as bytes, every reduction, and out-of-bounds handling.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

OPTIONS = {"optimization_level": "O1"}


def run(fn, *args, trace=False):
    device = jax.devices("tt")[0]
    options = dict(OPTIONS, enable_trace=str(trace).lower())
    compiled = jax.jit(fn, compiler_options=options)
    return np.asarray(compiled(*[jax.device_put(a, device) for a in args]))


def assert_equal(got, want):
    np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


# Trace capture copies its inputs with ttnn.copy, which has no float16 path, so
# float16 is tested without trace.
@pytest.mark.parametrize(
    "dtype,trace",
    [(d, t) for d in (jnp.bfloat16, np.float32, np.int32, np.uint32, np.uint8) for t in (False, True)]
    + [(np.float16, False)],
)
def test_embedding_lookup(dtype, trace):
    # A token embedding: rows of a table by int32 ids, with ids past the end
    # clamped (JAX normalizes negative ids itself).
    rng = np.random.default_rng(1)
    if np.issubdtype(np.dtype(dtype), np.integer):
        table = rng.integers(0, 200, size=(50, 96)).astype(dtype)
    else:
        table = rng.normal(size=(50, 96)).astype(dtype)
    ids = np.array([[3, 49, 0, 7], [-1, 12, 60, 49]], np.int32)
    got = run(lambda t, i: jnp.take(t, i, axis=0, mode="clip"), table, ids, trace=trace)
    assert_equal(got, np.take(np.asarray(table), ids, axis=0, mode="clip"))


def test_cross_entropy_gather():
    # Sampler and loss lookups: one element per row, two index components.
    rng = np.random.default_rng(2)
    logprobs = rng.normal(size=(64, 300)).astype(np.float32)
    labels = rng.integers(0, 300, size=64).astype(np.int32)
    got = run(lambda x, l: x[jnp.arange(64)[:, None], l[:, None]][:, 0], logprobs, labels)
    assert_equal(got, logprobs[np.arange(64), labels])
    got = run(lambda x, l: x.at[jnp.arange(64), l].get(mode="clip"), logprobs, labels)
    assert_equal(got, logprobs[np.arange(64), labels])


def test_take_along_axis():
    # A batched gather along the last axis: router weights by expert ids.
    rng = np.random.default_rng(3)
    logits = rng.normal(size=(40, 128)).astype(np.float32)
    ids = rng.integers(0, 128, size=(40, 8)).astype(np.int32)
    got = run(lambda x, i: jnp.take_along_axis(x, i, axis=1), logits, ids)
    assert_equal(got, np.take_along_axis(logits, ids, axis=1))


def test_gather_middle_axis_keeps_operand_order():
    # Indexing a middle axis: the leading axis is a table batch and the
    # trailing one the row, so the operand is not permuted.
    rng = np.random.default_rng(4)
    x = rng.normal(size=(3, 70, 33)).astype(np.float32)
    idx = np.array([69, 0, 5, 5, 71, -2], np.int32)
    got = run(lambda x, i: jnp.take(x, i, axis=1, mode="clip"), x, idx)
    assert_equal(got, np.take(x, idx, axis=1, mode="clip"))


def test_gather_window_and_static_slice():
    # x[idx, 2:5] indexes one axis and slices another: a two-component gather
    # whose second component is a constant with a window of 3.
    rng = np.random.default_rng(5)
    x = rng.normal(size=(20, 9)).astype(np.float32)
    idx = np.array([1, 19, 4, 4, 0], np.int32)
    got = run(lambda x, i: x[i, 2:5], x, idx)
    assert_equal(got, x[idx, 2:5])


def test_gather_window_along_indexed_axis():
    # Dynamic windows: lax.dynamic_slice in a vmap becomes a gather whose
    # indexed dimension keeps a window.
    rng = np.random.default_rng(6)
    x = rng.normal(size=(64, 8)).astype(np.float32)
    starts = np.array([0, 60, 17, 3], np.int32)

    def windows(x, starts):
        return jax.vmap(lambda s: jax.lax.dynamic_slice_in_dim(x, s, 4, axis=0))(starts)

    got = run(windows, x, starts)
    want = np.stack([x[min(s, 60) : min(s, 60) + 4] for s in starts])
    assert_equal(got, want)


def test_gather_with_batching_dims():
    # jnp.take_along_axis over two leading batch dims pairs operand and index
    # batching dims.
    rng = np.random.default_rng(7)
    x = rng.normal(size=(2, 3, 50, 4)).astype(np.float32)
    idx = rng.integers(0, 50, size=(2, 3, 7, 4)).astype(np.int32)
    got = run(lambda x, i: jnp.take_along_axis(x, i, axis=2), x, idx)
    assert_equal(got, np.take_along_axis(x, idx, axis=2))


def test_gather_bool_and_wide_rows():
    rng = np.random.default_rng(8)
    mask = rng.integers(0, 2, size=(16, 40)).astype(bool)
    idx = np.array([15, 1, 1, 0], np.int32)
    got = run(lambda m, i: m[i], mask, idx)
    assert_equal(got, mask[idx])
    # Rows wider than one 64 KiB transfer chunk.
    wide = rng.normal(size=(5, 40000)).astype(np.float32)
    got = run(lambda t, i: jnp.take(t, i, axis=0, mode="clip"), wide, idx)
    assert_equal(got, np.take(wide, idx, axis=0, mode="clip"))


@pytest.mark.parametrize("dtype", [jnp.bfloat16, np.float32, np.float16, np.int32, np.uint32, np.uint8])
def test_scatter_set_drops_out_of_bounds(dtype):
    rng = np.random.default_rng(9)
    x = rng.integers(0, 100, size=(30, 12)).astype(dtype)
    v = rng.integers(0, 100, size=(5, 12)).astype(dtype)
    idx = np.array([2, 29, 31, 2, 7], np.int32)  # 31 is dropped; 2 twice
    got = run(lambda x, i, v: x.at[i].set(v), x, idx, v)
    want = np.array(x)
    for i, row in zip(idx, np.asarray(v)):
        if i < 30:
            want[i] = row
    assert_equal(got, want)


@pytest.mark.parametrize(
    "reduce,combine",
    [("add", np.add), ("multiply", np.multiply), ("max", np.maximum), ("min", np.minimum)],
)
@pytest.mark.parametrize("dtype", [np.float32, jnp.bfloat16, np.float16, np.int32])
def test_scatter_reductions(reduce, combine, dtype):
    rng = np.random.default_rng(10)
    x = rng.integers(1, 5, size=(16, 8)).astype(dtype)
    v = rng.integers(1, 5, size=(10, 8)).astype(dtype)
    idx = np.array([0, 3, 3, 15, 0, 7, 3, 1, 1, 1], np.int32)
    got = run(lambda x, i, v: getattr(x.at[i], reduce)(v), x, idx, v)
    want = np.array(x)
    for i, row in zip(idx, np.asarray(v)):
        want[i] = combine(want[i], row).astype(dtype)
    assert_equal(got, want)


def test_scatter_max_propagates_nan():
    x = np.zeros((4, 4), np.float32)
    v = np.full((2, 4), 1.0, np.float32)
    v[0, 1] = np.nan
    idx = np.array([2, 2], np.int32)
    got = run(lambda x, i, v: x.at[i].max(v), x, idx, v)
    want = np.zeros((4, 4), np.float32)
    want[2] = 1.0
    want[2, 1] = np.nan
    np.testing.assert_array_equal(got, want)


def test_scatter_two_components_sampler_pattern():
    # The sampler's top-p: zeros.at[row, col].set(values) with a column index
    # per element.
    rng = np.random.default_rng(11)
    probs = rng.random((8, 50)).astype(np.float32)
    cols = np.argsort(-probs, axis=1).astype(np.int32)
    rows = np.arange(8, dtype=np.int32)[:, None]
    got = run(lambda p, c: jnp.zeros_like(p).at[rows, c].set(jnp.sort(p, axis=1)[:, ::-1]), probs, cols)
    want = np.zeros_like(probs)
    want[rows, cols] = -np.sort(-probs, axis=1)
    assert_equal(got, want)


def test_scatter_subtract_and_apply():
    rng = np.random.default_rng(12)
    x = rng.normal(size=(12, 6)).astype(np.float32)
    v = rng.normal(size=(3, 6)).astype(np.float32)
    idx = np.array([1, 5, 11], np.int32)
    got = run(lambda x, i, v: x.at[i].subtract(v), x, idx, v)
    want = np.array(x)
    want[idx] -= v
    assert_equal(got, want)
    got = run(lambda x, i: x.at[i].apply(jnp.negative), x, idx)
    want = np.array(x)
    want[idx] = -want[idx]
    assert_equal(got, want)


def test_scatter_middle_axis():
    # Writing columns: the indexed axis is in the middle, with a batch before
    # and a row after it.
    rng = np.random.default_rng(13)
    x = rng.normal(size=(3, 40, 5)).astype(np.float32)
    v = rng.normal(size=(3, 4, 5)).astype(np.float32)
    idx = np.array([39, 0, 20, 20], np.int32)
    got = run(lambda x, i, v: x.at[:, i].set(v), x, idx, v)
    want = np.array(x)
    for k, i in enumerate(idx):
        want[:, i] = v[:, k]
    assert_equal(got, want)


@pytest.mark.parametrize("trace", [False, True])
def test_dynamic_update_slice_cache(trace):
    # A KV-cache style write: one position in the batch and sequence dims,
    # full head dims before and after, without permuting the cache.
    rng = np.random.default_rng(14)
    cache = rng.normal(size=(2, 4, 16, 8)).astype(jnp.bfloat16)
    update = rng.normal(size=(2, 1, 1, 8)).astype(jnp.bfloat16)

    def write(cache, update, b, s):
        return jax.lax.dynamic_update_slice(cache, update, (0, b, s, 0))

    for b, s in ((1, 5), (3, 15), (0, 0), (9, 40)):
        got = run(write, cache, update, np.int32(b), np.int32(s), trace=trace)
        want = np.array(cache)
        bb, ss = min(b, 3), min(s, 15)
        want[:, bb, ss] = update[:, 0, 0]
        assert_equal(got, want)


def test_state_pool_update():
    # sglang's recurrent state pool: states.at[slots].set(new_states).
    rng = np.random.default_rng(15)
    pool = rng.normal(size=(32, 4, 64)).astype(np.float32)
    new = rng.normal(size=(3, 4, 64)).astype(np.float32)
    slots = np.array([31, 2, 2], np.int32)
    got = run(lambda p, s, n: p.at[s].set(n), pool, slots, new)
    want = np.array(pool)
    want[31] = new[0]
    want[2] = new[2]
    assert_equal(got, want)


def test_empty_updates():
    # Folds to the operand. (An empty gather result still crashes the layout
    # optimizer on a zero-volume tensor type, as it did before this lowering.)
    x = np.arange(20, dtype=np.float32).reshape(4, 5)
    idx = np.zeros((0,), np.int32)
    assert_equal(run(lambda x, i: x.at[i].set(jnp.zeros((0, 5), jnp.float32)), x, idx), x)
