"""stablehlo.sort with a lexicographic comparator orders by every key.

torch_tpu's sparse coalesce sorts (row, col, iota) with the comparator
select(eq(row), select(eq(col), lt(iota), lt(col)), lt(row)), and moves
unique coordinates first with select(eq(flag), lt(iota), gt(flag)). JAX's
lax.sort with num_keys > 1 builds or(lt(k0), and(eq(k0), lt(k1))). The sort
used to order by the first key alone, and an ascending iota as the last key
only says that the sort is stable.
"""

from functools import partial

import jax
from jax import core
from jax import lax
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import hlo
from jax.extend.core import Primitive
from jax.interpreters import mlir
import jax.numpy as jnp
import numpy as np
import pytest

# stablehlo.sort with a comparator written with lax ops on scalars
# (lhs0, rhs0, lhs1, rhs1, ...), which lax.sort doesn't take.
sort_p = Primitive("test_sort")
sort_p.multiple_results = True
sort_p.def_abstract_eval(
    lambda *operands, comparator, is_stable: [core.ShapedArray(x.shape, x.dtype) for x in operands]
)


def _sort_lowering(ctx, *operands, comparator, is_stable):
    sort = hlo.SortOp(
        [mlir.aval_to_ir_type(ctx.module_context, aval) for aval in ctx.avals_out],
        operands,
        dimension=mlir.i64_attr(len(ctx.avals_in[0].shape) - 1),
        is_stable=ir.BoolAttr.get(is_stable),
    )
    scalars = [core.ShapedArray((), aval.dtype) for aval in ctx.avals_in for _ in range(2)]
    block = sort.comparator.blocks.append(
        *(mlir.aval_to_ir_type(ctx.module_context, aval) for aval in scalars)
    )
    with ir.InsertionPoint(block):
        sub_ctx = ctx.replace(primitive=None, avals_in=scalars, avals_out=[core.ShapedArray((), np.bool_)])
        out = mlir.lower_fun(comparator, multiple_results=False)(sub_ctx, *block.arguments)
        hlo.return_(mlir.ir_tree_registry.flatten(out)[0])
    return sort.results


mlir.register_lowering(sort_p, _sort_lowering)

COMPARE = {"LT": lax.lt, "GT": lax.gt}


def select_comparator(*args, directions):
    """torch_tpu's lexicographic comparator: one direction per key, keys first."""
    n = len(directions)
    result = COMPARE[directions[-1]](args[2 * n - 2], args[2 * n - 1])
    for i in reversed(range(n - 1)):
        lhs, rhs = args[2 * i], args[2 * i + 1]
        result = lax.select(lax.eq(lhs, rhs), result, COMPARE[directions[i]](lhs, rhs))
    return result


def sort(operands, comparator, is_stable=False):
    return jax.jit(lambda *xs: sort_p.bind(*xs, comparator=comparator, is_stable=is_stable))(*operands)


def lexsort_order(keys, directions):
    """The stable lexicographic order of keys[0], keys[1], ... (NumPy's lexsort
    takes the primary key last)."""
    signed = [-k.astype(np.float64) if d == "GT" else k for k, d in zip(keys, directions)]
    return np.lexsort(signed[::-1], axis=-1)


def ties(dtype, shape, values, seed):
    return np.random.default_rng(seed).integers(0, values, size=shape).astype(dtype)


@pytest.mark.parametrize("dtype", [np.float32, jnp.bfloat16, np.int32, np.uint32, np.bool_])
@pytest.mark.parametrize("direction", ["LT", "GT"])
def test_stable_sort_keeps_ties_in_order(dtype, direction):
    """A key and an ascending iota: equal keys keep their order."""
    keys = ties(dtype, (3, 200), 2 if dtype == np.bool_ else 5, seed=0)
    iota = np.broadcast_to(np.arange(200, dtype=np.int32), keys.shape)
    sorted_keys, indices = sort((keys, iota), partial(select_comparator, directions=(direction, "LT")))
    order = lexsort_order([keys], [direction])
    np.testing.assert_array_equal(indices, order)
    np.testing.assert_array_equal(sorted_keys, np.take_along_axis(keys, order, -1))


def test_sparse_coalesce_order():
    """torch_tpu's coalesce: (row, col) ascending, then iota; values follow."""
    rows, cols = ties(np.int32, (64,), 4, seed=1), ties(np.int32, (64,), 6, seed=2)
    iota = np.arange(64, dtype=np.int32)
    values = np.random.default_rng(3).standard_normal(64).astype(np.float32)
    got = sort((rows, cols, iota, values), partial(select_comparator, directions=("LT", "LT", "LT")))
    order = lexsort_order([rows, cols], ["LT", "LT"])
    for result, operand in zip(got, (rows, cols, iota, values)):
        np.testing.assert_array_equal(result, operand[order])


def flags_first(lhs_flag, rhs_flag, lhs_iota, rhs_iota):
    return lax.select(lax.eq(lhs_flag, rhs_flag), lax.lt(lhs_iota, rhs_iota), lax.gt(lhs_flag, rhs_flag))


def test_unique_flags_first():
    """torch_tpu's coalesce moves the first occurrences ahead, in order."""
    flags = np.array([True, False, True, False, True, True, False, False])
    iota = np.arange(8, dtype=np.int32)
    sorted_flags, indices = sort((flags, iota), flags_first)
    np.testing.assert_array_equal(indices, [0, 2, 4, 5, 1, 3, 6, 7])
    np.testing.assert_array_equal(sorted_flags, flags[[0, 2, 4, 5, 1, 3, 6, 7]])


def test_flags_from_a_reduction():
    """On device, reduce_or is a sum: a column with two nonzero rows is 2. The
    sort compares truth values (coalesce's flags are such a reduction)."""
    x = np.array([[1, 0, 1, 1, 0, 1, 0, 1], [1, 0, 0, 1, 0, 1, 1, 0]], np.int32)
    iota = np.arange(8, dtype=np.int32)
    flags, indices = jax.jit(
        lambda x, iota: sort_p.bind(jnp.any(x != 0, axis=0), iota, comparator=flags_first, is_stable=False)
    )(x, iota)
    np.testing.assert_array_equal(indices, [0, 2, 3, 5, 6, 7, 1, 4])
    np.testing.assert_array_equal(flags, [True] * 6 + [False] * 2)


@pytest.mark.parametrize("directions", [("LT", "LT"), ("LT", "GT"), ("GT", "LT"), ("GT", "GT")])
@pytest.mark.parametrize("dtype", [np.float32, np.int32])
def test_two_keys_and_payload(directions, dtype):
    """No iota: ties of both keys may come in any order, so the keys and the
    multiset of payloads per tie group are checked."""
    k0, k1 = ties(dtype, (2, 300), 3, seed=4), ties(dtype, (2, 300), 4, seed=5)
    payload = np.broadcast_to(np.arange(300, dtype=np.float32), k0.shape)
    s0, s1, sp = sort((k0, k1, payload), partial(select_comparator, directions=directions))
    order = lexsort_order([k0, k1], directions)
    np.testing.assert_array_equal(s0, np.take_along_axis(k0, order, -1))
    np.testing.assert_array_equal(s1, np.take_along_axis(k1, order, -1))
    np.testing.assert_array_equal(np.take_along_axis(k0, sp.astype(np.int64), -1), s0)
    np.testing.assert_array_equal(np.take_along_axis(k1, sp.astype(np.int64), -1), s1)
    np.testing.assert_array_equal(np.sort(sp, -1), payload)


@pytest.mark.parametrize("num_keys", [2, 3])
def test_jax_lax_sort_num_keys(num_keys):
    """JAX's or(lt, and(eq, ...)) comparator, as in jnp.lexsort."""
    keys = [ties(np.int32, (500,), 3, seed=6 + i) for i in range(num_keys)]
    iota = np.arange(500, dtype=np.int32)
    got = jax.jit(lambda *xs: lax.sort(xs, num_keys=num_keys + 1))(*keys, iota)
    order = lexsort_order(keys, ["LT"] * num_keys)
    for result, operand in zip(got, (*keys, iota)):
        np.testing.assert_array_equal(result, operand[order])


def test_jnp_lexsort():
    k0, k1 = ties(np.float32, (400,), 5, seed=9), ties(np.float32, (400,), 5, seed=10)
    np.testing.assert_array_equal(jax.jit(jnp.lexsort)((k1, k0)), np.lexsort((k1, k0)))


def test_int64_keys():
    """64-bit keys, as torch_tpu's coalesce passes them, are 32-bit on device."""
    with jax.enable_x64(True):
        rows, cols = ties(np.int64, (64,), 4, seed=11), ties(np.int64, (64,), 6, seed=12)
        iota = np.arange(64, dtype=np.int64)
        got = sort((rows, cols, iota), partial(select_comparator, directions=("LT", "LT", "LT")))
    order = lexsort_order([rows, cols], ["LT", "LT"])
    for result, operand in zip(got, (rows, cols, iota)):
        np.testing.assert_array_equal(result, operand[order])


@pytest.mark.parametrize(
    "transform",
    [lambda x: lax.select(lax.lt(x, 0.0), 1.0, 0.0), lax.neg],
    ids=["sign_group", "negated"],
)
def test_transformed_keys_are_errors(transform):
    """A comparator of other values than the keys (or JAX's canonical floats)
    isn't recognized, rather than sorted by the raw keys."""
    keys = np.random.default_rng(13).standard_normal(16).astype(np.float32)
    with pytest.raises(jax.errors.JaxRuntimeError, match="isn't a lexicographic order of its inputs"):
        sort((keys,), lambda a, b: lax.lt(transform(a), transform(b)))


@pytest.mark.parametrize(
    "dtype, n",
    [(np.int8, 200), (np.uint8, 300), (np.bool_, 200), (np.int32, 300), (np.float32, 300)],
)
def test_converted_iota_as_last_key(dtype, n):
    """An iota converted to a type that doesn't hold every index (it wraps,
    saturates or repeats; StableHLO leaves out-of-range integer conversions to
    the implementation) is a key like any other, not a stability tie-break.
    On the device, int8 holds 128..199 (it is int32) but uint8 wraps at 256,
    which orders differently from the original indices."""
    keys = ties(np.int32, (n,), 3, seed=14)

    def f(k):
        converted = lax.iota(np.int32, n).astype(dtype)
        comparator = partial(select_comparator, directions=("LT", "LT"))
        return *sort_p.bind(k, converted, comparator=comparator, is_stable=False), converted

    got_keys, got_converted, converted = jax.jit(f)(keys)
    order = lexsort_order([keys, converted.astype(np.float64)], ["LT", "LT"])
    np.testing.assert_array_equal(got_keys, keys[order])
    np.testing.assert_array_equal(got_converted, converted[order])


def test_canonical_float_keys():
    """JAX canonicalizes float keys in its comparators: -0 equals 0 and NaNs sort
    last."""
    keys = np.array([0.5, -0.0, np.nan, 0.0, -1.0, np.nan, 0.5, -0.0], np.float32)
    iota = np.arange(8, dtype=np.int32)
    got_keys, got_iota = jax.jit(lambda k, i: lax.sort((k, i), num_keys=1, is_stable=True))(keys, iota)
    np.testing.assert_array_equal(got_iota, [4, 1, 3, 7, 0, 6, 2, 5])


@pytest.mark.parametrize("dtype", [np.int8, np.uint8])
def test_narrow_integer_payloads(dtype):
    """8-bit integers move with the keys exactly (they are gathered as int32)."""
    rng = np.random.default_rng(15)
    keys = rng.standard_normal((4, 300)).astype(np.float32)
    info = np.iinfo(dtype)
    values = rng.integers(info.min, info.max, size=keys.shape, dtype=dtype, endpoint=True)
    for _ in range(3):
        got_keys, got_values = jax.jit(lambda k, v: lax.sort((k, v), num_keys=1, is_stable=True))(keys, values)
        order = np.argsort(keys, axis=-1, kind="stable")
        np.testing.assert_array_equal(got_keys, np.take_along_axis(keys, order, -1))
        np.testing.assert_array_equal(got_values, np.take_along_axis(values, order, -1))
