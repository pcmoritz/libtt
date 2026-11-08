"""Variadic max / min reductions that return both values and indices.

PyTorch's torch.max(x, dim) is a (value, index) reduce whose body compares with
>= (<= for min), keeps the maximum, and breaks ties towards the smaller index,
with the lowest (highest) finite value as init. Both results are used, and the
init takes part in them: a row of -inf has the maximum finfo.min, and its
index 0 wins wherever it ties the maximum, as in [-inf, finfo.min].

Only that reducer may become an argmax (argmin), which picks the smaller index on
ties. A reducer that picks the larger one must not.
"""

import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
import pytest


def _arg_reduce(x, is_max, init_value, nan_aware=False, tie_break=jnp.minimum):
    def body(a, b):
        (va, ia), (vb, ib) = a, b
        better = va >= vb if is_max else va <= vb
        tie = va == vb
        if nan_aware:  # PyTorch's float reducer prefers NaNs.
            better = better | ((va != va) & (vb == vb))
            tie = tie | ((va != va) & (vb != vb))
        value = jnp.maximum(va, vb) if is_max else jnp.minimum(va, vb)
        index = jnp.where(tie, tie_break(ia, ib), jnp.where(better, ia, ib))
        return value, index

    init = (jnp.array(init_value, x.dtype), jnp.array(0, jnp.int32))
    iota = lax.broadcasted_iota(jnp.int32, x.shape, 1)
    return lax.reduce((x, iota), init, body, (1,))


@pytest.mark.parametrize("is_max", [True, False])
@pytest.mark.parametrize("finite_init", [True, False])
@pytest.mark.parametrize("nan_aware", [True, False])
def test_values_and_indices(is_max, finite_init, nan_aware):
    x = np.random.default_rng(0).standard_normal((4, 6)).astype(np.float32)
    # A row of the reduction's identity, a row that reaches it once, and a row
    # whose extreme is the finite init value.
    extreme = -np.inf if is_max else np.inf
    info = np.finfo(np.float32)
    finite = info.min if is_max else info.max
    x[1] = extreme
    x[2, 3] = extreme
    x[3] = extreme
    x[3, 2] = finite
    init = finite if finite_init else extreme
    values, indices = jax.jit(lambda x: _arg_reduce(x, is_max, init, nan_aware))(x)
    # The init (value, index 0) wins ties, so its index wins wherever it is at
    # least the maximum (at most the minimum).
    if is_max:
        expected_values = np.maximum(np.max(x, axis=1), init)
        init_wins = init >= np.max(x, axis=1)
        expected_indices = np.where(init_wins, 0, np.argmax(x, axis=1))
    else:
        expected_values = np.minimum(np.min(x, axis=1), init)
        init_wins = init <= np.min(x, axis=1)
        expected_indices = np.where(init_wins, 0, np.argmin(x, axis=1))
    np.testing.assert_array_equal(values, expected_values)
    np.testing.assert_array_equal(indices, expected_indices)


@pytest.mark.parametrize("is_max", [True, False])
def test_ties_to_the_larger_index(is_max):
    x = np.array([[2.0, 2.0]], np.float32)
    info = np.finfo(np.float32)
    init = info.min if is_max else info.max
    reduce = jax.jit(lambda x: _arg_reduce(x, is_max, init, tie_break=jnp.maximum))
    try:
        values, indices = reduce(x)
    except jax.errors.JaxRuntimeError:
        return  # Variadic reduces in general don't lower; a wrong index would.
    np.testing.assert_array_equal(values, [2.0])
    np.testing.assert_array_equal(indices, [1])


def test_sums_over_a_unit_dimension():
    # Over a dimension of size 1, an argmin's inputs and reducer aren't checked.
    # A reduce that sums x and x * x there, as layer_norm's backward does, is
    # still not an argmin.
    x = np.random.default_rng(0).standard_normal((3, 4, 1)).astype(np.float32)
    zero = np.float32(0)
    sums = lambda x: lax.reduce((x, x * x), (zero, zero), lambda a, b: (a[0] + b[0], a[1] + b[1]), (2,))
    total, squares = jax.jit(sums)(x)
    np.testing.assert_array_equal(total, x[..., 0])
    np.testing.assert_allclose(squares, x[..., 0] ** 2, rtol=1e-6)
    np.testing.assert_array_equal(jax.jit(lambda x: jnp.argmin(x, axis=2))(x), np.zeros((3, 4)))


@pytest.mark.parametrize("dtype", [np.float32, np.int32, np.bool_])
@pytest.mark.parametrize("shape,axis", [((4, 1), 1), ((1, 8), 0), ((2, 1, 3), 1)])
def test_argmax_over_a_unit_dimension(shape, axis, dtype):
    """Over a dimension of size 1, the reduce's iota of indices folds to a zero constant, and
    it is still an argmax: JAX's, and PyTorch's, which EAGLE's tree verification takes over a
    single draft token."""
    rng = np.random.default_rng(0)
    x = rng.random(shape) > 0.5 if dtype == np.bool_ else rng.integers(-5, 5, shape).astype(dtype)
    expected = np.zeros(np.delete(shape, axis), np.int32)
    np.testing.assert_array_equal(jax.jit(lambda x: jnp.argmax(x, axis=axis))(x), expected)
    if axis == 1 and dtype != np.bool_:
        lowest = np.finfo(dtype).min if dtype == np.float32 else np.iinfo(dtype).min
        _, indices = jax.jit(lambda x: _arg_reduce(x, True, lowest))(x)
        np.testing.assert_array_equal(indices, expected)


@pytest.mark.parametrize("is_max", [True, False])
@pytest.mark.parametrize("dtype", [np.int32, np.bool_])
@pytest.mark.parametrize("shape", [(1, 2), (4, 8)])
def test_jax_reducer_in_another_order(shape, dtype, is_max):
    """JAX's argmax (argmin) reducer, with the value's select right after the comparison and
    no NaN term, as canonicalization leaves it for SGLang's vision attention: still an argmax
    (argmin)."""

    def body(a, b):
        (va, ia), (vb, ib) = a, b
        better = va > vb if is_max else va < vb
        value = jnp.where(better, va, vb)
        index = jnp.where(better | ((va == vb) & (ia < ib)), ia, ib)
        return value, index

    x = np.random.default_rng(1).integers(0, 3, shape).astype(dtype)
    if dtype == np.bool_:
        init = np.array(not is_max)
    else:
        init = np.array(np.iinfo(dtype).min if is_max else np.iinfo(dtype).max, dtype)

    def reduce(x):
        iota = lax.broadcasted_iota(jnp.int32, x.shape, 1)
        return lax.reduce((x, iota), (init, np.int32(0)), body, (1,))[1]

    expected = np.argmax(x, axis=1) if is_max else np.argmin(x, axis=1)
    np.testing.assert_array_equal(jax.jit(reduce)(x), expected)


@pytest.mark.parametrize("dtype", [np.float32, jnp.bfloat16])
@pytest.mark.parametrize("trace", [False, True])
@pytest.mark.parametrize("rows,width", [(1, 248320), (4, 2048), (20, 96)])
# O1's optimizer must keep the input tiled too (its own argmax rule wants it
# row-major).
@pytest.mark.parametrize("level", ["O0", "O1"])
def test_index_of_a_tiled_row(dtype, trace, rows, width, level, tmp_path):
    """torch_tpu's greedy sampling argmax: only the index of a reduce from
    finfo.min. The rows of one tile row stay tiled and libtt's row argmax takes
    the init as its floor, with no maximum of the input against it: index 0
    where no element exceeds the init, as in a row of -inf or [-inf, finfo.min],
    else the first maximum."""
    info = np.finfo(np.float32)
    x = np.random.default_rng(rows).standard_normal((rows, width)).astype(np.float32)
    x[:, width // 3] = x[:, width // 5]  # ties go to the first index
    if rows > 2:
        x[1] = -np.inf
        x[2] = -np.inf
        x[2, 3] = info.min
    x = x.astype(dtype)
    run = jax.jit(
        lambda x: _arg_reduce(x, True, info.min, nan_aware=True)[1],
        compiler_options={"optimization_level": level, "enable_trace": str(trace).lower(), "export_path": str(tmp_path)},
    )
    got = np.asarray(run(jax.device_put(x, jax.devices("tt")[0])))
    xf = np.asarray(x, np.float32)
    want = np.where(np.max(xf, axis=1) > info.min, np.argmax(xf, axis=1), 0)
    np.testing.assert_array_equal(got, want)
    irs = [path.read_text() for path in (tmp_path / "irs").glob("ttnn_[0-9]*.mlir")]
    assert irs and all('"ttnn.maximum"' not in ir for ir in irs), "the init was combined into the input"
    # In BFLOAT16 finfo(float32).min rounds to -inf, the reduction's identity:
    # no floor then.
    if dtype == np.float32:
        assert all("floor = " in ir for ir in irs if '"ttnn.argmax"' in ir)
