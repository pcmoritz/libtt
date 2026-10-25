"""A where over 32-bit values reads its BFloat16 predicate without a cast.

A comparison's result reaches TTNN as BFloat16. On tt-metal before
tenstorrent/tt-metal#60282, where took its output type and kernel variant from
the predicate, so tt-mlir cast every predicate to the values' type: a pass as
wide as the values per where, about 1 ms of SGLang-JAX's sampler step. tt-metal
now switches Dest to 32 bits beside 32-bit values and moves them as raw bits,
so the predicate stays BFloat16 (tt_mlir_where_bf16_predicate.patch). The
values here cover what a numeric path would lose: integers beyond 2**24, unsigned
integers beyond 2**31, -0.0 and infinities.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

SHAPE = (64, 300)


def _values(dtype, rng):
    if dtype == np.float32:
        v = rng.standard_normal(SHAPE).astype(np.float32) * 100
        v.flat[:4] = [-0.0, np.inf, -np.inf, 3.0e38]
    elif dtype == np.int32:
        v = rng.integers(-(2**31), 2**31 - 1, SHAPE, dtype=np.int64).astype(np.int32)
        v.flat[:3] = [2**31 - 1, -(2**31), 2**24 + 1]
    else:
        v = rng.integers(0, 2**32 - 1, SHAPE, dtype=np.uint64).astype(np.uint32)
        v.flat[:2] = [2**32 - 1, 2**31 + 1]
    return v


def _ttnn_ir(export_path):
    (ir,) = (export_path / "irs").glob("ttnn_runtime_*.mlir")
    return ir.read_text()


@pytest.mark.parametrize("level", ["O0", "O1"])
@pytest.mark.parametrize("form", ["full", "row", "scalar"])
@pytest.mark.parametrize("dtype", [np.float32, np.int32, np.uint32])
def test_where_selects_exactly_without_a_predicate_cast(dtype, form, level, tmp_path):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((SHAPE[0], 1 if form == "row" else SHAPE[1])).astype(np.float32)
    a, b = _values(dtype, rng), _values(dtype, rng)
    if form == "scalar":
        fn = lambda x, a, b: jnp.where(x > 0, a, jnp.zeros((), a.dtype))
        want = np.where(x > 0, a, np.zeros((), dtype))
    else:
        fn = lambda x, a, b: jnp.where(x > 0, a, b)
        want = np.where(x > 0, a, b)
    options = {"optimization_level": level, "export_path": str(tmp_path)}
    dev = jax.devices("tt")[0]
    got = np.asarray(jax.jit(fn, compiler_options=options)(*(jax.device_put(v, dev) for v in (x, a, b))))
    assert got.dtype == want.dtype
    np.testing.assert_array_equal(got.view(np.uint32), want.view(np.uint32))
    ir = _ttnn_ir(tmp_path)
    assert '"ttnn.typecast"' not in ir, "the predicate was cast"
