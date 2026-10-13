"""lax.top_k on TTNN's selection path: up to two tile rows and wide rows.

The data-movement cores scan each row once, keep the best k with the lower
index first among equal values, and order floats like lax.top_k on CPU (IEEE
total order: NaN above inf, -0.0 below +0.0). Wide rows split into column
segments that the tile row's first core merges.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax


def reference(x, k):
    # lax.top_k on CPU is the contract, including NaN and signed zeros.
    with jax.default_device(jax.devices("cpu")[0]):
        values, indices = lax.top_k(jnp.asarray(x), k)
    return np.asarray(values), np.asarray(indices)


def run(x, k):
    return jax.jit(lambda a: lax.top_k(a, k))(jax.device_put(x, jax.devices("tt")[0]))


def check(x, k):
    values, indices = run(x, k)
    want_values, want_indices = reference(x, k)
    assert values.shape == indices.shape == x.shape[:-1] + (k,)
    np.testing.assert_array_equal(np.asarray(indices), want_indices)
    # Compare bits: NaN payloads and signed zeros included.
    got = np.asarray(values).astype(np.float32).view(np.uint32)
    np.testing.assert_array_equal(got, want_values.astype(np.float32).view(np.uint32))


@pytest.mark.parametrize("dtype", [np.float32, "bfloat16"])
@pytest.mark.parametrize(
    "shape,k",
    [
        # Routers: one token and a few, 32 to 512 experts.
        ((1, 32), 4),
        ((1, 60), 4),
        ((1, 128), 8),
        ((1, 128), 1),
        ((3, 128), 5),
        ((32, 128), 8),
        ((33, 256), 64),
        ((64, 512), 10),
        # Widths off the tile grid and rows past one tile.
        ((1, 1000), 33),
        ((40, 2048), 16),
        # Wide rows split into segments: logits over a vocabulary.
        ((1, 32768), 20),
        ((1, 151936), 20),
        ((8, 50000), 5),
        ((3, 4097), 64),
        # Too many candidates for one core per row: fewer segments, more row groups.
        ((32, 32768), 64),
        # Several [H, W] slabs.
        ((2, 3, 4096), 7),
    ],
)
def test_top_k_selection(dtype, shape, k):
    rng = np.random.default_rng(sum(shape) + k)
    # Few distinct values give long runs of ties.
    x = rng.integers(-50, 50, shape).astype(np.float32).astype(dtype)
    check(x, k)


# Special values by their bits as float32 and as bfloat16. lax.top_k on CPU returns bfloat16 NaNs with the
# canonical payload, so only float32 carries the other NaN payloads.
SPECIALS = {
    "+nan": (0x7FC00000, 0x7FC0),
    "+nan2": (0x7FC00123, 0x7FC0),
    "+snan": (0x7F800001, 0x7FC0),
    "-nan": (0xFFC00000, 0xFFC0),
    "-nan2": (0xFFC00123, 0xFFC0),
    "+inf": (0x7F800000, 0x7F80),
    "-inf": (0xFF800000, 0xFF80),
    "+max": (0x7F7FFFFF, 0x7F7F),
    "-max": (0xFF7FFFFF, 0xFF7F),
    "+1": (0x3F800000, 0x3F80),
    "-1": (0xBF800000, 0xBF80),
    "+den": (0x00000001, 0x0001),
    "-den": (0x80000001, 0x8001),
    "+0": (0x00000000, 0x0000),
    "-0": (0x80000000, 0x8000),
}


def specials(names, dtype):
    if dtype == np.float32:
        return np.array([SPECIALS[n][0] for n in names], np.uint32).view(np.float32)
    return np.array([SPECIALS[n][1] for n in names], np.uint16).view(jnp.bfloat16)


@pytest.mark.parametrize("dtype", [np.float32, "bfloat16"])
def test_top_k_selection_orders_special_values(dtype):
    """Rows of only special values, each at least twice, and k one below the width: all of lax.top_k's order but
    the lowest value, ties by index. (k equal to the width compiles to a full sort, not top_k.)"""
    names = list(SPECIALS) * 2 + ["+0", "-0", "-nan2"]
    rng = np.random.default_rng(0)
    check(np.stack([specials(rng.permutation(names), dtype) for _ in range(3)]), len(names) - 1)


@pytest.mark.parametrize("dtype", [np.float32, "bfloat16"])
def test_top_k_selection_boundary_among_zeros_and_nans(dtype):
    """The 12th of 5000 values, in rows split into segments, is a -0.0 in the first row and a -NaN in the
    second, with more of each left out: signed zeros, -inf and -NaN are ranked against each other and by
    index across segments."""
    width = 5000
    zeros = ["-1"] * width
    for col, name in {4900: "+nan", 3: "+inf", 777: "+max", 2500: "+1", 1200: "+den"}.items():
        zeros[col] = name
    for col in (4000, 100, 3300, 2100):
        zeros[col] = "+0"
    for col in (10, 4500, 600, 1700, 2900, 4999):
        zeros[col] = "-0"
    nans = ["-nan2"] * width
    for col, name in {4321: "-nan", 50: "-nan", 2600: "-nan", 1000: "-inf", 3999: "-inf", 2222: "-1", 4444: "-0"}.items():
        nans[col] = name
    check(np.stack([specials(zeros, dtype), specials(nans, dtype)]), 12)


@pytest.mark.parametrize("dtype", [np.float32, "bfloat16"])
def test_top_k_selection_all_equal(dtype):
    """Every value tied: the first k indices, in order."""
    check(np.full((2, 3000), 1.5, np.float32).astype(dtype), 9)


@pytest.mark.parametrize("level", ["O0", "O1"])
def test_top_k_int32_indices_need_no_typecast(level, tmp_path):
    # 32-bit indices come out of ttnn.topk as int32, lax.top_k's dtype, with no typecast after it. At O1
    # the optimizer checks the op's output types against ttnn's own.
    x = np.random.default_rng(0).standard_normal((1, 128)).astype(np.float32)
    options = {"optimization_level": level, "export_path": str(tmp_path)}
    f = jax.jit(lambda a: lax.top_k(a, 8), compiler_options=options)
    values, indices = f(jax.device_put(x, jax.devices("tt")[0]))
    want_values, want_indices = reference(x, 8)
    np.testing.assert_array_equal(np.asarray(indices), want_indices)
    np.testing.assert_array_equal(np.asarray(values), want_values)
    ir = "\n".join(path.read_text() for path in (tmp_path / "irs").glob("ttnn_[0-9]*.mlir"))
    assert "ttnn.topk" in ir
    assert "ttnn.typecast" not in ir
