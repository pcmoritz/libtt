"""A column-broadcast operand of an elementwise op, [rows, 1] against
[rows, cols], is filled in L1 to whole tiles before the op: each row of the
tile takes its first value. The fill writes the left faces (rows 0 to 15 of
faces 0 and 2) and copies them onto the right faces. Every row has its own
value here, so a row taken from the wrong face or position changes the result;
17 rows cross the boundary between faces 0 and 2, 40 rows take two tile rows,
and results in columns 16 and up come from the copied right faces. A tile
row with at most 16 rows of data writes each of them into both top faces
instead, and leaves the padding rows unfilled.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

ROWS = [1, 17, 40]


@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("cols", [8, 40])
@pytest.mark.parametrize("dtype", [jnp.int32, jnp.float32, jnp.bfloat16])
def test_where_column_mask(rows, cols, dtype):
    # A padding mask over tokens selecting their ids, as a MoE layer masks
    # its padding tokens' expert ids (8 per token): a column-broadcast
    # predicate and a scalar. 40 columns reach the right faces and a second
    # tile column.
    rng = np.random.default_rng(rows * 100 + cols)
    mask = rng.random(rows) < 0.5
    x = rng.integers(-100, 100, (rows, cols)).astype(dtype)
    actual = jax.jit(lambda m, x: jnp.where(m[:, None], x, -1))(
        *jax.device_put((mask, x), jax.devices("tt")[0])
    )
    np.testing.assert_array_equal(np.asarray(actual), np.where(mask[:, None], x, -1).astype(dtype))


@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("dtype", [jnp.int32, jnp.bfloat16])
def test_add_column(rows, dtype):
    # Small integers, exact in bfloat16 too. The operands are computed on the
    # device, so the add reads them as tiles (inputs from the host would be read
    # row-major, which fills rows another way).
    rng = np.random.default_rng(100 + rows)
    x = rng.integers(-20, 20, (rows, 96)).astype(dtype)
    column = (np.arange(rows) - rows // 2).reshape(rows, 1).astype(dtype)
    actual = jax.jit(lambda x, c: x * 2 + c * 3)(*jax.device_put((x, column), jax.devices("tt")[0]))
    np.testing.assert_array_equal(np.asarray(actual), (x * 2 + column * 3).astype(dtype))


@pytest.mark.parametrize("rows", [1, 5, 17, 40])
@pytest.mark.parametrize("dtype", [jnp.int32, jnp.float32, jnp.bfloat16])
def test_where_column_operands(rows, dtype):
    # A column predicate and a column false value, as a MoE router masks its
    # expert ids: the where reads both as column broadcasts. The operands are
    # computed on the device, so they are tiles.
    rng = np.random.default_rng(200 + rows)
    x = rng.integers(-100, 100, (rows, 40)).astype(dtype)
    m = rng.integers(-1, 2, rows).astype(dtype)
    c = (np.arange(rows) - rows // 2).astype(dtype)
    actual = jax.jit(lambda x, m, c: jnp.where((m > 0)[:, None], x, (c * 2)[:, None]))(
        *jax.device_put((x, m, c), jax.devices("tt")[0])
    )
    expected = np.where((m > 0)[:, None], x, (c * 2)[:, None]).astype(dtype)
    np.testing.assert_array_equal(np.asarray(actual), expected)
