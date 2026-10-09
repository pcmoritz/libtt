"""A column-broadcast operand of an elementwise op, [rows, 1] against
[rows, cols], is filled in L1 to whole tiles before the op: each row of the
tile takes its first value. The fill writes the left faces (rows 0 to 15 of
faces 0 and 2) and copies them onto the right faces. Every row has its own
value here, so a row taken from the wrong face or position changes the result;
17 rows cross the boundary between faces 0 and 2, 40 rows take two tile rows.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

ROWS = [1, 17, 40]


@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("dtype", [jnp.int32, jnp.float32, jnp.bfloat16])
def test_where_column_mask(rows, dtype):
    # A padding mask over tokens selecting their ids, as a MoE layer masks
    # its padding tokens' expert ids: a column-broadcast predicate and a scalar.
    rng = np.random.default_rng(rows)
    mask = rng.random(rows) < 0.5
    x = rng.integers(-100, 100, (rows, 8)).astype(dtype)
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
    column = rng.integers(-20, 20, (rows, 1)).astype(dtype)
    actual = jax.jit(lambda x, c: x * 2 + c * 3)(*jax.device_put((x, column), jax.devices("tt")[0]))
    np.testing.assert_array_equal(np.asarray(actual), (x * 2 + column * 3).astype(dtype))
