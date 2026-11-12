"""Scatter reductions into rows that most updates share.

The gradient of a token embedding sums the gradients of a batch's tokens into
the table rows of their ids. Text repeats a few common tokens, whose ids are
close together, so most updates land on a few nearby rows. rows_scatter
gives each core the updates to its own rows and combines them in software,
and it gave each core a contiguous range of rows: one core took most of the
updates (238 ms instead of 4 ms for 2048 tokens of a 151936 x 2048 BF16
table). A reduction now deals the rows out to the cores in turn.

Each core still applies the updates to a row in position order, so a BF16 sum
rounds after each update in the order NumPy's add.at does.
"""

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest


def _zipf_ids(rng, count, vocab):
    # Most ids small and repeated, a few anywhere in the table.
    ids = np.minimum(rng.zipf(1.3, count) - 1, 3000)
    ids[::97] = rng.integers(0, vocab, ids[::97].shape)
    return ids.astype(np.int32)


@pytest.mark.parametrize("dtype", [np.float32, ml_dtypes.bfloat16])
def test_embedding_gradient_of_repeated_tokens(dtype):
    rng = np.random.default_rng(0)
    vocab, width, tokens = 151936, 128, 2048
    ids = _zipf_ids(rng, tokens, vocab)
    grads = rng.integers(-3, 4, (tokens, width)).astype(dtype)
    run = jax.jit(lambda ids, g: jnp.zeros((vocab, width), dtype).at[ids].add(g))
    device = jax.devices("tt")[0]
    got = np.asarray(run(jax.device_put(ids, device), jax.device_put(grads, device)))
    want = np.zeros((vocab, width), dtype)
    np.add.at(want, ids, grads)
    np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("reduce", ["add", "max"])
def test_batched_table_reduction(reduce):
    rng = np.random.default_rng(1)
    batch, rows, width, updates = 3, 700, 64, 400
    table = rng.integers(-50, 50, (batch, rows, width)).astype(np.float32)
    ids = _zipf_ids(rng, updates, rows) % rows
    values = rng.integers(-50, 50, (batch, updates, width)).astype(np.float32)

    def scatter(table, ids, values):
        return getattr(table.at[:, ids], reduce)(values)

    device = jax.devices("tt")[0]
    got = np.asarray(jax.jit(scatter)(*(jax.device_put(a, device) for a in (table, ids, values))))
    want = table.copy()
    getattr(np, f"{'add' if reduce == 'add' else 'maximum'}").at(want, (slice(None), ids), values)
    np.testing.assert_array_equal(got, want)
