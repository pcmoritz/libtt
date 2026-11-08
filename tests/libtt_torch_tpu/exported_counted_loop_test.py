"""A JAX op that torch_tpu runs may loop a fixed number of times.

torch_tpu hands libtt an exported JAX function as StableHLO converted from
XLA's HLO, which leaves out the comparison type of the loop condition (signed
for integers). libtt didn't recognize such a fori_loop as counted, so it
neither unrolled it nor skipped its condition: SGLang-TorchTPU's top-p and
top-k filters search their cutoff bit by bit in 30 iterations, each reading
its condition back to the host.
"""

import jax
import jax.numpy as jnp
import numpy as np
import torch
from jax import lax
from torch_tpu._internal import pallas


def cutoff_search(probs: jax.Array, top_ps: jax.Array) -> jax.Array:
    """The largest float32 cutoff whose kept mass reaches top_p, bit by bit."""

    def step(i, bits):
        candidate = bits | jnp.left_shift(1, 29 - i)
        threshold = lax.bitcast_convert_type(candidate, jnp.float32)
        mass = jnp.sum(jnp.where(probs >= threshold[:, None], probs, 0.0), axis=-1)
        return jnp.where(mass >= top_ps, candidate, bits)

    bits = lax.fori_loop(0, 30, step, jnp.zeros(probs.shape[0], jnp.int32))
    return lax.bitcast_convert_type(bits, jnp.float32)


def reference(probs, top_ps):
    out = []
    for row, top_p in zip(probs, top_ps):
        bits = np.int32(0)
        for i in range(30):
            candidate = bits | np.int32(1 << (29 - i))
            threshold = np.array(candidate, np.int32).view(np.float32)
            if row[row >= threshold].sum(dtype=np.float32) >= top_p:
                bits = candidate
        out.append(np.array(bits, np.int32).view(np.float32))
    return np.array(out, np.float32)


def test_exported_fori_loop():
    op = pallas.jax_op("libtt_test::cutoff_search", cutoff_search)
    rng = np.random.default_rng(3)
    logits = rng.normal(0, 2.5, (2, 1024)).astype(np.float32)
    probs = np.exp(logits / 0.7) / np.exp(logits / 0.7).sum(axis=-1, keepdims=True)
    probs = probs.astype(np.float32)
    top_ps = np.array([0.5, 0.95], np.float32)
    for _ in range(3):
        got = op(torch.from_numpy(probs).to("tpu"), torch.from_numpy(top_ps).to("tpu"))
        np.testing.assert_array_equal(got.cpu().numpy(), reference(probs, top_ps))
