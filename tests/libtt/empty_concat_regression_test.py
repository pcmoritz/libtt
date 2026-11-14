"""Concatenations whose result has no elements.

TTNN's row-major concatenation divides by the size of the last dimension, so a
concatenation of [5, 0] tensors along dimension 0 died with SIGFPE. Matmuls
that share an operand fuse into one over the concatenation of their weights,
so empty weights make one, as in PyTorch's linear backward over a zero-size
contracting dimension.
"""

import jax
import numpy as np
from jax import lax

EMPTY = np.zeros((5, 0), np.float32)


def test_matmuls_sharing_an_operand_over_empty_weights():
    def f(x):
        return [lax.dot_general(x, EMPTY, (((1,), (1,)), ((), ()))) + i for i in range(4)]

    for i, out in enumerate(jax.jit(f)(np.zeros((12, 0), np.float32))):
        np.testing.assert_array_equal(np.asarray(out), np.full((12, 5), i, np.float32))
