"""Triangular solves run as matmuls.

stablehlo.triangular_solve (jax.lax.linalg.triangular_solve, PyTorch's
torch.linalg.solve_triangular) has no TTNN op. As XLA does for TPUs, it is
expanded into matmuls at the highest precision: the inverse of the triangular
matrix comes from its diagonal by recursive block inversion. Only the triangle
the op names is read: the other one holds values that must not matter.
"""

import jax
from jax import lax
import numpy as np
import pytest


def _matrix(rng, shape, lower, unit_diagonal):
    """Triangular matrices with values in the other triangle (and on the diagonal of a unit one) that the solve
    ignores, and the triangular matrix it solves with in float64."""
    a = rng.standard_normal(shape) * 0.2
    m = shape[-1]
    diagonal = np.eye(m, dtype=bool)
    a = np.where(diagonal, 7.0 if unit_diagonal else rng.uniform(1, 2, shape), a)
    triangle = np.tril(np.ones((m, m)), -1) if lower else np.triu(np.ones((m, m)), 1)
    a_used = a * triangle + (np.eye(m) if unit_diagonal else a * diagonal)
    return a.astype(np.float32), a_used


def _relative_error(got, want):
    return np.max(np.abs(np.asarray(got, np.float64) - want)) / np.max(np.abs(want))


@pytest.mark.parametrize("left_side", [True, False])
@pytest.mark.parametrize("lower", [True, False])
@pytest.mark.parametrize("transpose_a", [False, True])
@pytest.mark.parametrize("unit_diagonal", [False, True])
def test_triangular_solve(left_side, lower, transpose_a, unit_diagonal):
    # 48 isn't a power of 2: the last inversion round's blocks are partial.
    rng = np.random.default_rng(0)
    a, a_used = _matrix(rng, (2, 3, 48, 48), lower, unit_diagonal)
    b = rng.standard_normal((2, 3, 48, 16) if left_side else (2, 3, 16, 48)).astype(np.float32)
    x = jax.jit(
        lambda a, b: lax.linalg.triangular_solve(
            a, b, left_side=left_side, lower=lower, transpose_a=transpose_a, unit_diagonal=unit_diagonal
        )
    )(a, b)
    op_a = np.swapaxes(a_used, -1, -2) if transpose_a else a_used
    want = np.linalg.solve(op_a, b) if left_side else np.swapaxes(
        np.linalg.solve(np.swapaxes(op_a, -1, -2), np.swapaxes(b, -1, -2)), -1, -2
    )
    assert x.shape == b.shape and x.dtype == np.float32
    assert _relative_error(x, want) < 1e-4


def test_gated_delta_chunks():
    # The solve of a chunked gated delta rule (TorchTitan's Qwen3.5 GatedDeltaNet): unit lower triangular 64 x 64
    # matrices for every chunk and head, the strictly lower triangle near 1 (similar keys, little decay), against
    # a right-hand side of 256 columns.
    rng = np.random.default_rng(1)
    a = np.tril(rng.uniform(0.9, 1.0, (1, 4, 8, 64, 64)), -1) + np.eye(64)
    b = rng.standard_normal((1, 4, 8, 64, 256))
    x = jax.jit(lambda a, b: lax.linalg.triangular_solve(a, b, left_side=True, lower=True, unit_diagonal=True))(
        a.astype(np.float32), b.astype(np.float32)
    )
    assert _relative_error(x, np.linalg.solve(a, b)) < 1e-4
