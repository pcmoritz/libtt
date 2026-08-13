"""Numerical tests for the StableHLO linear algebra fallbacks."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_plugins import libtt


def _cpu_jit(fn):
    return jax.jit(fn, backend="cpu")


@pytest.mark.parametrize("shape", [(4, 3), (3, 4), (5, 5)])
@pytest.mark.parametrize("full_matrices", [False, True])
def test_qr(shape, full_matrices):
    rng = np.random.default_rng(123)
    a = rng.normal(size=shape).astype(np.float32)
    q, r = _cpu_jit(partial(
        libtt._qr, pivoting=False, full_matrices=full_matrices,
        use_magma=None))(a)
    q, r = np.asarray(q), np.asarray(r)
    np.testing.assert_allclose(q.T @ q, np.eye(q.shape[1]), atol=2e-5)
    np.testing.assert_allclose(q @ r, a, atol=2e-5)


@pytest.mark.parametrize("shape", [(5, 3), (3, 5)])
def test_qr_rank_deficient(shape):
    row = np.arange(1, shape[1] + 1, dtype=np.float32)
    a = np.broadcast_to(row, shape).copy()
    q, r = _cpu_jit(partial(
        libtt._qr, pivoting=False, full_matrices=True,
        use_magma=None))(a)
    q, r = np.asarray(q), np.asarray(r)
    np.testing.assert_allclose(q.T @ q, np.eye(q.shape[1]), atol=3e-5)
    np.testing.assert_allclose(q @ r, a, atol=3e-5)


def test_qr_small_scale():
    rng = np.random.default_rng(234)
    a = (1e-12 * rng.normal(size=(5, 3))).astype(np.float32)
    q, r = _cpu_jit(partial(
        libtt._qr, pivoting=False, full_matrices=False,
        use_magma=None))(a)
    q, r = np.asarray(q), np.asarray(r)
    np.testing.assert_allclose(q.T @ q, np.eye(3), atol=2e-5)
    np.testing.assert_allclose(q @ r, a, rtol=5e-5, atol=1e-18)


def test_qr_batch():
    rng = np.random.default_rng(321)
    a = rng.normal(size=(2, 4, 3)).astype(np.float32)
    q, r = _cpu_jit(partial(
        libtt._qr, pivoting=False, full_matrices=False,
        use_magma=None))(a)
    np.testing.assert_allclose(
        np.swapaxes(q, -1, -2) @ q,
        np.broadcast_to(np.eye(3), (2, 3, 3)),
        atol=2e-5,
    )
    np.testing.assert_allclose(q @ r, a, atol=2e-5)


@pytest.mark.parametrize("shape", [(0, 3), (3, 0), (0, 0)])
@pytest.mark.parametrize("full_matrices", [False, True])
def test_qr_empty(shape, full_matrices):
    a = np.empty(shape, dtype=np.float32)
    q, r = _cpu_jit(partial(
        libtt._qr, pivoting=False, full_matrices=full_matrices,
        use_magma=None))(a)
    np.testing.assert_allclose(np.asarray(q) @ np.asarray(r), a)


def test_cholesky():
    rng = np.random.default_rng(456)
    factor = rng.normal(size=(6, 6)).astype(np.float32)
    a = factor @ factor.T + np.eye(6, dtype=np.float32)
    result = np.asarray(_cpu_jit(libtt._cholesky)(a))
    np.testing.assert_allclose(result @ result.T, a, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(result, np.tril(result), atol=0)


def test_cholesky_batch():
    rng = np.random.default_rng(654)
    factor = rng.normal(size=(2, 4, 4)).astype(np.float32)
    a = factor @ np.swapaxes(factor, -1, -2)
    a += 2 * np.eye(4, dtype=np.float32)
    result = np.asarray(_cpu_jit(libtt._cholesky)(a))
    np.testing.assert_allclose(
        result @ np.swapaxes(result, -1, -2), a,
        rtol=2e-5, atol=2e-5,
    )


def test_empty_factorizations():
    empty_matrix = np.empty((0, 0), dtype=np.float32)
    assert _cpu_jit(libtt._cholesky)(empty_matrix).shape == (0, 0)
    vectors, values = _cpu_jit(partial(
        libtt._eigh,
        lower=True,
        sort_eigenvalues=True,
        subset_by_index=None,
    ))(empty_matrix)
    assert vectors.shape == (0, 0)
    assert values.shape == (0,)

    result = _cpu_jit(partial(
        libtt._triangular_solve,
        left_side=True,
        lower=True,
        transpose_a=False,
        conjugate_a=False,
        unit_diagonal=False,
    ))(empty_matrix, np.empty((0, 2), dtype=np.float32))
    assert result.shape == (0, 2)


@pytest.mark.parametrize("left_side", [False, True])
@pytest.mark.parametrize("lower", [False, True])
@pytest.mark.parametrize("transpose_a", [False, True])
def test_triangular_solve(left_side, lower, transpose_a):
    rng = np.random.default_rng(789)
    a = rng.normal(size=(5, 5)).astype(np.float32)
    a = np.tril(a) if lower else np.triu(a)
    a[np.diag_indices(5)] += 4
    b_shape = (5, 3) if left_side else (3, 5)
    b = rng.normal(size=b_shape).astype(np.float32)
    result = _cpu_jit(partial(
        libtt._triangular_solve,
        left_side=left_side,
        lower=lower,
        transpose_a=transpose_a,
        conjugate_a=False,
        unit_diagonal=False,
    ))(a, b)
    op_a = a.T if transpose_a else a
    reconstructed = op_a @ result if left_side else result @ op_a
    np.testing.assert_allclose(reconstructed, b, rtol=2e-5, atol=2e-5)


def test_triangular_solve_complex_unit_diagonal():
    rng = np.random.default_rng(246)
    a = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    a = np.tril(a).astype(np.complex64)
    a[np.diag_indices(4)] = np.nan
    b = (rng.normal(size=(4, 2)) + 1j * rng.normal(size=(4, 2))).astype(
        np.complex64)
    result = _cpu_jit(partial(
        libtt._triangular_solve,
        left_side=True,
        lower=True,
        transpose_a=False,
        conjugate_a=True,
        unit_diagonal=True,
    ))(a, b)
    op_a = np.conj(a.copy())
    op_a[np.diag_indices(4)] = 1
    np.testing.assert_allclose(op_a @ result, b, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("complex_input", [False, True])
def test_eigh(complex_input):
    rng = np.random.default_rng(987)
    a = rng.normal(size=(6, 6)).astype(np.float32)
    if complex_input:
        a = a + 1j * rng.normal(size=(6, 6)).astype(np.float32)
    a = a + a.T.conj()
    vectors, values = _cpu_jit(partial(
        libtt._eigh,
        lower=True,
        sort_eigenvalues=True,
        subset_by_index=None,
    ))(a)
    vectors, values = np.asarray(vectors), np.asarray(values)
    np.testing.assert_allclose(
        vectors.T.conj() @ vectors, np.eye(6), rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(
        a @ vectors, vectors * values, rtol=2e-5, atol=2e-5)
    np.testing.assert_array_less(values[:-1], values[1:] + 1e-6)


def test_eigh_batch_and_subset():
    rng = np.random.default_rng(135)
    a = rng.normal(size=(2, 5, 5)).astype(np.float32)
    a += np.swapaxes(a, -1, -2)
    vectors, values = _cpu_jit(partial(
        libtt._eigh,
        lower=False,
        sort_eigenvalues=True,
        subset_by_index=(1, 4),
    ))(a)
    expected = np.linalg.eigvalsh(a)[:, 1:4]
    np.testing.assert_allclose(values, expected, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(
        a @ vectors, vectors * values[:, None, :],
        rtol=2e-5, atol=2e-5,
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
