"""JAX plugin and lowering extensions for the libtt PJRT backend.

JAX implements SVD for non-CPU/GPU platforms with a QDWH algorithm.  That
algorithm is expressed in terms of QR, Cholesky, triangular solve, and
Hermitian eigendecomposition primitives.  OpenXLA normally delegates those
primitives to backend-specific libraries; TT-MLIR does not currently provide
such library calls.

This module supplies numerically stable, JAX-expressible fallbacks for the TT
platform.  JAX discovers it through the ``jax_plugins`` namespace and calls
``initialize`` before creating the PJRT client.  The implementations lower to
ordinary StableHLO, so they require no host callbacks and remain compatible
with JIT and batching.
"""

from __future__ import annotations

from functools import partial
import os
from pathlib import Path

import jax
from jax import lax
import jax.numpy as jnp


def _adjoint(x):
    return jnp.swapaxes(jnp.conj(x), -1, -2)


def _vmap_batch(fn, x, batch_rank):
    for _ in range(batch_rank):
        fn = jax.vmap(fn)
    return fn(x)


def _orthogonal_candidate(q):
    """Returns a stable unit vector orthogonal to the populated columns."""
    m = q.shape[0]
    projector = jnp.eye(m, dtype=q.dtype) - q @ _adjoint(q)
    norms_sq = jnp.real(jnp.sum(jnp.conj(projector) * projector, axis=0))
    column = jnp.argmax(norms_sq)
    selector = (jnp.arange(m) == column).astype(q.dtype)
    candidate = projector @ selector
    norm = jnp.sqrt(jnp.maximum(jnp.max(norms_sq), 0))
    # ``projector`` starts as the identity.  The guard also keeps an exactly
    # rank-deficient input from introducing NaNs.
    safe_norm = jnp.where(norm > 0, norm, jnp.array(1, norm.dtype))
    return candidate / safe_norm.astype(candidate.dtype)


def _qr_2d(a, *, full_matrices):
    """Modified Gram-Schmidt QR with reorthogonalization and completion."""
    m, n = a.shape
    k = m if full_matrices else min(m, n)
    if m == 0 or n == 0:
        q = jnp.eye(m, k, dtype=a.dtype)
        return q, jnp.zeros((k, n), dtype=a.dtype)

    q = jnp.zeros((m, k), dtype=a.dtype)
    eps = jnp.finfo(a.dtype).eps
    scale = jnp.sqrt(jnp.real(jnp.sum(jnp.conj(a) * a)))

    def add_column(q, j):
        source = a[:, j] if j < n else _orthogonal_candidate(q)

        # Two MGS passes avoid the loss of orthogonality of classical
        # Gram-Schmidt for ill-conditioned columns.
        def reorthogonalize(vector):
            coeffs = _adjoint(q) @ vector
            active = jnp.arange(k) < j
            coeffs = jnp.where(active, coeffs, jnp.zeros_like(coeffs))
            return vector - q @ coeffs

        # The two passes are intentionally explicit.  Besides making the
        # fixed amount of work visible to the compiler, this avoids adding an
        # unnecessary third nested loop when QR is called from QDWH.
        vector = reorthogonalize(source)
        vector = reorthogonalize(vector)
        norm = jnp.sqrt(jnp.maximum(
            jnp.real(jnp.vdot(vector, vector)), jnp.array(0, eps.dtype)))
        use_fallback = norm <= eps * scale
        fallback = _orthogonal_candidate(q)
        safe_norm = jnp.where(use_fallback, jnp.array(1, norm.dtype), norm)
        normalized = vector / safe_norm.astype(vector.dtype)
        vector = jnp.where(use_fallback, fallback, normalized)
        return q.at[:, j].set(vector)

    # Shapes are static in JAX, so the columns are deliberately expanded here.
    # This keeps QR loop-free when QDWH places it inside its own iteration.
    for j in range(k):
        q = add_column(q, j)
    r = _adjoint(q) @ a
    return q, r


def _qr(a, *, pivoting, full_matrices, use_magma):
    del use_magma
    if pivoting:
        raise NotImplementedError("pivoted QR is not implemented for TT")
    fn = partial(_qr_2d, full_matrices=full_matrices)
    for _ in range(a.ndim - 2):
        fn = jax.vmap(fn)
    return fn(a)


def _cholesky_2d(a):
    """Unblocked lower Cholesky factorization."""
    n = a.shape[0]
    if n == 0:
        return a
    result = jnp.zeros_like(a)
    indices = jnp.arange(n)

    def factor_column(j, result):
        row = lax.dynamic_index_in_dim(result, j, axis=0, keepdims=False)
        column_residual = (
            lax.dynamic_index_in_dim(a, j, axis=1, keepdims=False)
            - result @ jnp.conj(row)
        )
        diagonal = jnp.sqrt(jnp.real(column_residual[j]))
        column = column_residual / diagonal.astype(a.dtype)
        column = jnp.where(indices > j, column, jnp.zeros_like(column))
        column = column.at[j].set(diagonal.astype(a.dtype))
        return result.at[:, j].set(column)

    return lax.fori_loop(0, n, factor_column, result)


def _cholesky(a):
    return _vmap_batch(_cholesky_2d, a, a.ndim - 2)


def _left_triangular_solve_2d(a, b, *, lower, unit_diagonal):
    n = a.shape[0]
    if n == 0:
        return b
    result = jnp.zeros_like(b)

    def solve_step(step, result):
        row_index = jnp.where(lower, step, n - 1 - step)
        row = lax.dynamic_index_in_dim(a, row_index, axis=0,
                                       keepdims=False)
        rhs = lax.dynamic_index_in_dim(b, row_index, axis=0,
                                       keepdims=False) - row @ result
        diagonal = jnp.where(
            unit_diagonal,
            jnp.array(1, dtype=a.dtype),
            row[row_index],
        )
        solution = rhs / diagonal
        return result.at[row_index, :].set(solution)

    return lax.fori_loop(0, n, solve_step, result)


def _triangular_solve_2d(
    a,
    b,
    *,
    left_side,
    lower,
    transpose_a,
    conjugate_a,
    unit_diagonal,
):
    if transpose_a:
        a = jnp.swapaxes(a, -1, -2)
        lower = not lower
    if conjugate_a:
        a = jnp.conj(a)
    if unit_diagonal:
        a = a.at[jnp.diag_indices(a.shape[0])].set(
            jnp.array(1, dtype=a.dtype))

    if left_side:
        return _left_triangular_solve_2d(
            a, b, lower=lower, unit_diagonal=unit_diagonal)
    return jnp.swapaxes(
        _left_triangular_solve_2d(
            jnp.swapaxes(a, -1, -2),
            jnp.swapaxes(b, -1, -2),
            lower=not lower,
            unit_diagonal=unit_diagonal,
        ),
        -1,
        -2,
    )


def _triangular_solve(
    a,
    b,
    *,
    left_side,
    lower,
    transpose_a,
    conjugate_a,
    unit_diagonal,
):
    fn = partial(
        _triangular_solve_2d,
        left_side=left_side,
        lower=lower,
        transpose_a=transpose_a,
        conjugate_a=conjugate_a,
        unit_diagonal=unit_diagonal,
    )
    for _ in range(a.ndim - 2):
        fn = jax.vmap(fn)
    return fn(a, b)


def _hermitian_from_triangle(a, *, lower):
    n = a.shape[0]
    row = jnp.arange(n)[:, None]
    column = jnp.arange(n)[None, :]
    mask = row >= column if lower else row <= column
    reflected = _adjoint(a)
    result = jnp.where(mask, a, reflected)
    diagonal = jnp.real(jnp.diag(result)).astype(result.dtype)
    return result.at[jnp.diag_indices(n)].set(diagonal)


def _eigh_2d(a, *, lower, sort_eigenvalues, subset_by_index):
    """Cyclic Jacobi eigensolver for real symmetric/Hermitian matrices."""
    n = a.shape[0]
    if n == 0:
        return a, jnp.empty((0,), dtype=jnp.real(a).dtype)

    matrix = _hermitian_from_triangle(a, lower=lower)
    vectors = jnp.eye(n, dtype=a.dtype)
    real_dtype = jnp.real(a).dtype
    eps = jnp.finfo(real_dtype).eps

    def rotate_pair(pair, state):
        matrix, vectors = state
        p = pair // n
        q = pair % n
        active_pair = p < q
        app = jnp.real(matrix[p, p])
        aqq = jnp.real(matrix[q, q])
        apq = matrix[p, q]
        magnitude = jnp.abs(apq)
        threshold = eps * jnp.sqrt(jnp.abs(app * aqq))
        active = active_pair & (magnitude > threshold)
        safe_magnitude = jnp.where(active, magnitude, jnp.array(1, real_dtype))
        phase = apq / safe_magnitude.astype(apq.dtype)
        tau = (aqq - app) / (2 * safe_magnitude)
        sign = jnp.where(tau >= 0, jnp.array(1, real_dtype),
                         jnp.array(-1, real_dtype))
        tangent = sign / (jnp.abs(tau) + jnp.sqrt(1 + tau * tau))
        tangent = jnp.where(active, tangent, jnp.array(0, real_dtype))
        cosine = lax.rsqrt(1 + tangent * tangent)
        sine = tangent * cosine
        c = cosine.astype(a.dtype)
        s = sine.astype(a.dtype)

        column_p = matrix[:, p]
        column_q = matrix[:, q]
        matrix = matrix.at[:, p].set(c * column_p - s * jnp.conj(phase) * column_q)
        matrix = matrix.at[:, q].set(s * phase * column_p + c * column_q)
        row_p = matrix[p, :]
        row_q = matrix[q, :]
        matrix = matrix.at[p, :].set(c * row_p - s * phase * row_q)
        matrix = matrix.at[q, :].set(s * jnp.conj(phase) * row_p + c * row_q)

        vector_p = vectors[:, p]
        vector_q = vectors[:, q]
        vectors = vectors.at[:, p].set(
            c * vector_p - s * jnp.conj(phase) * vector_q)
        vectors = vectors.at[:, q].set(s * phase * vector_p + c * vector_q)
        return matrix, vectors

    # A cyclic sweep touches each unordered pair once.  Jacobi converges
    # quadratically near a diagonal matrix; 30 sweeps is conservative for the
    # small and medium matrices used by JAX's QDWH SVD.
    def sweep(_, state):
        return lax.fori_loop(0, n * n, rotate_pair, state)

    matrix, vectors = lax.fori_loop(0, 30, sweep, (matrix, vectors))
    values = jnp.real(jnp.diag(matrix))
    if sort_eigenvalues or subset_by_index is not None:
        # Sort every eigenvector row with the same repeated eigenvalue keys.
        # A key/value sort avoids the batched gather emitted by argsort-based
        # indexing, which is both more work and less portable across backends.
        keys = jnp.broadcast_to(values, vectors.shape)
        keys, vectors = lax.sort(
            (keys, vectors), dimension=-1, is_stable=True, num_keys=1)
        values = keys[0]
    if subset_by_index is not None:
        start, end = subset_by_index
        values = values[start:end]
        vectors = vectors[:, start:end]
    return vectors, values


def _eigh(a, *, lower, sort_eigenvalues, subset_by_index):
    fn = partial(
        _eigh_2d,
        lower=lower,
        sort_eigenvalues=sort_eigenvalues,
        subset_by_index=subset_by_index,
    )
    for _ in range(a.ndim - 2):
        fn = jax.vmap(fn)
    return fn(a)


def initialize():
    """Registers the PJRT library and TT-specific lowering rules."""
    from jax._src import xla_bridge
    from jax._src.interpreters import mlir
    from jax._src.lax import linalg as lax_linalg

    library_path = os.environ.get("LIBTT_LIBRARY_PATH")
    if library_path is None:
        bundled_library = Path(__file__).with_name("libtt.so")
        if bundled_library.is_file():
            library_path = str(bundled_library)
    if library_path is None:
        return
    xla_bridge.register_plugin("tt", library_path=library_path)

    mlir.register_lowering(
        lax_linalg.qr_p,
        mlir.lower_fun(_qr, multiple_results=True),
        platform="tt",
    )
    mlir.register_lowering(
        lax_linalg.cholesky_p,
        mlir.lower_fun(_cholesky, multiple_results=False),
        platform="tt",
    )
    mlir.register_lowering(
        lax_linalg.triangular_solve_p,
        mlir.lower_fun(_triangular_solve, multiple_results=False),
        platform="tt",
    )
    mlir.register_lowering(
        lax_linalg.eigh_p,
        mlir.lower_fun(_eigh, multiple_results=True),
        platform="tt",
    )
