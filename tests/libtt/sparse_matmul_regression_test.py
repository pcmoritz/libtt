"""tt.sparse_matmul computes mixture-of-experts layers at dense-matmul precision.

Without a compute config, TTNN runs any matmul with a program config at LoFi,
and sparse matmuls always carry one: experts were off by about 3% of their
range. With tt-mlir's HiFi4 and FP32 accumulation they were still off by 1.7%
at K=2048, since the sparse factory reloaded its FP32 partials through SrcA,
rounding them to TF32 after every K block. Expert pairs the sparsity
skips must come back as zeros, since MoE layers sum over every expert.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.xla_metadata import set_xla_metadata


def sparse_matmul(a, b, sparsity, shape, *, a_sparse, b_sparse):
    with set_xla_metadata(
        is_input_a_sparse=str(a_sparse).lower(), is_input_b_sparse=str(b_sparse).lower(), nnz="0"
    ):
        return jax.ffi.ffi_call("tt.sparse_matmul", jax.ShapeDtypeStruct(shape, a.dtype))(
            a, b, sparsity
        )


def bf16(rng, shape, scale):
    return (rng.standard_normal(shape) * scale).astype(jnp.bfloat16)


def as_f64(x):
    return np.asarray(x).astype(np.float64)


def assert_close_where_active(out, want, active):
    """Active pairs match a float64 reference to within 0.5% of their range,
    about twice bf16's rounding; inactive pairs are exactly zero."""
    for index in zip(*np.nonzero(active)):
        error = np.abs(out[index] - want[index]).max() / np.abs(want[index]).max()
        assert error < 0.005, (index, error)
    assert not out[~active.astype(bool)].any()


@pytest.mark.parametrize("blocks,tile,experts,k,n", [(2, 32, 8, 1024, 128), (1, 32, 16, 2048, 64)])
def test_expert_weights_sparse(blocks, tile, experts, k, n):
    """Token tiles [1, B, M, K] times experts [1, E, K, N]: the gate and up
    projections."""
    rng = np.random.default_rng(blocks * experts)
    a, b = bf16(rng, (1, blocks, tile, k), 1.0), bf16(rng, (1, experts, k, n), k**-0.5)
    active = (rng.random((blocks, experts)) < 0.5).astype(np.float32)
    active[0, 0] = 1
    sparsity = active.reshape(1, blocks, 1, experts).astype(jnp.bfloat16)
    shape = (1, blocks, 1, experts, tile, n)
    f = jax.jit(lambda a, b, s: sparse_matmul(a, b, s, shape, a_sparse=False, b_sparse=True))
    out = as_f64(f(*jax.device_put((a, b, sparsity), jax.devices("tt")[0])))[0, :, 0]
    want = np.einsum("bmk,ekn->bemn", as_f64(a)[0], as_f64(b)[0])
    assert_close_where_active(out, want, active)


@pytest.mark.parametrize("blocks,tile,experts,k,n", [(2, 32, 8, 768, 512), (4, 32, 4, 1536, 256)])
def test_expert_inputs_sparse(blocks, tile, experts, k, n):
    """Per-expert activations [A, E, M, K] times experts [1, E, K, N]: the down
    projection."""
    rng = np.random.default_rng(blocks * experts + 1)
    a, b = bf16(rng, (blocks, experts, tile, k), 1.0), bf16(rng, (1, experts, k, n), k**-0.5)
    active = (rng.random((blocks, experts)) < 0.5).astype(np.float32)
    active[0, 0] = 1
    sparsity = active.reshape(1, 1, blocks, experts).astype(jnp.bfloat16)
    shape = (blocks, experts, tile, n)
    f = jax.jit(lambda a, b, s: sparse_matmul(a, b, s, shape, a_sparse=True, b_sparse=False))
    out = as_f64(f(*jax.device_put((a, b, sparsity), jax.devices("tt")[0])))
    want = np.einsum("aemk,ekn->aemn", as_f64(a), as_f64(b)[0])
    assert_close_where_active(out, want, active)
