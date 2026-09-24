"""Numerical and trace-replay checks for decode projections with BF8 weights."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

@pytest.mark.parametrize('rows,inner,width', [(1, 2048, 4096), (1, 4096, 3072), (1, 6144, 4096), (2, 6144, 4096)])
@pytest.mark.parametrize('trace', [False, True])
def test_decode_dram_matmul(rows, inner, width, trace):
    def project(x, w):
        w = w[None]
        w = jax.ffi.ffi_call('tt.weight_dtype_override', jax.ShapeDtypeStruct(w.shape, w.dtype),
                              vmap_method='sequential')(w, **{'ttcore.weight_dtype': 'bfp_bf8'})[0]
        return x @ w
    run = jax.jit(project, compiler_options={'optimization_level': 'O1', 'enable_trace': str(trace).lower()})
    device = jax.devices('tt')[0]
    rng = np.random.default_rng(113)
    for iteration in range(3):
        # Exact BF8 inputs isolate accumulation and stale-layout-cache errors.
        x = (rng.integers(-4, 5, (rows, inner)) / 16).astype(jnp.bfloat16)
        w = (rng.integers(-4, 5, (inner, width)) / 16).astype(jnp.bfloat16)
        if iteration == 0:
            x.fill(0.125)
            w.fill(0.125)
        expected = x.astype(np.float32) @ w.astype(np.float32)
        actual = np.asarray(run(jax.device_put(x, device), jax.device_put(w, device))).astype(np.float32)
        np.testing.assert_allclose(actual, expected, atol=0.015625, rtol=0.01)
