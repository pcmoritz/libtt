"""TorchTitan's gated RMSNorm (Qwen3.5's GatedDeltaNet output norm) without Pallas.

In TPU compatibility mode, the forward and backward Pallas kernels behind
TorchTitan's `rms_norm_gated` run as JAX (`jax_plugins.libtt.torchtitan_tpu_kernels`).
"""

import jax
import ml_dtypes
import numpy as np

from jax_plugins.libtt import torchtitan_tpu_kernels

ROWS, DIM, EPS = 256, 128, 1e-6


def _relative_error(got, want):
    return np.max(np.abs(np.asarray(got).astype(np.float64) - want)) / np.max(np.abs(want))


def test_forward_and_backward():
    rng = np.random.default_rng(0)
    x, gate, dy = (rng.standard_normal((ROWS, DIM)).astype(ml_dtypes.bfloat16) for _ in range(3))
    w = rng.standard_normal((1, DIM)).astype(np.float32)
    xf, gf, dyf, wf = (t.astype(np.float64) for t in (x, gate, dy, w))
    rstd = 1 / np.sqrt(np.mean(xf * xf, axis=-1, keepdims=True) + EPS)
    normed = xf * rstd
    sigmoid = 1 / (1 + np.exp(-gf))
    dscaled = dyf * gf * sigmoid
    dnormed = dscaled * wf

    out = jax.jit(torchtitan_tpu_kernels.rms_norm_gated_fwd, static_argnums=3)(x, gate, w, EPS)
    assert out.dtype == x.dtype
    assert _relative_error(out, wf * normed * gf * sigmoid) < 2e-2
    dx, dgate, dw = jax.jit(torchtitan_tpu_kernels.rms_norm_gated_bwd, static_argnums=4)(x, gate, w, dy, EPS)
    assert (dx.dtype, dgate.dtype, dw.dtype, dw.shape) == (x.dtype, gate.dtype, np.float32, (1, DIM))
    assert _relative_error(dx, rstd * (dnormed - normed * np.mean(dnormed * normed, axis=-1, keepdims=True))) < 2e-2
    assert _relative_error(dgate, dyf * wf * normed * sigmoid * (1 + gf * (1 - sigmoid))) < 2e-2
    assert _relative_error(dw, np.sum(dscaled * normed, axis=0, keepdims=True)) < 2e-2
