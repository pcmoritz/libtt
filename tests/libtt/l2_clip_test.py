"""tt.l2_clip: Inkling 2's per-head L2 clips (sglang_tpu_kernels.patch_inkling2_l2_clip's call) run as one
ttnn.l2_clip and match the math in float64 within a BF16 rounding: v = x * scale, then each group of columns
times min(1, threshold / sqrt(sum(v^2) + eps^2)), times a per-row scale."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

EPS = 1e-6


def l2_clip(x, scale, row_scale, *, group, threshold):
    operands = [x, *([scale] if scale is not None else []), *([row_scale] if row_scale is not None else [])]
    return jax.ffi.ffi_call("tt.l2_clip", jax.ShapeDtypeStruct(x.shape, x.dtype))(
        *operands,
        group=np.int64(group),
        offset=np.int64(0),
        width=np.int64(x.shape[-1]),
        threshold=np.float32(threshold),
        epsilon=np.float32(EPS),
        has_scale=scale is not None,
        has_row_scale=row_scale is not None,
    )


def reference(x, scale, row_scale, group, threshold):
    v = np.asarray(x, np.float64) * (1 if scale is None else np.asarray(scale, np.float64))
    rows, width = v.shape
    g = v.reshape(rows, width // group, group)
    f = np.minimum(1, threshold / np.sqrt((g * g).sum(-1, keepdims=True) + EPS**2))
    out = (g * f).reshape(rows, width)
    return out if row_scale is None else out * np.asarray(row_scale, np.float64)


@pytest.mark.parametrize("trace", [False, True])
@pytest.mark.parametrize(
    "rows,width,group,scaled,row_scaled",
    [
        # Inkling-2-Mini per chip at TP4: queries (6 heads of 128, scaled, tau per token), keys (2 heads), relative
        # projections (6 heads of 16); one and four tokens; a 64-token prefill.
        (1, 768, 128, True, True),
        (1, 256, 128, False, False),
        (4, 96, 16, True, True),
        (64, 768, 128, True, True),
    ],
)
def test_l2_clip(trace, rows, width, group, scaled, row_scaled, tmp_path):
    threshold = 1.05 * np.sqrt(group)
    run = jax.jit(
        lambda x, s, r: l2_clip(x, s, r, group=group, threshold=threshold),
        compiler_options={"enable_trace": str(trace).lower(), "export_path": str(tmp_path)},
    )
    device = jax.devices("tt")[0]
    rng = np.random.default_rng(rows * 7 + group)
    x = jnp.asarray(rng.normal(0, 1, (rows, width)), jnp.bfloat16)
    s = jnp.asarray(rng.normal(1, 0.3, width), jnp.float32) if scaled else None
    r = jnp.asarray(1 + 0.2 * np.abs(rng.normal(0, 1, (rows, 1))), jnp.float32) if row_scaled else None
    put = lambda a: None if a is None else jax.device_put(a, device)  # noqa: E731
    got = np.asarray(run(put(x), put(s), put(r)), np.float64)
    np.testing.assert_allclose(got, reference(x, s, r, group, threshold), rtol=2**-7, atol=1e-6)
    irs = [path.read_text() for path in (tmp_path / "irs").glob("ttnn_*.mlir")]
    assert irs and any('"ttnn.l2_clip"' in ir for ir in irs)
