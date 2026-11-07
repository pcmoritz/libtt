"""A float8 weight argument is read in place by the trace that uses it, as
torch_tpu passes FP8 checkpoints.

The compiler stores float8 tensors as their bytes (Float8DataTypeConversion),
so the argument becomes a uint8 tensor, and its float8 matrix-unit operand a
view of those bytes. A loop (which the host runs) splits the program: the
matmul after it is in a trace of its own. The view must be in that trace, next
to the matmul that reads it: a value passed into a trace is copied into a slot
of that trace on every call, which for a model's weights takes gigabytes per
program.
"""

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np


def test_float8_weight_stays_with_its_trace(tmp_path):
    def program(weight, x, start):
        start = jax.lax.while_loop(lambda s: s > 32, lambda s: s - 32, start)
        rows = jax.lax.dynamic_slice(x, (start, 0), (32, 256)).astype(jnp.float8_e4m3fn)
        return jnp.dot(rows, weight, preferred_element_type=jnp.float32)

    options = {"optimization_level": "O0", "enable_trace": "true", "export_path": str(tmp_path)}
    run = jax.jit(program, compiler_options=options)
    device = jax.devices("tt")[0]
    rng = np.random.default_rng(0)
    weight = rng.normal(0, 1, (256, 128)).astype(ml_dtypes.float8_e4m3fn)
    x = rng.normal(0, 1, (64, 256)).astype(jnp.bfloat16)
    args = [jax.device_put(a, device) for a in (weight, x)]
    rows = x[16:48].astype(ml_dtypes.float8_e4m3fn).astype(np.float32)
    expected = rows @ weight.astype(np.float32)
    for _ in range(3):  # Warmup, capture, replay.
        out = np.asarray(run(*args, jax.device_put(np.int32(16), device)))
        # The matrix unit adds in its own precision; within 2% of the output scale.
        np.testing.assert_allclose(out, expected, atol=0.02 * np.abs(expected).max())

    irs = [path.read_text() for path in (tmp_path / "irs").glob("ttnn_[0-9]*.mlir")]
    assert any("while" in ir for ir in irs), "the program has no loop"
    for ir in irs:
        for line in ir.splitlines():
            if '"ttnn.copy"' in line or '"ttnn.write_tensor"' in line:
                assert "f8E4M3FN" not in line, f"the weight is staged into a trace slot: {line.strip()}"
