"""Const-eval functions that differ only in op order share their cached results.

A const-eval function's results are cached under a hash of the function, so
that programs computing the same tensors from the same weights (a prefill and a
decode step) share them. Passes may order a function's independent ops either
way from one program to the next; with the order in the hash, two programs then
cached the same tensors under two keys and kept both. Replicated MoE experts
were prepared twice that way and ran out of device memory.
"""

import re

import jax
import jax.numpy as jnp
import numpy as np


def weight(w):
    """Marks w as a weight, prepared once rather than per call."""
    return jax.ffi.ffi_call("tt.weight_dtype_override", jax.ShapeDtypeStruct(w.shape, w.dtype))(
        w, **{"ttcore.weight_dtype": "bf16"}
    )


def a_first(x, a, b):
    a, b = weight(a), weight(b)
    scaled_a = jnp.exp(a)
    scaled_b = b * 2
    return x @ jnp.concatenate([scaled_a, scaled_b], axis=1)


def b_first(x, a, b):
    a, b = weight(a), weight(b)
    scaled_b = b * 2
    scaled_a = jnp.exp(a)
    return x @ jnp.concatenate([scaled_a, scaled_b], axis=1)


def const_eval_hashes(path):
    """The const-eval function hashes (SHA-256, hex) in the program's flatbuffer."""
    (flatbuffer,) = path.glob("fb_*.ttnn")
    return set(re.findall(rb"(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])", flatbuffer.read_bytes()))


def const_eval_ops(path):
    (ir,) = (path / "irs").glob("ttnn_runtime_*.mlir")
    return re.findall(r"func\.func private @\w+_const_eval_\d+\(.*?\n\s*\} loc", ir.read_text(), re.S)


def test_programs_share_results_whatever_the_op_order(tmp_path):
    device = jax.devices("tt")[0]
    rng = np.random.default_rng(0)
    x, a, b = (jax.device_put(rng.standard_normal(shape).astype(jnp.bfloat16), device)
               for shape in ((32, 64), (64, 64), (64, 64)))
    outputs, hashes = [], []
    for name, fn in (("a_first", a_first), ("b_first", b_first)):
        run = jax.jit(fn, compiler_options={"export_path": str(tmp_path / name)})
        outputs.append(np.asarray(run(x, a, b)).astype(np.float32))
        functions = const_eval_ops(tmp_path / name)
        assert any("ttnn.exp" in f and "ttnn.multiply" in f for f in functions), functions
        hashes.append(const_eval_hashes(tmp_path / name))
    np.testing.assert_array_equal(outputs[0], outputs[1])
    assert hashes[0] and hashes[0] == hashes[1]
