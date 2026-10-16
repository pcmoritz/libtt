"""Const-eval functions that differ only in op order share their cached results.

A const-eval function's results are cached under a hash of the function, so
that programs computing the same tensors from the same weights (a prefill and a
decode step) share them. Passes may order a function's independent ops either
way from one program to the next; with the order in the hash, two programs then
cached the same tensors under two keys and kept both. Qwen3-30B-A3B's MoE
expert weights were prepared twice that way, about 2.9 GB per chip on 4 chips.
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


def reversed_concat(x, a, b):
    """a_first with the concatenation's operands swapped: other results, so another hash."""
    a, b = weight(a), weight(b)
    scaled_a = jnp.exp(a)
    scaled_b = b * 2
    return x @ jnp.concatenate([scaled_b, scaled_a], axis=1)


def const_eval_hashes(path):
    """The const-eval function hashes (SHA-256, hex) in the program's flatbuffer."""
    (flatbuffer,) = path.glob("fb_*.ttnn")
    return set(re.findall(rb"(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])", flatbuffer.read_bytes()))


def const_eval_ops(path):
    (ir,) = (path / "irs").glob("ttnn_runtime_*.mlir")
    return re.findall(r"func\.func private @\w+_const_eval_\d+\(.*?\n\s*\} loc", ir.read_text(), re.S)


def exp_first(path):
    """Whether the const-eval function preparing the weights runs ttnn.exp before ttnn.multiply."""
    (function,) = [f for f in const_eval_ops(path) if "ttnn.exp" in f and "ttnn.multiply" in f]
    return function.index("ttnn.exp") < function.index("ttnn.multiply")


def run_program(fn, path, x, a, b):
    run = jax.jit(fn, compiler_options={"export_path": str(path)})
    return np.asarray(run(x, a, b)).astype(np.float32), const_eval_hashes(path)


def test_programs_share_results_whatever_the_op_order(tmp_path):
    device = jax.devices("tt")[0]
    rng = np.random.default_rng(0)
    x, a, b = (jax.device_put(rng.standard_normal(shape).astype(jnp.bfloat16), device)
               for shape in ((32, 64), (64, 64), (64, 64)))
    out_a, hashes_a = run_program(a_first, tmp_path / "a_first", x, a, b)
    out_b, hashes_b = run_program(b_first, tmp_path / "b_first", x, a, b)
    # The two const-eval functions really differ in op order; otherwise this
    # test would not exercise the order-free hash.
    assert exp_first(tmp_path / "a_first") != exp_first(tmp_path / "b_first")
    np.testing.assert_array_equal(out_a, out_b)
    assert hashes_a and hashes_a == hashes_b


def test_programs_with_other_results_do_not_share(tmp_path):
    """Sharing a cache entry between functions with different results would be
    the worst failure: swapping the concatenation's operands must change the hash."""
    device = jax.devices("tt")[0]
    rng = np.random.default_rng(0)
    x, a, b = (jax.device_put(rng.standard_normal(shape).astype(jnp.bfloat16), device)
               for shape in ((32, 64), (64, 64), (64, 64)))
    out_a, hashes_a = run_program(a_first, tmp_path / "a_first", x, a, b)
    out_r, hashes_r = run_program(reversed_concat, tmp_path / "reversed", x, a, b)
    assert not np.array_equal(out_a, out_r)
    assert hashes_a and hashes_r and hashes_a != hashes_r
