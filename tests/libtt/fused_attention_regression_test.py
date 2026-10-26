"""jax.nn.dot_product_attention runs tt-train's fused attention kernels.

Supported calls lower to ttnn.sdpa_fw and, under a gradient, ttnn.sdpa_bw, so
no [B, N, T, S] scores reach device memory; other calls take JAX's own
implementation. Both match JAX's FP32 result on the CPU.
"""

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

B, T, N, D = 2, 128, 4, 64


def ops(ir):
    main = ir[ir.index("func.func public @main") :]
    return re.findall(r'"(ttnn\.[a-z_0-9]+)"\(', main)


def run(fn, tmp_path, args, level="O0"):
    compiled = jax.jit(fn, compiler_options={"optimization_level": level, "export_path": str(tmp_path)})
    device = jax.devices("tt")[0]
    out = compiled(*(jax.device_put(x, device) for x in args))
    (ir,) = [path.read_text() for path in (tmp_path / "irs").glob("ttnn_[0-9]*.mlir")]
    cpu = jax.devices("cpu")[0]
    want = jax.jit(fn)(*(jax.device_put(np.asarray(x, np.float32), cpu) for x in args))
    for a, w in zip(jax.tree.leaves(out), jax.tree.leaves(want)):
        a, w = np.asarray(a, np.float32), np.asarray(w, np.float32)
        np.testing.assert_allclose(a, w, atol=0.02 * np.abs(w).max())
    return ops(ir)


def inputs(kv_heads, dtype=jnp.bfloat16):
    rng = np.random.default_rng(0)
    shapes = [(B, T, N, D), (B, T, kv_heads, D), (B, T, kv_heads, D)]
    return [(rng.standard_normal(shape, np.float32) * 0.5).astype(dtype) for shape in shapes]


def grads(fn):
    return jax.grad(lambda *args: jnp.sum(fn(*args).astype(jnp.float32) ** 2), argnums=(0, 1, 2))


@pytest.mark.parametrize("is_causal", [False, True], ids=["full", "causal"])
@pytest.mark.parametrize("kv_heads", [N, 2], ids=["mha", "gqa"])
def test_forward_and_backward_use_the_fused_kernels(is_causal, kv_heads, tmp_path):
    attention = lambda q, k, v: jax.nn.dot_product_attention(q, k, v, is_causal=is_causal)
    forward = run(attention, tmp_path / "forward", inputs(kv_heads))
    assert "ttnn.sdpa_fw" in forward and "ttnn.matmul" not in forward
    backward = run(grads(attention), tmp_path / "backward", inputs(kv_heads))
    assert "ttnn.sdpa_bw" in backward and "ttnn.matmul" not in backward


def test_optimized_backward_uses_the_fused_kernels(tmp_path):
    attention = lambda q, k, v: jax.nn.dot_product_attention(q, k, v, is_causal=True)
    backward = run(grads(attention), tmp_path, inputs(N), level="O1")
    assert "ttnn.sdpa_fw" in backward and "ttnn.sdpa_bw" in backward and "ttnn.matmul" not in backward


def test_vmapped_attention_folds_into_the_batch(tmp_path):
    attention = jax.vmap(lambda q, k, v: jax.nn.dot_product_attention(q, k, v, is_causal=True))
    args = [np.stack([x, -x]) for x in inputs(N)]
    assert "ttnn.sdpa_fw" in run(attention, tmp_path, args)


@pytest.mark.parametrize(
    "kwargs, dtype",
    [({"implementation": "xla"}, jnp.bfloat16), ({}, jnp.float32), ({"scale": 0.5}, jnp.bfloat16)],
    ids=["xla", "float32", "scale"],
)
def test_other_calls_take_jax_implementation(kwargs, dtype, tmp_path):
    attention = lambda q, k, v: jax.nn.dot_product_attention(q, k, v, is_causal=True, **kwargs)
    forward = run(attention, tmp_path, inputs(N, dtype))
    assert "ttnn.sdpa_fw" not in forward and "ttnn.matmul" in forward
