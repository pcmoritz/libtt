"""jax.nn.dot_product_attention runs without padded size-1 rows, and errors carry their cause.

With one query head per key head, dot_product_attention computes in shapes
like [B, N, T, 1, S]. TTNN pads a size-1 second-to-last dim to a whole tile, so
such a tensor takes 32 times the memory and time of its [B, N, T, S] view; a
training step at S = 512 ran out of device memory. A runtime failure such as
that one reaches Python with its message rather than a bare status code.
"""

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

B, T, N, H = 2, 128, 4, 64


# JAX's implementation, which supported calls on the tt backend otherwise
# leave for the fused kernels (fused_attention_regression_test.py).
def attention(q, k, v):
    return jax.nn.dot_product_attention(q, k, v, is_causal=True, implementation="xla")


def attention_grads(q, k, v):
    loss = lambda *args: jnp.sum(attention(*args).astype(jnp.float32))
    return jax.grad(loss, argnums=(0, 1, 2))(q, k, v)


def padded_rows(ir):
    """The tensors of the main function whose second-to-last dim is 1 in more than one row of tiles."""
    main = ir[ir.index("func.func public @main") :]
    shapes = []
    for dims in re.findall(r"tensor<((?:[0-9]+x)+)[a-z]", main):
        dims = [int(d) for d in dims.rstrip("x").split("x")]
        if len(dims) >= 3 and dims[-2] == 1 and np.prod(dims[:-2]) > 1:
            shapes.append(dims)
    return shapes


@pytest.mark.parametrize("fn", [attention, attention_grads], ids=["forward", "backward"])
def test_dot_product_attention_has_no_padded_rows(fn, tmp_path):
    rng = np.random.default_rng(0)
    q, k, v = [(rng.standard_normal((B, T, N, H), np.float32) * 0.5).astype(jnp.bfloat16) for _ in range(3)]
    run = jax.jit(fn, compiler_options={"optimization_level": "O1", "export_path": str(tmp_path)})
    device = jax.devices("tt")[0]
    actual = run(*(jax.device_put(x, device) for x in (q, k, v)))

    (ir,) = [path.read_text() for path in (tmp_path / "irs").glob("ttnn_[0-9]*.mlir")]
    assert padded_rows(ir) == []

    cpu = jax.devices("cpu")[0]
    want = jax.jit(fn, device=cpu)(*(jax.device_put(x.astype(np.float32), cpu) for x in (q, k, v)))
    for a, w in zip(jax.tree.leaves(actual), jax.tree.leaves(want)):
        a, w = np.asarray(a, np.float32), np.asarray(w, np.float32)
        np.testing.assert_allclose(a, w, atol=0.02 * np.abs(w).max())


def test_out_of_memory_error_has_its_message():
    # Nine FP32 results of 4 GiB each, more than a device's DRAM.
    run = jax.jit(lambda x: [jnp.exp(jnp.broadcast_to(x + i, (16, 8192, 8192))) for i in range(9)])
    x = jax.device_put(np.float32(0.5), jax.devices("tt")[0])
    with pytest.raises(jax.errors.JaxRuntimeError, match="Out of Memory"):
        jax.block_until_ready(run(x))
