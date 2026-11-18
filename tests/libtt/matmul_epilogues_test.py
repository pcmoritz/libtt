"""ttir-matmul-epilogues keeps a fused matmul where the matmul was."""

import jax
import jax.numpy as jnp
import numpy as np


def _cache_update_args(rng):
    """A paged KV cache and one decode step's update of 16 tokens."""
    cache = np.asarray(rng.normal(size=(17, 8, 32, 128)), dtype=jnp.bfloat16)
    tokens = 16
    values = np.asarray(rng.normal(size=(1, tokens, 8, 128)), dtype=jnp.bfloat16)
    positions = np.arange(tokens, dtype=np.int32) + 24
    pages = np.tile(np.arange(4, dtype=np.int32)[::-1], (tokens, 1))
    return cache, values, positions, pages


def _update(cache, values, positions, pages):
    """The cache updated in place."""
    return jax.ffi.ffi_call(
        "tt.paged_update_cache",
        jax.ShapeDtypeStruct(cache.shape, cache.dtype),
        input_output_aliases={0: 0},
    )(cache, values, positions, pages, share_cache=True)


def _ttnn_ir(tmp_path):
    (ir,) = [path.read_text() for path in (tmp_path / "irs").glob("ttnn_[0-9]*.mlir")]
    return ir


def test_activation_after_a_cache_update(tmp_path):
    """p = cache @ w, an in-place update of the cache, then silu(p): the matmul
    with the fused SiLU reads the cache before the update, as the program
    does."""
    device = jax.devices("tt")[0]
    rng = np.random.default_rng(0)
    cache, values, positions, pages = _cache_update_args(rng)
    # A weight per page and head: a batched matmul that reads the cache itself.
    w = np.asarray(rng.normal(0, 0.1, size=(17, 8, 128, 64)), dtype=jnp.bfloat16)

    def step(cache, values, positions, pages, w):
        p = jnp.matmul(cache, w)
        return _update(cache, values, positions, pages), jax.nn.silu(p)

    want = jax.jit(lambda c, w: jax.nn.silu(jnp.matmul(c, w)))(jax.device_put(cache, device), jax.device_put(w, device))
    run = jax.jit(step, donate_argnums=(0,), compiler_options={"export_path": str(tmp_path)})
    _, got = run(*[jax.device_put(x, device) for x in (cache, values, positions, pages, w)])
    np.testing.assert_array_equal(np.asarray(got, np.float32), np.asarray(want, np.float32))

    lines = _ttnn_ir(tmp_path).splitlines()
    matmul = [i for i, line in enumerate(lines) if '"ttnn.matmul"' in line and 'activation = "silu"' in line]
    update = [i for i, line in enumerate(lines) if '"ttnn.paged_update_cache"' in line]
    assert matmul and update, "expected a matmul with a fused SiLU and a cache update"
    assert matmul[0] < update[0], "the fused matmul runs after the cache update"


def test_scales_after_a_cache_update(tmp_path):
    """An FP8 linear, (x @ w) * row_scale * col_scale in FP32, with an in-place
    cache update between the matmul and the scale multiplies: a scaled matmul
    reads its scales where the matmul was, so the ops stay apart."""
    device = jax.devices("tt")[0]
    rng = np.random.default_rng(1)
    x = np.asarray(rng.normal(0, 1, (1, 512)), dtype=jnp.float8_e4m3fn)
    w = np.asarray(rng.normal(0, 30, (512, 256)), dtype=jnp.float8_e4m3fn)
    row = rng.uniform(0.5, 1, (1, 1)).astype(np.float32)
    col = rng.uniform(1e-4, 1e-3, (1, 256)).astype(np.float32)
    cache, values, positions, pages = _cache_update_args(rng)

    def product(x, w):
        return jax.lax.dot_general(x, w, (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32)

    def step(x, w, row, col, cache, values, positions, pages):
        acc = product(x, w)
        cache = _update(cache, values, positions, pages)
        return cache, (acc * row * col).astype(jnp.bfloat16)

    args = [jax.device_put(a, device) for a in (x, w, row, col)]
    want = jax.jit(lambda x, w, row, col: (product(x, w) * row * col).astype(jnp.bfloat16))(*args)
    run = jax.jit(step, donate_argnums=(4,), compiler_options={"export_path": str(tmp_path)})
    _, got = run(*args, *[jax.device_put(a, device) for a in (cache, values, positions, pages)])
    np.testing.assert_array_equal(np.asarray(got, np.float32), np.asarray(want, np.float32))
    assert '"ttnn.scaled_matmul"' not in _ttnn_ir(tmp_path), "the scaled matmul formed across the cache update"
