"""ttir-matmul-epilogues keeps a fused matmul where the matmul was."""

import jax
import jax.numpy as jnp
import numpy as np


def test_activation_after_a_cache_update(tmp_path):
    """p = cache @ w, an in-place update of the cache, then silu(p): the matmul
    with the fused SiLU reads the cache before the update, as the program
    does."""
    device = jax.devices("tt")[0]
    rng = np.random.default_rng(0)
    cache = np.asarray(rng.normal(size=(17, 8, 32, 128)), dtype=jnp.bfloat16)
    # A weight per page and head: a batched matmul that reads the cache itself.
    w = np.asarray(rng.normal(0, 0.1, size=(17, 8, 128, 64)), dtype=jnp.bfloat16)
    tokens = 16
    values = np.asarray(rng.normal(size=(1, tokens, 8, 128)), dtype=jnp.bfloat16)
    positions = np.arange(tokens, dtype=np.int32) + 24
    pages = np.tile(np.arange(4, dtype=np.int32)[::-1], (tokens, 1))

    def step(cache, values, positions, pages, w):
        p = jnp.matmul(cache, w)
        cache = jax.ffi.ffi_call(
            "tt.paged_update_cache",
            jax.ShapeDtypeStruct(cache.shape, cache.dtype),
            input_output_aliases={0: 0},
        )(cache, values, positions, pages, share_cache=True)
        return cache, jax.nn.silu(p)

    want = jax.jit(lambda c, w: jax.nn.silu(jnp.matmul(c, w)))(jax.device_put(cache, device), jax.device_put(w, device))
    run = jax.jit(step, donate_argnums=(0,), compiler_options={"export_path": str(tmp_path)})
    _, got = run(*[jax.device_put(x, device) for x in (cache, values, positions, pages, w)])
    np.testing.assert_array_equal(np.asarray(got, np.float32), np.asarray(want, np.float32))

    (ir,) = [path.read_text() for path in (tmp_path / "irs").glob("ttnn_[0-9]*.mlir")]
    lines = ir.splitlines()
    matmul = [i for i, line in enumerate(lines) if '"ttnn.matmul"' in line and 'activation = "silu"' in line]
    update = [i for i, line in enumerate(lines) if '"ttnn.paged_update_cache"' in line]
    assert matmul and update, "expected a matmul with a fused SiLU and a cache update"
    assert matmul[0] < update[0], "the fused matmul runs after the cache update"
