"""Wide tiled argmax must preserve indices when using row-major reduction."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize(
    "shape", [(4096,), (4, 4097), (2, 3, 4097), (1, 248320), (32, 248320), (33, 4096)]
)
@pytest.mark.parametrize("trace", [False, True])
def test_wide_bf16_argmax(shape, trace):
    tt, cpu = jax.devices("tt")[0], jax.devices("cpu")[0]

    @partial(
        jax.jit,
        compiler_options={
            "optimization_level": "1",
            "enable_trace": str(trace).lower(),
        },
    )
    def choose(x, scale):
        # A dynamic producer gives argmax a tiled input, as in model decode.
        values = x * scale
        return values, jnp.argmax(values, axis=-1, keepdims=True)

    with jax.default_device(cpu):
        scale = jnp.asarray(1, dtype=jnp.bfloat16)
    scale = jax.device_put(scale, tt)
    for seed in range(3):
        values = np.random.default_rng(seed).normal(size=shape).astype(np.float32)
        rows = values.reshape(-1, shape[-1])
        # Equal maxima cross tile and core boundaries; the first index wins.
        rows[0, 31 + seed] = rows[0, -1] = 10
        if len(rows) >= 4:
            rows[1, shape[-1] // 2 + seed] = rows[1, -1] = np.nan
            rows[2] = -np.inf
            rows[3, 63 + seed] = rows[3, -1] = np.inf
        if len(rows) >= 32:
            rows[4] = 0
            rows[5] = np.nan
        with jax.default_device(cpu):
            values = jnp.asarray(values, dtype=jnp.bfloat16)
        produced, indices = choose(jax.device_put(values, tt), scale)
        expected = np.argmax(
            np.asarray(produced).astype(np.float32), axis=-1, keepdims=True
        )
        np.testing.assert_array_equal(np.asarray(indices), expected)


def test_wide_fp32_argmax_preserves_close_maxima():
    values = np.full((4, 4097), -100, dtype=np.float32)
    values[:, 0] = 1
    values[:, -1] = 1 + 2**-20
    tt = jax.devices("tt")[0]

    @partial(
        jax.jit, compiler_options={"optimization_level": "1", "enable_trace": "true"}
    )
    def choose(x, scale):
        return jnp.argmax(x * scale, axis=-1)

    actual = choose(jax.device_put(values, tt), jax.device_put(np.float32(1), tt))
    np.testing.assert_array_equal(np.asarray(actual), np.full(4, 4096))
