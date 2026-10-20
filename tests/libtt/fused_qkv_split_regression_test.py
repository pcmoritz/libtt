"""TorchTitan's fused QKV projection splits [S, G * R * D] into [S, G, R, D]
and slices R into the query, key and value heads; the backward pass
concatenates them again. Tiles would pad R (or G) there, so the compiler keeps
these tensors as [S * G, R * D] rows: the slices and the concatenation run on
columns. Only data moves, so the results are exact."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

S, G, R, D = 64, 8, 4, 128


def split(qkv):
    heads = qkv.reshape(S, G, R, D)
    q = heads[:, :, : R - 2].reshape(S, G * (R - 2), D)
    k = heads[:, :, R - 2]
    v = heads[:, :, R - 1]
    return q, k, v


@pytest.mark.parametrize("dtype", [jnp.bfloat16, np.float32])
def test_split_and_join_fused_qkv(dtype):
    x = (np.arange(S * G * R * D) % 251 - 125).astype(dtype).reshape(S, G * R * D)

    def run(x):
        q, k, v = split(x * 2)
        joined = jnp.concatenate(
            [(q + 1).reshape(S, G, R - 2, D), (k + 2)[:, :, None], (v + 3)[:, :, None]], axis=2
        )
        return q, k, v, joined.reshape(S, G * R * D)

    got = jax.jit(run)(jax.device_put(x, jax.devices("tt")[0]))
    want = run(np.asarray(x))
    for g, w in zip(got, want):
        np.testing.assert_array_equal(np.asarray(g), np.asarray(w))
