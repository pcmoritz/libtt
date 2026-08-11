import jax
from jax import lax
import jax.numpy as jnp
import numpy as np


def test_fp32_uint32_roundtrip_is_bit_exact():
    values = np.array(
        [-5.995684, -0.0, 0.0, 0.1, 1.0000001, 12345.678],
        dtype=np.float32,
    )
    roundtrip = jax.jit(
        lambda x: lax.bitcast_convert_type(
            lax.bitcast_convert_type(x, jnp.uint32), jnp.float32
        )
    )(values)

    np.testing.assert_array_equal(
        np.asarray(roundtrip).view(np.uint32), values.view(np.uint32)
    )


def test_uint32_fp32_roundtrip_is_bit_exact():
    values = np.array(
        [
            0x00000000,
            0x80000000,
            0x3F800001,
            0x3F812345,
            0x7F800000,
            0xFF800000,
            0x7FC12345,
            0xBF812345,
        ],
        dtype=np.uint32,
    )
    roundtrip = jax.jit(
        lambda x: lax.bitcast_convert_type(
            lax.bitcast_convert_type(x, jnp.float32), jnp.uint32
        )
    )(values)

    np.testing.assert_array_equal(np.asarray(roundtrip), values)
