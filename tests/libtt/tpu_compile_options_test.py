"""libtpu's xla_tpu_* compile options, which torch_tpu passes through, are
accepted and ignored; other unknown options are still rejected."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def run(options):
    device = jax.devices("tt")[0]
    add = jax.jit(lambda x: x + 1, compiler_options=options)
    return np.asarray(add(jax.device_put(jnp.arange(4, dtype=jnp.float32), device)))


@pytest.mark.parametrize("value", ["AUTO", "ENABLED", "DISABLED"])
def test_xla_tpu_option_is_ignored(value):
    np.testing.assert_array_equal(run({"xla_tpu_enable_deduplicated_calls": value}), [1, 2, 3, 4])


def test_unknown_option_is_rejected():
    with pytest.raises(Exception, match="No such compile option: 'not_a_compile_option'"):
        run({"not_a_compile_option": "1"})
