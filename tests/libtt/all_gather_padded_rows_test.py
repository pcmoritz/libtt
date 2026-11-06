"""All-gathers of tensors padded to whole tiles.

tt-metal gathers a tensor padded on the gathered dimension (one row per
device, padded to a 32-row tile) by broadcasting it to every device and
concatenating. It now broadcasts the logical rows and columns, row-major,
instead of whole tiles: a [1, 37984] row of logits (Qwen3's vocabulary on four
devices) moved 2.4 MB of tiles per device for 74 KB of values.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


def mesh():
    devices = jax.devices("tt")
    if len(devices) < 2:
        pytest.skip("needs at least two devices")
    return Mesh(np.array(devices), ("x",))


@pytest.mark.parametrize(
    "dtype,shape,axis",
    [
        (jnp.bfloat16, (1, 37984), 0),
        (jnp.float32, (1, 37984), 0),
        (jnp.bfloat16, (3, 100), 0),
        (jnp.int32, (1, 64), 0),
        (jnp.bfloat16, (4, 50), 1),
    ],
)
def test_all_gather_padded(dtype, shape, axis):
    m = mesh()
    n = len(m.devices)
    full = list(shape)
    full[axis] *= n
    x = np.random.default_rng(0).normal(0, 100, full).astype(dtype)
    spec = P("x", None) if axis == 0 else P(None, "x")
    gather = jax.shard_map(
        lambda b: jax.lax.all_gather(b, "x", axis=axis, tiled=True),
        mesh=m,
        in_specs=spec,
        out_specs=P(),
        check_vma=False,
    )
    out = jax.jit(gather)(jax.device_put(x, NamedSharding(m, spec)))
    np.testing.assert_array_equal(np.asarray(out), x)
