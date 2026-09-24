import os
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
print('Devices:', jax.devices(), flush=True)
mesh = Mesh(np.array(jax.devices()), ('tp',))
n = len(jax.devices())
assert n == int(os.environ.get("EXPECTED_DEVICES", "2")), f"Expected 2 devices, got {n}"
x = np.arange(n*32*32, dtype=np.float32).reshape(n*32,32)
a = jax.device_put(x, NamedSharding(mesh,P('tp',None)))
@jax.jit
@jax.shard_map(mesh=mesh, in_specs=P("tp",None), out_specs=P("tp",None), check_vma=False)
def f(a):
    return a + 1
np.testing.assert_allclose(np.asarray(f(a)), x+1)
@jax.jit
@jax.shard_map(mesh=mesh, in_specs=P('tp',None), out_specs=P(), check_vma=False)
def sum_shards(a):
    return jax.lax.psum(a, 'tp')
np.testing.assert_allclose(np.asarray(sum_shards(a)), x.reshape(n,32,32).sum(axis=0))
print('PASS sharded add and psum', flush=True)
