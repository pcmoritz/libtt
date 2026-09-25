"""Load executables back from JAX's persistent compilation cache."""

import jax
import jax.numpy as jnp
import numpy as np
from jax._src import compilation_cache


def test_persistent_cache_round_trip(tmp_path):
    options = {"optimization_level": "O1", "enable_trace": "true"}
    run = lambda: jax.jit(
        lambda x, w: jnp.tanh(x @ w) @ w.T, compiler_options=options
    )
    device = jax.devices("tt")[0]
    rng = np.random.default_rng(0)
    x = jax.device_put((rng.standard_normal((32, 256)) / 8).astype(jnp.bfloat16), device)
    w = jax.device_put((rng.standard_normal((256, 512)) / 16).astype(jnp.bfloat16), device)

    events = []
    listener = lambda event, **kwargs: events.append(event)
    settings = {
        "jax_compilation_cache_dir": str(tmp_path),
        "jax_persistent_cache_min_compile_time_secs": 0,
        "jax_persistent_cache_min_entry_size_bytes": 0,
    }
    previous = {name: getattr(jax.config, name) for name in settings}
    jax.monitoring.register_event_listener(listener)
    try:
        for name, value in settings.items():
            jax.config.update(name, value)
        compilation_cache.reset_cache()
        first = np.asarray(run()(x, w), dtype=np.float32)
        # The second compilation must come from the cache written by the first.
        jax.clear_caches()
        second = np.asarray(run()(x, w), dtype=np.float32)
    finally:
        jax.monitoring.unregister_event_listener(listener)
        for name, value in previous.items():
            jax.config.update(name, value)
        compilation_cache.reset_cache()

    assert "/jax/compilation_cache/cache_hits" in events
    np.testing.assert_array_equal(first, second)
