# libtt

`libtt.so` is a Bazel-built PJRT plugin for Tenstorrent devices. The PJRT
implementation comes from the pinned `tt-xla` repository, with `tt-mlir` and
`tt-metal` built through Bazel overlays in this repository.

The local code in this repository is intentionally small:

- it materializes the embedded TT-Metal runtime archive before the plugin starts;
- it links the upstream `tt-xla` PJRT plugin into the final shared library;
- it hides internal symbols so the shared object only exports the PJRT entrypoints.

## Build

```bash
bazel build //:tt
```

The output is `bazel-bin/libtt.so` on Linux and `bazel-bin/libtt.dylib` on
macOS.

## Smoke Test

```bash
env -u TT_METAL_RUNTIME_ROOT PJRT_NAMES_AND_LIBRARY_PATHS=tt:bazel-bin/libtt.so JAX_PLATFORMS=tt python -c "import jax, jax.numpy as jnp; x = jnp.arange(4, dtype=jnp.float32); y = jax.jit(lambda a: a + jnp.float32(1.0))(x); print(jax.devices('tt')); print(y)"
```
