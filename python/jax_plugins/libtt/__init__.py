"""JAX startup hooks for the libtt PJRT plugin."""

from pathlib import Path


def initialize() -> None:
    """Register the bundled PJRT plugin and enable input donation and caching.

    JAX currently gates donation and its persistent compilation cache with
    internal platform allowlists before a PJRT plugin is loaded. Registering
    this namespace plugin lets libtt opt in during normal JAX plugin discovery.
    """
    import jax._src.xla_bridge as xb
    from jax._src import cache_key, compilation_cache
    from jax._src.interpreters import mlir

    platforms = getattr(mlir, "_platforms_with_donation", None)
    if platforms is not None and "tt" not in platforms:
        platforms.append("tt")

    library_path = Path(__file__).with_name("libtt.so")
    if not library_path.is_file():
        raise FileNotFoundError(f"libtt PJRT plugin not found: {library_path}")

    # The persistent compilation cache is gated the same way. libtt executables
    # serialize and load back. Cache keys include the hash of libtt.so that the
    # build stores next to it, so entries compiled by another build are not
    # reused; without that hash the cache stays off.
    hash_path = library_path.with_name("libtt.so.sha256")
    is_cache_used = compilation_cache.is_cache_used
    if hash_path.is_file() and not getattr(is_cache_used, "libtt", False):
        library_hash = hash_path.read_text().strip()
        custom_hook = cache_key.custom_hook

        def is_cache_used_or_tt(backend):
            if backend.platform == "tt":
                return compilation_cache._is_cache_enabled()
            return is_cache_used(backend)

        is_cache_used_or_tt.libtt = True
        compilation_cache.is_cache_used = is_cache_used_or_tt
        cache_key.custom_hook = lambda: custom_hook() + library_hash

    xb.register_plugin(
        "tt",
        priority=500,
        library_path=str(library_path),
        options=None,
    )
