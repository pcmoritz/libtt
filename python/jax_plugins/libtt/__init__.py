"""JAX startup hooks for the libtt PJRT plugin."""

from pathlib import Path


def initialize() -> None:
    """Register the bundled PJRT plugin and enable input donation.

    JAX currently gates donation with an internal platform allowlist before a
    PJRT plugin is loaded. Registering this namespace plugin lets libtt opt in
    during normal JAX plugin discovery.
    """
    import jax._src.xla_bridge as xb
    from jax._src.interpreters import mlir

    platforms = getattr(mlir, "_platforms_with_donation", None)
    if platforms is not None and "tt" not in platforms:
        platforms.append("tt")

    library_path = Path(__file__).with_name("libtt.so")
    if not library_path.is_file():
        raise FileNotFoundError(f"libtt PJRT plugin not found: {library_path}")

    xb.register_plugin(
        "tt",
        priority=500,
        library_path=str(library_path),
        options=None,
    )
