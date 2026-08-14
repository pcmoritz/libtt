"""JAX startup hooks for the libtt PJRT plugin."""


def initialize() -> None:
    """Enable input donation for JAX's ``tt`` platform.

    JAX currently gates donation with an internal platform allowlist before a
    PJRT plugin is loaded. Registering this namespace plugin lets libtt opt in
    during normal JAX plugin discovery.
    """
    from jax._src.interpreters import mlir

    platforms = getattr(mlir, "_platforms_with_donation", None)
    if platforms is not None and "tt" not in platforms:
        platforms.append("tt")
