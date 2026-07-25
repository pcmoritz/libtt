#!/usr/bin/env python3
"""Run upstream JAX tests against this checkout's libtt PJRT plugin."""

import argparse
import os
from pathlib import Path
import sys

from python.runfiles import runfiles


def _rlocation(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    resolved = runfiles.Create().Rlocation(path)
    if not resolved:
        raise FileNotFoundError(f"runfile not found: {path}")
    return Path(resolved)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jax-tests-anchor", required=True)
    parser.add_argument("--libtt", required=True)
    parser.add_argument("--skip-device-check", action="store_true")
    args, pytest_args = parser.parse_known_args()

    libtt = _rlocation(args.libtt).resolve(strict=True)
    jax_repo = _rlocation(args.jax_tests_anchor).resolve(strict=True).parent.parent

    os.environ.pop("TT_METAL_RUNTIME_ROOT", None)
    os.environ["PJRT_NAMES_AND_LIBRARY_PATHS"] = f"tt:{libtt}"
    os.environ["JAX_PLATFORMS"] = "tt"
    os.environ["JAX_USE_SHARDY_PARTITIONER"] = "false"
    os.environ.setdefault(
        "JAX_COMPILATION_CACHE_DIR",
        str(Path(os.environ.get("TEST_TMPDIR", "/tmp")) / "jax_compilation_cache"),
    )

    if not args.skip_device_check:
        import jax

        devices = jax.devices("tt")
        if not devices:
            raise RuntimeError("JAX returned no TT devices")
        print(f"Using JAX {jax.__version__} with {len(devices)} TT device(s)")

    import pytest

    os.chdir(jax_repo)
    return pytest.main(pytest_args or ["tests"])


if __name__ == "__main__":
    sys.exit(main())
