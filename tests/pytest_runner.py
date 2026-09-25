#!/usr/bin/env python3
"""Run JAX or libtt tests against this checkout's PJRT plugin wheel."""

import argparse
import os
from pathlib import Path
import sys
import zipfile

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
    parser.add_argument(
        "--tests-anchor",
        required=True,
        help="Runfile in the working directory for pytest.",
    )
    parser.add_argument("--jax-plugin-wheel", required=True)
    parser.add_argument("--skip-device-check", action="store_true")
    args, pytest_args = parser.parse_known_args()

    plugin_wheel = _rlocation(args.jax_plugin_wheel).resolve(strict=True)
    test_root = _rlocation(args.tests_anchor).resolve(strict=True).parent

    # A wheel is a zip archive, but libtt.so must be a real file for dlopen.
    # Extracting it also exercises the exact artifact users install.
    plugin_root = Path(os.environ.get("TEST_TMPDIR", "/tmp")) / "jax_tt_plugin"
    with zipfile.ZipFile(plugin_wheel) as wheel:
        wheel.extractall(plugin_root)
    sys.path.insert(0, str(plugin_root))

    os.environ.pop("TT_METAL_RUNTIME_ROOT", None)
    os.environ.pop("PJRT_NAMES_AND_LIBRARY_PATHS", None)
    # Keep TT as the default while making CPU available to upstream tests that
    # explicitly exercise default-device and cross-backend behavior.
    os.environ["JAX_PLATFORMS"] = "tt,cpu"
    os.environ["JAX_USE_SHARDY_PARTITIONER"] = "false"
    os.environ.setdefault(
        "JAX_COMPILATION_CACHE_DIR",
        str(Path(os.environ.get("TEST_TMPDIR", "/tmp")) / "jax_compilation_cache"),
    )

    # Newer JAX wheels omit this helper required by the upstream tests.
    source_helpers = test_root / "jax" / "_src"
    if source_helpers.is_dir():
        import jax._src

        jax._src.__path__.append(str(source_helpers))

    # Upstream tests assert that no test replaces the global compilation cache
    # or changes config, which creating the cache on first use would do.
    from jax._src import compilation_cache

    compilation_cache._initialize_cache()

    if not args.skip_device_check:
        import jax

        devices = jax.devices("tt")
        if not devices:
            raise RuntimeError("JAX returned no TT devices")
        print(
            f"Using JAX {jax.__version__} with {len(devices)} TT device(s) "
            f"from {plugin_wheel.name}"
        )

    import pytest

    os.chdir(test_root)
    return pytest.main(pytest_args)


if __name__ == "__main__":
    sys.exit(main())
