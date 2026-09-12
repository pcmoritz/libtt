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
    suite = parser.add_mutually_exclusive_group(required=True)
    suite.add_argument("--jax-tests-anchor")
    suite.add_argument("--libtt-tests-anchor")
    parser.add_argument("--jax-plugin-wheel", required=True)
    parser.add_argument("--skip-device-check", action="store_true")
    args, pytest_args = parser.parse_known_args()

    plugin_wheel = _rlocation(args.jax_plugin_wheel).resolve(strict=True)
    if args.jax_tests_anchor:
        test_root = _rlocation(args.jax_tests_anchor).resolve(strict=True).parent.parent
        default_tests = ["tests"]
    else:
        test_root = _rlocation(args.libtt_tests_anchor).resolve(strict=True).parent
        default_tests = ["."]

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
    return pytest.main(pytest_args or default_tests)


if __name__ == "__main__":
    sys.exit(main())
