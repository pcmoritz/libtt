#!/usr/bin/env python3
"""Run upstream JAX tests against this repository's libtt PJRT plugin."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


def _rlocation(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate

    manifest = os.environ.get("RUNFILES_MANIFEST_FILE")
    if manifest:
        with open(manifest, encoding="utf-8") as f:
            for line in f:
                logical, _, physical = line.rstrip("\n").partition(" ")
                if logical == path:
                    return Path(physical)

    runfiles_dir = os.environ.get("RUNFILES_DIR")
    if runfiles_dir:
        return Path(runfiles_dir) / path

    return Path.cwd() / path


def _rewrite_pytest_arg(arg: str, jax_repo: Path, tests_root: Path) -> str:
    if arg.startswith("-"):
        return arg

    path = Path(arg)
    if path.is_absolute():
        return arg

    if arg.startswith("tests/"):
        return str(jax_repo / path)

    tests_path = tests_root / path
    if tests_path.exists():
        return str(tests_path)

    repo_path = jax_repo / path
    if repo_path.exists():
        return str(repo_path)

    return arg


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jax-tests-anchor", required=True)
    parser.add_argument("--libtt", required=True)
    parser.add_argument(
        "--skip-device-check",
        action="store_true",
        help="Configure the plugin but do not eagerly initialize jax.devices('tt').",
    )
    args, pytest_args = parser.parse_known_args()

    libtt = _rlocation(args.libtt).resolve()
    if not libtt.exists():
        raise FileNotFoundError(f"libtt shared library runfile not found: {libtt}")

    tests_anchor = _rlocation(args.jax_tests_anchor).resolve()
    if not tests_anchor.exists():
        raise FileNotFoundError(f"JAX test anchor runfile not found: {tests_anchor}")

    tests_root = tests_anchor.parent
    jax_repo = tests_root.parent

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
            raise RuntimeError(
                "JAX initialized the tt platform but returned no TT devices"
            )
        print(f"Using JAX {jax.__version__} with {len(devices)} TT device(s)")

    import pytest

    rewritten_pytest_args = [
        _rewrite_pytest_arg(arg, jax_repo, tests_root) for arg in pytest_args
    ]
    if not rewritten_pytest_args:
        rewritten_pytest_args = [str(tests_root)]

    return pytest.main(rewritten_pytest_args)


if __name__ == "__main__":
    sys.exit(main())
