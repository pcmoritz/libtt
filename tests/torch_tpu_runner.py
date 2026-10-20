#!/usr/bin/env python3
"""Run torch_tpu's tests against this checkout's PJRT plugin wheel.

Every test file runs in its own pytest process, as in torch_tpu's own Bazel
setup: test modules define conflicting absl flags, and a few tests abort the
process. Positional arguments that are paths in the torch_tpu repository, such
as `tests/scan_test.py` or `tests/compile`, select the tests (by default all of
`tests` and `src/torch_tpu`); other arguments go to pytest. Each file's log and
JUnit XML are written to the test's undeclared outputs.
"""

import argparse
import collections
import importlib.util
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import time

# Bazel's Python bootstrap doesn't put this directory on the path.
sys.path.insert(0, str(Path(__file__).parent))
from pytest_runner import _rlocation, extract_plugin_wheel  # noqa: E402

# Once a process logs one of these, the device fails every later file.
DEVICE_HANG_MARKERS = ("NOC0 is hung", "MMIO per-op timeout")

# pytest's final line, e.g. "== 3 failed, 9 passed, 2 subtests passed in 4.2s ==".
SUMMARY = re.compile(r"^=+ (.*) in [0-9.]+s\b.*=+$")


def _overlay_test_modules(root: Path, overlay: Path) -> None:
    """Mirrors the installed torch_tpu package and adds its test-only modules.

    The wheel leaves out modules such as test_utils.py, and the __init__.py
    files that would make the repository's packages shadow the wheel's compiled
    modules, so the source tree can't go on the path as it is. The mirror is
    complete, since the shared libraries load each other by relative paths.
    """
    installed = Path(importlib.util.find_spec("torch_tpu").origin).parent
    source = root / "src" / "torch_tpu"
    for path in [*installed.rglob("*"), *source.rglob("*.py")]:
        base = installed if path.is_relative_to(installed) else source
        mirrored = overlay / "torch_tpu" / path.relative_to(base)
        if path.is_dir():
            mirrored.mkdir(parents=True, exist_ok=True)
        elif not mirrored.exists() and not path.name.endswith("_test.py"):
            mirrored.parent.mkdir(parents=True, exist_ok=True)
            mirrored.symlink_to(path)


def _select(root: Path, selections: list[str], excludes: list[str]) -> list[str]:
    targets = []
    for selection in selections or ["tests", "src/torch_tpu"]:
        path = root / selection.split("::")[0]
        if path.is_dir():
            tests = (str(test.relative_to(root)) for test in path.rglob("*_test.py"))
            targets += sorted(test for test in tests if test not in excludes)
        else:
            targets.append(selection)
    return targets


def _counts(log: str) -> collections.Counter:
    counts = collections.Counter()
    for line in reversed(log.splitlines()):
        match = SUMMARY.match(line.strip())
        if match:
            for part in match.group(1).split(", "):
                number, _, kind = part.partition(" ")
                if number.isdigit():
                    counts[kind.replace("errors", "error")] += int(number)
            break
    return counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tests-anchor",
        required=True,
        help="Runfile in the root of the torch_tpu repository.",
    )
    parser.add_argument("--jax-plugin-wheel", required=True)
    parser.add_argument(
        "--absl-flag",
        action="append",
        default=[],
        metavar="MODULE:FLAG",
        help="absl flag for the tests whose MODULE defines it; repeatable.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Test file to leave out of the directories it is in; repeatable.",
    )
    parser.add_argument(
        "--file-timeout",
        type=float,
        default=3600,
        help="Seconds after which a test file's process is killed.",
    )
    args, rest = parser.parse_known_args()

    root = _rlocation(args.tests_anchor).resolve(strict=True).parent
    plugin_wheel = _rlocation(args.jax_plugin_wheel).resolve(strict=True)
    plugin_root = extract_plugin_wheel(plugin_wheel)
    selections = [arg for arg in rest if (root / arg.split("::")[0]).exists()]
    pytest_args = [arg for arg in rest if arg not in selections]
    targets = _select(root, selections, args.exclude)

    tmp = Path(os.environ.get("TEST_TMPDIR") or tempfile.mkdtemp())
    out = Path(os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR") or tmp / "outputs")
    out.mkdir(parents=True, exist_ok=True)

    overlay = tmp / "torch_tpu_overlay"
    _overlay_test_modules(root, overlay)

    env = dict(os.environ)
    for name in ("TT_METAL_RUNTIME_ROOT", "PJRT_NAMES_AND_LIBRARY_PATHS"):
        env.pop(name, None)
    # The plugin wheel, torch_tpu_pytest_plugin, a sitecustomize that runs
    # torch_tpu on libtt, and torch_tpu with its test modules.
    here = Path(__file__).parent
    env["PYTHONPATH"] = os.pathsep.join(
        [str(plugin_root), str(here), str(here / "torch_tpu_site"), str(overlay)]
        + sys.path
    )
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["TORCH_TPU_ABSL_FLAGS"] = shlex.join(args.absl_flag)
    # Some tests use JAX for Pallas kernels or shardings; keep it off the
    # device that torch_tpu uses.
    env["JAX_PLATFORMS"] = "cpu"

    print(f"Running {len(targets)} torch_tpu test files on {plugin_wheel.name}")
    totals = collections.Counter()
    failed = []
    for index, target in enumerate(targets, 1):
        name = re.sub(r"[^A-Za-z0-9_.-]", "_", target)
        log_path = out / f"{name}.log"
        file_env = dict(env, TEST_TMPDIR=str(tmp / name))
        Path(file_env["TEST_TMPDIR"]).mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-m",
            "pytest",
            target,
            "-p",
            "torch_tpu_pytest_plugin",
            "-p",
            "no:cacheprovider",
            f"--junitxml={out / name}.xml",
            *pytest_args,
        ]
        if target.startswith("src/"):
            # Keep the test's directory off the path: modules there, such as
            # torch_tpu's multiprocessing.py, would shadow the standard library.
            command.append("--import-mode=importlib")
        start = time.monotonic()
        with open(log_path, "w") as log:
            process = subprocess.Popen(
                command,
                cwd=root,
                env=file_env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                returncode = process.wait(timeout=args.file_timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                returncode = None
        text = log_path.read_text(errors="replace")
        counts = _counts(text)
        totals.update(counts)
        if returncode is None:
            status = "timed out"
        elif returncode < 0:
            status = f"crashed ({signal.Signals(-returncode).name})"
        else:
            status = ", ".join(f"{n} {kind}" for kind, n in counts.items())
            if returncode not in (0, 5) and not counts["failed"] + counts["error"]:
                status += f"{', ' if status else ''}exit code {returncode}"
        # 5 means that pytest collected no tests.
        if returncode not in (0, 5):
            failed.append(target)
        seconds = time.monotonic() - start
        print(
            f"[{index}/{len(targets)}] {target}: {status} ({seconds:.0f}s)", flush=True
        )
        if any(marker in text for marker in DEVICE_HANG_MARKERS):
            print(f"The device hung in {target}; stopping.")
            failed += targets[index:]
            break

    kinds = ("passed", "failed", "error", "skipped", "subtests passed")
    print(
        f"\n{len(targets) - len(failed)} of {len(targets)} files passed. Tests: "
        + ", ".join(f"{totals[kind]} {kind}" for kind in kinds)
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
