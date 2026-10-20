"""Pytest plugin that sets up torch_tpu's tests as its own runner would.

It parses absl flags, which absltest.main() would otherwise do. Extra flags
come from $TORCH_TPU_ABSL_FLAGS as MODULE:FLAG items, each for the module that
defines the flag: several modules define flags of the same name, e.g.
--test_mode. The flags are set again after collection, since test modules
define their flags when they are imported.

In Google's monorepo, the tests are the package torch_tpu.tests, and some
patch their own modules by that name, so the plugin makes the `tests`
modules available as torch_tpu.tests too.
"""

import os
import shlex
import sys

from absl import flags


def _set_module_flags():
    for item in shlex.split(os.environ.get("TORCH_TPU_ABSL_FLAGS", "")):
        module, _, flag = item.partition(":")
        name = flag.lstrip("-").partition("=")[0]
        if name not in flags.FLAGS:
            continue
        defining = flags.FLAGS.find_module_defining_flag(name)
        if defining == module or defining.endswith("." + module):
            flags.FLAGS(["pytest", flag])


def pytest_configure(config):
    del config
    if not flags.FLAGS.is_parsed():
        flags.FLAGS(["pytest"], known_only=True)
    _set_module_flags()


def _alias_tests_package():
    tests = sys.modules.get("tests")
    if tests is None:
        return
    import torch_tpu

    torch_tpu.tests = tests
    for name, module in list(sys.modules.items()):
        if name == "tests" or name.startswith("tests."):
            sys.modules["torch_tpu." + name] = module


def pytest_collection_finish(session):
    del session
    _set_module_flags()
    _alias_tests_package()
