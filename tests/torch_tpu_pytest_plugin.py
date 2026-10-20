"""Pytest plugin that parses absl flags for torch_tpu's tests.

absltest.main() would otherwise do this. Extra flags come from
$TORCH_TPU_ABSL_FLAGS as MODULE:FLAG items, each for the module that defines
the flag: several modules define flags of the same name, e.g. --test_mode. The
flags are set again after collection, since test modules define their flags
when they are imported.
"""

import os
import shlex

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


def pytest_collection_finish(session):
    del session
    _set_module_flags()
