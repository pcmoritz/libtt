"""Pytest plugin that parses absl flags for torch_tpu's tests.

absltest.main() would otherwise do this. Extra flags come from
$ABSL_TEST_FLAGS. They are parsed again after collection, since test modules
define their flags when they are imported.
"""

import os
import shlex

from absl import flags


def _parse_extra_flags():
    # The flags are for torch_tpu's op tests. A module that defines a flag of
    # the same name with other values keeps its default.
    for flag in shlex.split(os.environ.get("ABSL_TEST_FLAGS", "")):
        try:
            flags.FLAGS(["pytest", flag], known_only=True)
        except flags.IllegalFlagValueError:
            pass


def pytest_configure(config):
    del config
    if not flags.FLAGS.is_parsed():
        flags.FLAGS(["pytest"], known_only=True)
    _parse_extra_flags()


def pytest_collection_finish(session):
    del session
    _parse_extra_flags()
