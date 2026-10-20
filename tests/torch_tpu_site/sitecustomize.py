"""Runs torch_tpu on libtt in every Python process of torch_tpu's tests.

The test runner puts this directory on PYTHONPATH, so that the processes the
tests start load libtt as torch_tpu's "tpu" PJRT plugin too.
"""

import jax_plugins.libtt.tpu_compat  # noqa: F401
