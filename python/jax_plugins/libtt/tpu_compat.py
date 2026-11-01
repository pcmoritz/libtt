"""TPU compatibility mode: runs torch_tpu on Tenstorrent devices.

`LIBTT_TPU_COMPAT=1` imports this module at interpreter startup (through
`libtt_tpu_compat.pth`), before torch_tpu loads:

- torch_tpu loads its "tpu" PJRT plugin from `TPU_LIBRARY_PATH`, which now
  defaults to this package's `libtt.so`.
- torch_tpu only opens a device when its PCI scan finds a Google TPU, so its
  device count reports the Tenstorrent chips instead.
- torch_tpu's flash attention is a Pallas kernel that only TPUs run, so
  `scaled_dot_product_attention` takes its StableHLO implementation instead.
"""

import importlib.abc
import importlib.util
import os
import pathlib
import sys

_HARDWARE_MODULE = "torch_tpu._internal.utils.hardware"


def _tenstorrent_device_count() -> int:
    try:
        return sum(path.name.isdigit() for path in pathlib.Path("/dev/tenstorrent").iterdir())
    except OSError:
        return 0


class _PatchTorchTpu(importlib.abc.MetaPathFinder):
    """Patches torch_tpu's hardware module when it is imported."""

    def find_spec(self, name, path, target=None):
        if name != _HARDWARE_MODULE:
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(name)
        exec_module = spec.loader.exec_module

        def patched_exec_module(module):
            exec_module(module)
            module.get_tpu_device_count = _tenstorrent_device_count
            # torch_tpu imports torch before its hardware module.
            import torch

            torch.backends.cuda.enable_flash_sdp(False)

        spec.loader.exec_module = patched_exec_module
        return spec


os.environ.setdefault("TPU_LIBRARY_PATH", str(pathlib.Path(__file__).with_name("libtt.so")))
sys.meta_path.insert(0, _PatchTorchTpu())
