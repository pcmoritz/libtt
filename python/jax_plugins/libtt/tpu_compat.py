"""TPU compatibility mode: runs torch_tpu on Tenstorrent devices.

`LIBTT_TPU_COMPAT=1` imports this module at interpreter startup (through
`libtt_tpu_compat.pth`), before torch_tpu loads:

- torch_tpu loads its "tpu" PJRT plugin from `TPU_LIBRARY_PATH`, which now
  defaults to this package's `libtt.so`.
- torch_tpu only opens a device when its PCI scan finds a Google TPU, so its
  device count, topology, name and memory describe the Tenstorrent chips
  instead. The topology is a line of chips; `python -m torch.tpu.distributed`
  uses it to set up the launch environment for `torchrun`.
- torch_tpu's flash attention is a Pallas kernel that only TPUs run, so
  `scaled_dot_product_attention` takes its StableHLO implementation instead.
"""

import importlib.abc
import importlib.util
import os
import pathlib
import sys

_HARDWARE_MODULE = "torch_tpu._internal.utils.hardware"
# Each Blackhole chip has 32 GB of GDDR6.
_BLACKHOLE_DRAM_BYTES = 32 * 1024**3


def _tenstorrent_device_count() -> int:
    try:
        return sum(path.name.isdigit() for path in pathlib.Path("/dev/tenstorrent").iterdir())
    except OSError:
        return 0


def _tenstorrent_topology(world_size: int | None = None) -> str | None:
    count = world_size if world_size is not None else _tenstorrent_device_count()
    return f"{count},1,1" if count else None


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
            module.get_tpu_topology = _tenstorrent_topology
            module.get_tpu_device_name = lambda: "Tenstorrent Blackhole"
            module.get_hbm_bytes_per_device = lambda: _BLACKHOLE_DRAM_BYTES
            # torch_tpu imports torch before its hardware module.
            import torch

            torch.backends.cuda.enable_flash_sdp(False)

        spec.loader.exec_module = patched_exec_module
        return spec


os.environ.setdefault("TPU_LIBRARY_PATH", str(pathlib.Path(__file__).with_name("libtt.so")))
sys.meta_path.insert(0, _PatchTorchTpu())
