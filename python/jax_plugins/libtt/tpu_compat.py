"""TPU compatibility mode: runs torch_tpu on Tenstorrent devices.

`LIBTT_TPU_COMPAT=1` imports this module at interpreter startup (through
`libtt_tpu_compat.pth`), before torch_tpu loads:

- torch_tpu loads its "tpu" PJRT plugin from `TPU_LIBRARY_PATH`, which now
  defaults to this package's `libtt.so`.
- torch_tpu only opens a device when its PCI scan finds a Google TPU, so its
  device count reports the Tenstorrent chips instead. Its other hardware
  helpers (device name, memory per chip and device, devices per chip,
  topology) describe them too, from the kernel driver's sysfs entries and the
  boards' DRAM, without opening a device.
- torch_tpu's flash attention is a Pallas kernel that only TPUs run, so
  `scaled_dot_product_attention` takes its StableHLO implementation instead.
"""

import importlib.abc
import importlib.util
import os
import pathlib
import re
import sys

_HARDWARE_MODULE = "torch_tpu._internal.utils.hardware"


# DRAM per chip by board, the card type without its variant letter: Blackhole
# chips have 8 banks of 4 GiB (one is harvested on the p100), Wormhole chips 6
# banks of 2 GiB.
_DRAM_BYTES_PER_CHIP = {
    "n150": 12 * 1024**3,
    "n300": 12 * 1024**3,
    "p100": 28 * 1024**3,
    "p150": 32 * 1024**3,
    "p300": 32 * 1024**3,
}


def _tenstorrent_device_count() -> int:
    try:
        return sum(path.name.isdigit() for path in pathlib.Path("/dev/tenstorrent").iterdir())
    except OSError:
        return 0


def _tenstorrent_card_type() -> str | None:
    """The first Tenstorrent device's card type from the kernel driver, e.g. "p150a"."""
    try:
        paths = sorted(pathlib.Path("/sys/class/tenstorrent").glob("*/tt_card_type"))
        return paths[0].read_text().strip().lower() if paths else None
    except OSError:
        return None


def _tenstorrent_device_name() -> str:
    card_type = _tenstorrent_card_type()
    if card_type is None:
        return "Tenstorrent"
    architecture = {"n": "Wormhole", "p": "Blackhole"}.get(card_type[:1])
    return " ".join(filter(None, ["Tenstorrent", architecture, card_type]))


def _dram_bytes_per_chip() -> int | None:
    match = re.match(r"[a-z]+\d+", _tenstorrent_card_type() or "")
    return _DRAM_BYTES_PER_CHIP.get(match.group()) if match else None


def _device_count() -> int:
    # torch_tpu's (patched) device count, which tests may replace.
    return sys.modules[_HARDWARE_MODULE].get_tpu_device_count()


def _devices_per_chip() -> int | None:
    # A chip is one PJRT device.
    return 1 if _device_count() else None


def _tenstorrent_topology(world_size: int | None = None) -> str | None:
    """`world_size` chips (by default, all of them) as a 1xN mesh, libtt's default."""
    count = _device_count()
    if not count:
        return None
    target = count if world_size is None else world_size
    if target < 1:
        raise RuntimeError(f"No TPU topology found for count: {target}")
    return f"1,{target},1"


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
            module.get_tpu_device_name = _tenstorrent_device_name
            module.get_hbm_bytes_per_chip = _dram_bytes_per_chip
            module.get_hbm_bytes_per_device = _dram_bytes_per_chip
            module.get_devices_per_chip = _devices_per_chip
            module.get_tpu_topology = _tenstorrent_topology
            # torch_tpu imports torch before its hardware module.
            import torch

            torch.backends.cuda.enable_flash_sdp(False)

        spec.loader.exec_module = patched_exec_module
        return spec


os.environ.setdefault("TPU_LIBRARY_PATH", str(pathlib.Path(__file__).with_name("libtt.so")))
sys.meta_path.insert(0, _PatchTorchTpu())
