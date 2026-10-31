"""TPU compatibility mode: runs torch_tpu on Tenstorrent devices.

`LIBTT_TPU_COMPAT=1` imports this module at interpreter startup (through
`libtt_tpu_compat.pth`), before torch_tpu loads:

- torch_tpu loads its "tpu" PJRT plugin from `TPU_LIBRARY_PATH`, which now
  defaults to this package's `libtt.so`.
- torch_tpu only opens a device when its PCI scan finds a Google TPU, so its
  device count reports the Tenstorrent chips instead.
- libtt reports the platform as "tpu", so torch_tpu compiles its eager ops as it
  does on a TPU, which libtt doesn't trace.
- torch_tpu's flash attention is a Pallas kernel that only TPUs run, so
  `scaled_dot_product_attention` takes its StableHLO implementation instead.
- SGLang-TorchTPU's attention kernel runs as JAX (`sglang_tpu_kernels`).
- torch_tpu destroys its PJRT client at exit only if it initialized its runtime
  options, which `torch.manual_seed` alone doesn't, and a device left open
  keeps the fabric routers of a multi-chip mesh running into the next process.
  So at exit `tt_pjrt_shutdown`, which tt-xla's own plugin package calls at
  exit too, closes the device of a client that is still alive.
"""

import atexit
import ctypes
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


def _patch_hardware(module):
    module.get_tpu_device_count = _tenstorrent_device_count
    # torch_tpu imports torch before its hardware module.
    import torch

    torch.backends.cuda.enable_flash_sdp(False)


class _PatchModules(importlib.abc.MetaPathFinder):
    """Runs `patches[name](module)` once module `name` is imported."""

    def __init__(self, patches):
        self._patches = dict(patches)

    def find_spec(self, name, path, target=None):
        patch = self._patches.pop(name, None)
        if patch is None:
            return None
        if not self._patches:
            sys.meta_path.remove(self)
        spec = importlib.util.find_spec(name)
        exec_module = spec.loader.exec_module

        def patched_exec_module(module):
            exec_module(module)
            patch(module)

        spec.loader.exec_module = patched_exec_module
        return spec


def _shutdown_plugin():
    try:
        # Only a process that loaded the plugin has a device to close.
        plugin = ctypes.CDLL(os.environ["TPU_LIBRARY_PATH"], mode=os.RTLD_NOLOAD | os.RTLD_LAZY)
    except (KeyError, OSError):
        return
    plugin.tt_pjrt_shutdown()


os.environ.setdefault("TPU_LIBRARY_PATH", str(pathlib.Path(__file__).with_name("libtt.so")))
# libtt.so reads it too; torch_tpu's tests turn compatibility mode on by import.
os.environ["LIBTT_TPU_COMPAT"] = "1"
# Registered before torch_tpu's own exit handlers, so it runs after them.
atexit.register(_shutdown_plugin)

from jax_plugins.libtt import sglang_tpu_kernels  # noqa: E402

sys.meta_path.insert(
    0,
    _PatchModules(
        {
            _HARDWARE_MODULE: _patch_hardware,
            **sglang_tpu_kernels.PATCHES,
        }
    ),
)
