"""TPU compatibility mode: runs torch_tpu on Tenstorrent devices.

`LIBTT_TPU_COMPAT=1` imports this module at interpreter startup (through
`libtt_tpu_compat.pth`), before torch_tpu loads:

- torch_tpu loads its "tpu" PJRT plugin from `TPU_LIBRARY_PATH`, which now
  defaults to this package's `libtt.so`.
- torch_tpu only opens a device when its PCI scan finds a Google TPU, so its
  device count reports the Tenstorrent chips instead.
- torch_tpu's flash attention is a Pallas kernel that only TPUs run, so
  `scaled_dot_product_attention` takes its StableHLO implementation instead,
  also where `sdpa_kernel` asks for FLASH_ATTENTION.
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


def _flash_attention_runs_as_stablehlo() -> None:
    """Makes `sdpa_kernel(FLASH_ATTENTION)` enable torch_tpu's StableHLO
    implementation (OVERRIDEABLE) and MATH instead of the Pallas kernel.

    torch_tpu then picks OVERRIDEABLE where it applies, and MATH for dropout
    and masks that need gradients, as it does when FLASH_ATTENTION is
    disabled: the same attention, but no native flash attention. The flag
    setter that `sdpa_kernel` (and torch.compile's tracing of it) calls is
    wrapped, so FLASH_ATTENTION, which this mode turns off, stays off; other
    selections and the flags restored on exit are unchanged.
    """
    import torch.nn.attention as attention

    backend = attention.SDPBackend
    set_backends = attention._sdpa_kernel

    def _sdpa_kernel(backends, set_priority=False):
        replaced = []
        for b in backends:
            if b == backend.FLASH_ATTENTION:
                replaced += [backend.OVERRIDEABLE, backend.MATH]
            else:
                replaced.append(b)
        set_backends(list(dict.fromkeys(replaced)), set_priority)

    attention._sdpa_kernel = _sdpa_kernel


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
            _flash_attention_runs_as_stablehlo()

        spec.loader.exec_module = patched_exec_module
        return spec


os.environ.setdefault("TPU_LIBRARY_PATH", str(pathlib.Path(__file__).with_name("libtt.so")))
sys.meta_path.insert(0, _PatchTorchTpu())
