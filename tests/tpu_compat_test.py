"""LIBTT_TPU_COMPAT turns explicit FLASH_ATTENTION requests into torch_tpu's
StableHLO implementation (OVERRIDEABLE) and MATH, and leaves the rest of
torch's attention backend selection as it was. Runs on the CPU."""

import unittest

import torch
from torch.nn import attention
from torch.nn.attention import SDPBackend

from jax_plugins.libtt import tpu_compat

tpu_compat._flash_attention_runs_as_stablehlo()

FLASH = SDPBackend.FLASH_ATTENTION
STABLEHLO = {SDPBackend.OVERRIDEABLE, SDPBackend.MATH}


def enabled():
    return set(attention._cur_sdpa_kernel_backends())


def without_flash(priority):
    return [b for b in priority if b != int(FLASH)]


class SdpaKernelTest(unittest.TestCase):
    def setUp(self):
        # The compatibility mode turns FLASH_ATTENTION off at startup.
        torch.backends.cuda.enable_flash_sdp(False)
        self.flags = attention._cur_sdpa_kernel_backends()
        self.priority = torch._C._get_sdp_priority_order()

    def tearDown(self):
        self.assertEqual(attention._cur_sdpa_kernel_backends(), self.flags)
        # sdpa_kernel restores the priorities of the enabled backends; the
        # disabled FLASH_ATTENTION's moves to the end, as without this mode.
        self.assertEqual(without_flash(torch._C._get_sdp_priority_order()), without_flash(self.priority))
        self.assertNotIn(FLASH, enabled())

    def test_flash_selects_stablehlo_and_math(self):
        for backends in (FLASH, [FLASH]):
            with attention.sdpa_kernel(backends):
                self.assertEqual(enabled(), STABLEHLO)

    def test_flash_keeps_the_other_selected_backends(self):
        with attention.sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, FLASH]):
            self.assertEqual(enabled(), STABLEHLO | {SDPBackend.EFFICIENT_ATTENTION})

    def test_other_selections_are_unchanged(self):
        for backends in (
            [SDPBackend.MATH],
            [SDPBackend.OVERRIDEABLE],
            [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION],
        ):
            with attention.sdpa_kernel(backends):
                self.assertEqual(enabled(), set(backends))

    def test_nested_contexts_restore_the_flags(self):
        with attention.sdpa_kernel([SDPBackend.MATH]):
            with attention.sdpa_kernel(FLASH):
                self.assertEqual(enabled(), STABLEHLO)
            self.assertEqual(enabled(), {SDPBackend.MATH})
            with attention.sdpa_kernel([SDPBackend.OVERRIDEABLE]):
                self.assertEqual(enabled(), {SDPBackend.OVERRIDEABLE})
            self.assertEqual(enabled(), {SDPBackend.MATH})

    def test_priority(self):
        with attention.sdpa_kernel([FLASH, SDPBackend.EFFICIENT_ATTENTION], set_priority=True):
            self.assertEqual(
                torch._C._get_sdp_priority_order()[:3],
                [int(SDPBackend.OVERRIDEABLE), int(SDPBackend.MATH), int(SDPBackend.EFFICIENT_ATTENTION)],
            )
        with attention.sdpa_kernel([SDPBackend.MATH, SDPBackend.OVERRIDEABLE], set_priority=True):
            self.assertEqual(
                torch._C._get_sdp_priority_order()[:2],
                [int(SDPBackend.MATH), int(SDPBackend.OVERRIDEABLE)],
            )


if __name__ == "__main__":
    unittest.main()
