"""A process that exits with torch_tpu's client alive leaves the device usable.

torch_tpu's exit handler only drops its own reference to the PJRT client, and
tensors still alive keep the client alive too, such as the generator state that
`torch.manual_seed` creates, so the client is never destroyed. On a multi-chip
host the device that client opened runs fabric routers on the chips' Ethernet
cores, and if they kept running, the next process would time out waiting for
those cores. The test's own process doesn't open the device.
"""

import subprocess
import sys


def run(code):
    subprocess.run([sys.executable, "-c", "import torch, torch_tpu\n" + code], check=True, timeout=300)


def test_device_after_process_that_only_seeds():
    run("torch.manual_seed(0)")
    run("assert torch.ones(8, device='tpu').sum().item() == 8")


def test_device_after_process_that_runs_programs():
    """Closing the device at exit frees the programs it cached, which uses
    tt-metal's thread-local state, already destroyed on the exiting thread."""
    run("x = torch.randn(64, 64, device='tpu')\nassert torch.cumsum((x @ x).relu().sum(0), 0).shape == (64,)")
    run("assert torch.ones(8, device='tpu').sum().item() == 8")
