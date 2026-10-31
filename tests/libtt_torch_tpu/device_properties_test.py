"""torch_tpu's hardware helpers describe the Tenstorrent devices (tpu_compat)."""

import pytest
import torch
from torch_tpu._internal.utils import hardware


def test_device_properties_describe_the_chip():
    props = torch.tpu.get_device_properties()
    assert props.name.startswith("Tenstorrent")
    assert props.name == torch.tpu.get_device_name() == hardware.get_tpu_device_name()
    assert props.total_memory == hardware.get_hbm_bytes_per_device()
    assert props.total_memory == hardware.get_hbm_bytes_per_chip()
    assert hardware.get_devices_per_chip() == 1
    # The allocator manages the chip's DRAM except for its reserved regions.
    _, allocatable = torch.accelerator.get_memory_info()
    assert 0 < allocatable <= props.total_memory


def test_topology_is_a_row_of_the_chips():
    count = hardware.get_tpu_device_count()
    assert count >= 1
    assert hardware.get_tpu_topology() == f"1,{count},1"
    assert hardware.get_tpu_topology(1) == "1,1,1"
    assert hardware.get_tpu_topology(4) == "1,4,1"
    with pytest.raises(RuntimeError):
        hardware.get_tpu_topology(0)
