"""Exercise torch_tpu's unchanged uint64 RNG state updates on libtt."""

import struct

import pytest
import torch
from torch_tpu._internal.utils import annotations


def setup_module():
    annotations._internal_apis_allowed_for_testing = True


def state(seed, counter):
    return torch.tensor(list(struct.pack("<QQ", seed, counter)), dtype=torch.uint8)


@pytest.mark.parametrize("compiled", [False, True])
def test_counter_carry_and_state_restore(compiled):
    draw = lambda: torch.rand(64, device="tpu")
    if compiled:
        draw = torch.compile(draw, fullgraph=True)
    seed, counter = 2**32 + 42, 2**32 - 16
    initial = state(seed, counter)
    torch.tpu.set_rng_state(initial)
    assert torch.equal(torch.tpu.get_rng_state(), initial)
    first = draw().cpu()
    assert torch.equal(torch.tpu.get_rng_state(), state(seed, 2**32))
    after_carry = draw().cpu()
    assert torch.equal(torch.tpu.get_rng_state(), state(seed, 2**32 + 16))
    torch.tpu.set_rng_state(state(seed, 0))
    low_counter = draw().cpu()
    assert not torch.equal(after_carry, low_counter)
    torch.tpu.set_rng_state(initial)
    assert torch.equal(draw().cpu(), first)


@pytest.mark.parametrize("compiled", [False, True])
def test_high_seed_bits(compiled):
    draw = lambda: torch.rand(64, device="tpu")
    if compiled:
        draw = torch.compile(draw, fullgraph=True)
    torch.manual_seed(42)
    low_seed = draw().cpu()
    torch.manual_seed(2**32 + 42)
    high_seed = draw().cpu()
    assert not torch.equal(low_seed, high_seed)
    torch.manual_seed(2**32 + 42)
    assert torch.equal(draw().cpu(), high_seed)


@pytest.mark.parametrize("compiled", [False, True])
def test_counter_high_word_does_not_alias_another_seed(compiled):
    draw = lambda: torch.rand(64, device="tpu")
    if compiled:
        draw = torch.compile(draw, fullgraph=True)
    torch.tpu.set_rng_state(state(42, 2**32))
    carried = draw().cpu()
    torch.manual_seed(43)
    assert not torch.equal(carried, draw().cpu())


@pytest.mark.parametrize("compiled", [False, True])
def test_large_draw_does_not_repeat_after_1024_calls(compiled):
    # torch_tpu advances by 2**22 for this shape. Truncating its uint64
    # counter made the entire tensor repeat after only 1024 draws.
    draw = lambda: torch.rand(4096, 4096, device="tpu")
    if compiled:
        draw = torch.compile(draw, fullgraph=True)
    torch.manual_seed(7)
    first = draw().cpu()
    for _ in range(1024):
        last = draw()
    last = last.cpu()
    assert torch.equal(torch.tpu.get_rng_state(), state(7, 1025 * 2**22))
    assert not torch.equal(first, last)
