"""SGLang-TorchTPU's recurrent-state slot copies, radix-cache copy-on-write, without its Pallas
kernel (sglang_tpu_kernels.patch_mamba_state_copy): every pool's first `count` source slots go
to their destinations, and the pairs that pad the bucket write nothing."""

from types import SimpleNamespace

import torch
import torch_tpu  # noqa: F401
from jax_plugins.libtt import sglang_tpu_kernels


def test_copies_the_counted_slots():
    module = SimpleNamespace()
    sglang_tpu_kernels.patch_mamba_state_copy(module)
    pools = [
        torch.arange(8 * 3 * 4, dtype=torch.float32).reshape(8, 3, 4),
        torch.arange(8 * 2 * 2 * 2, dtype=torch.float32).reshape(8, 2, 2, 2) + 1000,
    ]
    expected = [pool.clone() for pool in pools]
    for pool in expected:
        pool[[5, 6]] = pool[[2, 3]]
    on_device = [pool.to("tpu") for pool in pools]
    # Two pairs in a bucket of 8; the padding pairs (1 to 7) must not copy.
    ids = lambda x: torch.tensor(x, dtype=torch.int32, device="tpu")
    module.compiled_state_copy()(
        on_device, ids([2, 3, 1, 1, 1, 1, 1, 1]), ids([5, 6, 7, 7, 7, 7, 7, 7]), ids([2])
    )
    for got, want in zip(on_device, expected):
        assert torch.equal(got.cpu(), want)
