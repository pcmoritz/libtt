"""libtt keeps SGLang-TorchTPU's compiled all-gathers native.

SGLang-TorchTPU replaces all-gathers in Dynamo-traced graphs with an all-reduce
over a zero-padded buffer (sglang_tpu.all_gather_via_all_reduce), a workaround
for a TPU compiler bug that moves twice the bytes. Before
sgl-project/sglang-torchtpu#2041 its hook wrapped every
GroupCoordinator.all_gather_into_tensor; since, hooks on the logits gathers ask
_use_all_reduce_gather. The module stand-ins here have each version's names.
"""

import types

from jax_plugins.libtt import sglang_tpu_kernels


def test_before_2041_the_around_hook_calls_the_native_gather():
    module = types.ModuleType("sglang_tpu.all_gather_via_all_reduce")
    module.all_gather_into_tensor_around = lambda original, group, output, input: "all-reduce"
    sglang_tpu_kernels.patch_all_gather_via_all_reduce(module)
    calls = []
    module.all_gather_into_tensor_around(lambda *args: calls.append(args) or "native", "group", "out", "in")
    assert calls == [("group", "out", "in")]


def test_since_2041_the_logits_gathers_stay_native():
    module = types.ModuleType("sglang_tpu.all_gather_via_all_reduce")
    module._use_all_reduce_gather = lambda group: True

    def gather_attn_tp_logits_around(original, processor, logits):
        # Upstream's hooks look _use_all_reduce_gather up when called.
        return "all-reduce" if module._use_all_reduce_gather("group") else original(processor, logits)

    sglang_tpu_kernels.patch_all_gather_via_all_reduce(module)
    assert not module._use_all_reduce_gather("group")
    assert gather_attn_tp_logits_around(lambda processor, logits: "native", None, None) == "native"
