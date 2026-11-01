"""Read a tensor while a queued op takes it as its input.

torch_tpu dispatches ops on a thread of its own, so a tensor read on the
caller's thread can be the input of an op dispatched at the same time, as
SGLang's sampled token ids are of the next step. Reading moves the tensor to
the host and an input on the host moves back to the device; interleaved, the
read returned the previous step's values, after a few hundred steps nearly
every time, or found the tensor on the device ("Tensor must be on host").
"""

import torch


def test_read_while_a_queued_op_takes_it():
    t = torch.zeros(1, 32, dtype=torch.int64, device="tpu")
    wrong = []
    for i in range(1, 1001):
        t = t + 1
        consumer = t * 2
        value = t.cpu()
        doubled = consumer.cpu()
        if not (value == i).all() or not (doubled == 2 * i).all():
            wrong.append((i, value[0, 0].item(), doubled[0, 0].item()))
    assert not wrong, f"{len(wrong)} of 1000 steps wrong, e.g. {wrong[:5]}"
