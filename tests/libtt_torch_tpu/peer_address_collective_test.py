"""Collectives between processes whose buffers' addresses differ.

ttnn's collectives write to their peers at the addresses of their own buffers,
which processes driving one chip each need not share: here rank 1 holds an extra
buffer first. The unicast all-gather trades output addresses between neighbors
and the line reduce-scatter intermediate addresses. All-reduces, gathers and
scatters of padded or unaligned dimensions, and the bytes of object collectives
are built from them; decode-sized all-reduces exchange into their program's own
scratch. The 512 MB all-gather result stays where the peers wrote it, and the
reduce-scatter, whose intermediate takes twice its 256 MB input, runs in two
pieces.
"""

import os
import pathlib
import socket
import subprocess
import sys

import pytest

CODE = """
import os

import torch
import torch.distributed as dist
import torch_tpu  # noqa: F401
from torch.tpu import distributed

ROWS, COLUMNS = 4096, 16384


def worker(rank):
    # torch_tpu binds a process to the chip of its local rank.
    os.environ["RANK"] = os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = os.environ["LOCAL_WORLD_SIZE"] = "2"
    dist.init_process_group("tpu_dist", rank=rank, world_size=2)
    held = None
    if rank == 1:
        # Rank 1's later buffers land at other addresses than rank 0's.
        held = torch.full((3, 1024, 1024), 7.0, device="tpu")
        held.sum().item()
    values = torch.arange(ROWS * COLUMNS, dtype=torch.float32).reshape(ROWS, COLUMNS) % 251
    x = (values + 1000 * rank).to("tpu")

    for _ in range(2):
        gathered = torch.empty(2 * ROWS, COLUMNS, device="tpu")
        dist.all_gather_into_tensor(gathered, x)
        assert torch.equal(gathered.cpu(), torch.cat([values, values + 1000])), "all-gather"

        scattered = torch.empty(ROWS // 2, COLUMNS, device="tpu")
        dist.reduce_scatter_tensor(scattered, x)
        want = (2 * values + 1000)[rank * ROWS // 2 : (rank + 1) * ROWS // 2]
        assert torch.equal(scattered.cpu(), want), "reduce-scatter"

        # An all-reduce as a reduce-scatter and an all-gather.
        summed = x.clone()
        dist.all_reduce(summed)
        assert torch.equal(summed.cpu(), 2 * values + 1000), "all-reduce"

    # Padded and unaligned dimensions, a scalar and a maximum.
    small = torch.arange(3 * 50, dtype=torch.float32).reshape(3, 50) + 100 * rank
    gathered = torch.empty(6, 50, device="tpu")
    dist.all_gather_into_tensor(gathered, small.to("tpu"))
    assert torch.equal(gathered.cpu(), torch.cat([small - 100 * rank, small - 100 * rank + 100])), "padded all-gather"

    rows = torch.arange(16 * 50, dtype=torch.float32).reshape(16, 50) + 100 * rank
    scattered = torch.empty(8, 50, device="tpu")
    dist.reduce_scatter_tensor(scattered, rows.to("tpu"))
    want = (2 * (rows - 100 * rank) + 100)[8 * rank : 8 * (rank + 1)]
    assert torch.equal(scattered.cpu(), want), "unaligned reduce-scatter"

    loss = torch.tensor(1.5 + rank, device="tpu")
    dist.all_reduce(loss)
    assert loss.item() == 4.0, f"scalar all-reduce: {loss.item()}"

    peak = small.to("tpu")
    dist.all_reduce(peak, op=dist.ReduceOp.MAX)
    assert torch.equal(peak.cpu(), small - 100 * rank + 100), "max all-reduce"

    # A decode-sized all-reduce (one tile row of BF16): ttnn's direct exchange, into its program's own scratch.
    row = (torch.arange(4096, dtype=torch.float32).reshape(1, 4096) % 97 + rank).to(torch.bfloat16)
    reduced = row.to("tpu")
    dist.all_reduce(reduced)
    want = (2 * (row.float() - rank) + 1).to(torch.bfloat16)
    assert torch.equal(reduced.cpu(), want), "direct all-reduce"

    # A padded row wider than 64 KB (a vocabulary's logits): gathered as row-major rows, each split across packets.
    logits = (torch.arange(37984, dtype=torch.float32).reshape(1, 37984) % 113 + rank).to(torch.bfloat16)
    gathered = torch.empty(2, 37984, dtype=torch.bfloat16, device="tpu")
    dist.all_gather_into_tensor(gathered, logits.to("tpu"))
    assert torch.equal(gathered.cpu(), torch.cat([logits - rank, logits - rank + 1])), "wide padded all-gather"

    # One tile each, repeatedly: fewer pages than worker slices, so some worker pairs have nothing to send. They
    # exchange no addresses; an address handshake without data could let an invocation's address arrive before
    # the previous one was taken.
    tile = torch.arange(32 * 32, dtype=torch.float32).reshape(32, 32) + rank
    for _ in range(100):
        gathered = torch.empty(64, 32, device="tpu")
        dist.all_gather_into_tensor(gathered, tile.to("tpu"))
        scattered = torch.empty(32, 32, device="tpu")
        dist.reduce_scatter_tensor(scattered, gathered)
    assert torch.equal(gathered.cpu(), torch.cat([tile - rank, tile - rank + 1])), "repeated one-tile all-gather"
    want = 2 * torch.cat([tile - rank, tile - rank + 1])[32 * rank : 32 * (rank + 1)]
    assert torch.equal(scattered.cpu(), want), "repeated one-tile reduce-scatter"

    # Object collectives send pickled bytes (UINT8), as distributed checkpoints do.
    objects = [None, None]
    dist.all_gather_object(objects, {"rank": rank, "text": "x" * (100 + rank)})
    assert objects == [{"rank": r, "text": "x" * (100 + r)} for r in range(2)], "all_gather_object"
    box = [{"from": rank}]
    dist.broadcast_object_list(box, src=0)
    assert box == [{"from": 0}], f"broadcast_object_list: {box}"
    if held is not None:
        assert held.sum().item() == 7.0 * 3 * 1024 * 1024
    print(f"rank {rank} done", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    distributed.set_tpu_launch_env(nproc_per_node=2)
    torch.multiprocessing.spawn(worker, nprocs=2)
"""


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def chip_count():
    try:
        return sum(path.name.isdigit() for path in pathlib.Path("/dev/tenstorrent").iterdir())
    except OSError:
        return 0


@pytest.mark.skipif(chip_count() < 2, reason="needs two chips")
def test_collectives_with_different_addresses(tmp_path):
    script = tmp_path / "collectives.py"
    script.write_text(CODE)
    # Each rank takes its own chip, in a mesh of both: not the runner's one chip.
    env = {k: v for k, v in os.environ.items() if k not in ("TT_VISIBLE_DEVICES", "TT_MESH_GRAPH_DESC_PATH")}
    env.update(LIBTT_TPU_COMPAT="1", JAX_PLATFORMS="cpu")
    env.setdefault("MASTER_ADDR", "127.0.0.1")
    env["MASTER_PORT"] = str(free_port())
    result = subprocess.run([sys.executable, str(script)], timeout=600, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-4000:]
    assert "rank 0 done" in result.stdout and "rank 1 done" in result.stdout
