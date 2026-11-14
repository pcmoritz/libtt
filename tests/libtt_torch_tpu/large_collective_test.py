"""Collectives larger than the symmetric DRAM region run in pieces.

Collectives between processes run with their buffers in a DRAM region every
process reserves the same way (LIBTT_SYMMETRIC_DRAM_MB). An all-gather's
result, or a reduce-scatter's buffers (two times its input and its result),
that would overflow it run on pieces of the input split along another
dimension, and the pieces' results are concatenated: here a 64 MB region per
chip and 64 MB inputs, which take three pieces of 86, 86 and 84 tiles.
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

ROWS, COLUMNS = 2048, 8192


def worker(rank):
    # torch_tpu binds a process to the chip of its local rank.
    os.environ["RANK"] = os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = os.environ["LOCAL_WORLD_SIZE"] = "2"
    dist.init_process_group("tpu_dist", rank=rank, world_size=2)
    values = torch.arange(ROWS * COLUMNS, dtype=torch.float32).reshape(ROWS, COLUMNS) % 251
    x = (values + 1000 * rank).to("tpu")

    gathered = torch.empty(2 * ROWS, COLUMNS, device="tpu")
    dist.all_gather_into_tensor(gathered, x)
    assert torch.equal(gathered.cpu(), torch.cat([values, values + 1000])), "all-gather"

    scattered = torch.empty(ROWS // 2, COLUMNS, device="tpu")
    dist.reduce_scatter_tensor(scattered, x)
    want = (2 * values + 1000)[rank * ROWS // 2 : (rank + 1) * ROWS // 2]
    assert torch.equal(scattered.cpu(), want), "reduce-scatter"
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
def test_collectives_larger_than_symmetric_region(tmp_path):
    script = tmp_path / "collectives.py"
    script.write_text(CODE)
    # Each rank takes its own chip, in a mesh of both: not the runner's one chip.
    env = {k: v for k, v in os.environ.items() if k not in ("TT_VISIBLE_DEVICES", "TT_MESH_GRAPH_DESC_PATH")}
    env.update(LIBTT_TPU_COMPAT="1", JAX_PLATFORMS="cpu", LIBTT_SYMMETRIC_DRAM_MB="64")
    env.setdefault("MASTER_ADDR", "127.0.0.1")
    env["MASTER_PORT"] = str(free_port())
    result = subprocess.run([sys.executable, str(script)], timeout=600, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-4000:]
    assert "rank 0 done" in result.stdout and "rank 1 done" in result.stdout
