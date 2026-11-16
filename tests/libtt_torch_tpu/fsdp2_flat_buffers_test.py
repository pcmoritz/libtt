"""PyTorch's FSDP2 on two chips, one process each, against the same run on CPU.

FSDP2 gathers parameters into flat rank-1 buffers and reduces gradients from
[world size, N] ones, and views each parameter in its own shape from them. In
tiles, those buffers' rows are padded to 32; libtt stores them as [N / 32, 32]
and programs take them that way (ttir-dense-flat). Here the embedding's flat
buffers are 512 MB in BF16 and 1 GB in FP32; padded, its FP32 gradients alone
would take 17 to 34 GB, more than a chip's DRAM can spare.
"""

import os
import pathlib
import re
import socket
import subprocess
import sys

import pytest

CODE = """
import os
import sys

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

CPU = sys.argv[1] == "cpu"
if not CPU:
    import torch_tpu  # noqa: F401
    from torch.tpu import distributed
D, V, LAYERS, TOKENS = 1024, 262144, 2, 256


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.RMSNorm(D)
        self.up = nn.Linear(D, 2 * D, bias=False)
        self.down = nn.Linear(2 * D, D, bias=False)

    def forward(self, x):
        return x + self.down(nn.functional.silu(self.up(self.norm(x))))


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(V, D)
        self.blocks = nn.ModuleList(Block() for _ in range(LAYERS))
        self.norm = nn.RMSNorm(D)

    def forward(self, tokens):
        x = self.embed(tokens)
        for block in self.blocks:
            x = block(x)
        # Tied output layer, as Qwen3-0.6B's: the embedding's gradient comes from both uses.
        return self.norm(x) @ self.embed.weight.t()


def worker(rank):
    os.environ["RANK"] = os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = os.environ["LOCAL_WORLD_SIZE"] = "2"
    device = "cpu" if CPU else "tpu"
    dist.init_process_group("gloo" if CPU else "tpu_dist", rank=rank, world_size=2)
    with torch.device("meta"):
        model = Model()
    mesh = init_device_mesh(device, (2,))
    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for block in model.blocks:
        fully_shard(block, mesh=mesh, mp_policy=policy)
    fully_shard(model, mesh=mesh, mp_policy=policy)
    model.to_empty(device=device)
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for parameter in model.parameters():
            full = torch.randn(parameter.shape, generator=generator) * 0.02
            parameter.to_local().copy_(full.chunk(2)[rank])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=False)
    tokens = torch.randint(0, V, (1, TOKENS), generator=torch.Generator().manual_seed(1 + rank)).to(device)
    for step in range(3):
        logits = model(tokens)
        loss = nn.functional.cross_entropy(logits.float().view(-1, V), tokens.view(-1))
        loss.backward()
        norm = torch.stack([p.grad.to_local().float().norm() for p in model.parameters()]).norm() ** 2
        dist.all_reduce(norm)
        optimizer.step()
        optimizer.zero_grad()
        print(f"rank {rank} step {step} loss {loss.item():.6f} grad_norm {norm.sqrt().item():.6f}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    if not CPU:
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


def run(script, device):
    env = {k: v for k, v in os.environ.items() if k not in ("TT_VISIBLE_DEVICES", "TT_MESH_GRAPH_DESC_PATH")}
    env.update(JAX_PLATFORMS="cpu", MASTER_ADDR="127.0.0.1", MASTER_PORT=str(free_port()))
    if device == "tt":
        env["LIBTT_TPU_COMPAT"] = "1"
    else:
        # The CPU reference doesn't load torch_tpu's device.
        env.pop("LIBTT_TPU_COMPAT", None)
        env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    result = subprocess.run(
        [sys.executable, str(script), device], timeout=900, env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-4000:]
    lines = re.findall(r"rank (\d) step (\d) loss (\S+) grad_norm (\S+)", result.stdout)
    assert len(lines) == 6, result.stdout[-2000:]
    return {(int(r), int(s)): (float(loss), float(norm)) for r, s, loss, norm in lines}


@pytest.mark.skipif(chip_count() < 2, reason="needs two chips")
def test_fsdp2_matches_cpu(tmp_path):
    script = tmp_path / "fsdp2.py"
    script.write_text(CODE)
    want = run(script, "cpu")
    got = run(script, "tt")
    for key, (loss, norm) in want.items():
        tt_loss, tt_norm = got[key]
        # BF16 parameters and activations: the TT and CPU roundings differ.
        assert abs(tt_loss - loss) < 2e-3 * abs(loss), (key, tt_loss, loss)
        assert abs(tt_norm - norm) < 2e-2 * abs(norm), (key, tt_norm, norm)
