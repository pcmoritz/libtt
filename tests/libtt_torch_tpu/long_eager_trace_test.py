"""torch_tpu's long eager ops trace; its short ones don't.

torch_tpu compiles eager ops at XLA's O1 effort level, which libtt didn't trace:
an eager op runs once or on other buffers each time, and tracing them all made
SGLang-TorchTPU slower. But SGLang-TorchTPU samples with eager ops, and its
top-p filter's 30-round cutoff search lowers to hundreds of programs that cost
more to dispatch than to run. An O1 program of at least 128 TTIR ops now traces.
The process that runs the ops dumps its compiled programs (export_path), which
the test reads.
"""

import os
import subprocess
import sys

CODE = """
import torch, torch_tpu
import jax, jax.numpy as jnp
from jax import lax
from torch_tpu._internal import pallas

def search(probs: jax.Array, top_ps: jax.Array) -> jax.Array:
    def step(i, bits):
        candidate = bits | jnp.left_shift(1, 29 - i)
        threshold = lax.bitcast_convert_type(candidate, jnp.float32)
        mass = jnp.sum(jnp.where(probs >= threshold[:, None], probs, 0.0), axis=-1)
        return jnp.where(mass >= top_ps, candidate, bits)
    bits = lax.fori_loop(0, 30, step, jnp.zeros(probs.shape[0], jnp.int32))
    return lax.bitcast_convert_type(bits, jnp.float32)

op = pallas.jax_op("libtt_test::long_eager_search", search)
probs = torch.rand(2, 4096, device="tpu")
probs = probs / probs.sum(-1, keepdim=True)
top_ps = torch.full((2,), 0.9, device="tpu")
for _ in range(3):
    cutoff = op(probs, top_ps).cpu()
assert (cutoff > 0).all(), cutoff
assert (torch.ones(8, device="tpu") + 1).sum().item() == 16
"""


def test_long_eager_op_traces(tmp_path):
    env = dict(os.environ)
    options = env.get("TORCH_TPU_INTERNAL_XLA_OPTIONS", "")
    env["TORCH_TPU_INTERNAL_XLA_OPTIONS"] = f"{options} enable_trace=true export_path={tmp_path}".strip()
    subprocess.run([sys.executable, "-c", CODE], check=True, timeout=600, env=env)

    programs = []
    for name in sorted(os.listdir(tmp_path / "irs")):
        if name.startswith("ttnn_runtime_"):
            text = (tmp_path / "irs" / name).read_text()
            programs.append((text.count('"ttnn.'), "capture_or_execute_trace" in text))
    long_ones = [traced for ops, traced in programs if ops >= 128]
    short_ones = [traced for ops, traced in programs if ops < 32]
    assert long_ones and all(long_ones), programs
    assert short_ones and not any(short_ones), programs
