"""The compiler takes the system descriptor from memory.

Every libtt process wrote its system descriptor to one file in the temporary
directory, which every compilation read and a client removed when it went
away. A second process on the same machine overwrote or removed it: one on
another chip or with another build made the first one's compilations fail with
"system desc schema mismatch", and one whose client failed to start made them
find no file. The compiler now takes the descriptor from the module.
"""

import os
import re
import subprocess
import sys

import jax
import jax.numpy as jnp


def test_compilation_uses_the_system_descriptor(tmp_path):
    # Without a descriptor, the compiler would fall back to a mock one, whose
    # arch is Wormhole.
    device = jax.devices()[0]
    f = jax.jit(lambda x: x * 5 + 1, compiler_options={"export_path": str(tmp_path)})
    assert int(f(jnp.int32(2))) == 11
    [ir] = (tmp_path / "irs").glob("ttnn_[0-9]*.mlir")
    arches = set(re.findall(r"\barch = <(\w+)>", ir.read_text()))
    assert arches and all(arch.startswith(device.device_kind.lower()) for arch in arches), arches


def test_a_process_without_a_chip_leaves_compilation_alone():
    jax.devices()
    # The child imports what this process does, libtt among it.
    env = dict(os.environ, TT_VISIBLE_DEVICES="99", PYTHONPATH=os.pathsep.join(filter(None, sys.path)))
    child = subprocess.run(
        [sys.executable, "-c", "import jax; jax.devices()"], env=env, capture_output=True, text=True
    )
    assert "Invalid device ID in TT_VISIBLE_DEVICES" in child.stderr, child.stderr[-2000:]
    assert int(jax.jit(lambda x: x * 3 + 7)(jnp.int32(2))) == 13
