"""Each process keeps its own system descriptor.

Every libtt process wrote its system descriptor to one file in the temporary
directory, which every compilation reads and a client removes when it exits.
A second process on the same machine, on another chip or with another build,
overwrote or removed it, and the first one's compilations failed with "system
desc schema mismatch" or found no file.
"""

import os
import subprocess
import sys
import tempfile

import jax
import jax.numpy as jnp


def test_system_descriptor_is_per_process():
    jax.devices()
    path = os.path.join(tempfile.gettempdir(), f"tt_pjrt_system_descriptor_{os.getpid()}")
    assert os.path.exists(path)
    # Compilations read it.
    assert int(jax.jit(lambda x: x + 1)(jnp.int32(1))) == 2


def test_a_process_without_a_chip_leaves_the_descriptor():
    """A process that finds no chip removes only its own descriptor when it exits."""
    jax.devices()
    # The child imports what this process does, libtt among it.
    env = dict(os.environ, TT_VISIBLE_DEVICES="99", PYTHONPATH=os.pathsep.join(filter(None, sys.path)))
    child = subprocess.run(
        [sys.executable, "-c", "import jax; jax.devices()"], env=env, capture_output=True, text=True
    )
    assert "Invalid device ID in TT_VISIBLE_DEVICES" in child.stderr, child.stderr[-2000:]
    assert int(jax.jit(lambda x: x * 3 + 7)(jnp.int32(2))) == 13
