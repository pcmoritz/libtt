"""Results read back while another thread captures traces have their values.

Reading a tiled result untilizes it on device first, a program. While a trace is
captured, the command queue skips the programs it gets or records them into the
trace, so a read from another thread that ran then got what an earlier buffer
left in the row-major one: in SGLang-TorchTPU, the token of the step before.
"""

import threading
import time

import jax
import jax.numpy as jnp
import numpy as np

TRACED = {"enable_trace": "true"}


def test_reads_during_trace_captures():
    # More traced programs than the trace cache holds (16), so that every call
    # captures its trace again.
    programs = []
    for k in range(20):

        def f(x, k=k):
            for _ in range(8):
                x = jnp.tanh(x * 1.01 + k)
            return x.sum(axis=0)

        x = jnp.ones((64 + 32 * k, 256), jnp.float32)
        programs.append((jax.jit(f, compiler_options=TRACED), x))
    for f, x in programs:
        np.asarray(f(x))

    stop = threading.Event()

    def capture():
        while not stop.is_set():
            for f, x in programs:
                f(x)
                time.sleep(0.002)

    add = jax.jit(lambda x: x + 1)
    thread = threading.Thread(target=capture)
    thread.start()
    wrong = []
    try:
        for i in range(2000):
            y = np.asarray(add(jax.device_put(np.full((32, 32), i, np.int32))))
            if not (y == i + 1).all():
                wrong.append((i, np.unique(y).tolist()))
    finally:
        stop.set()
        thread.join()
    assert not wrong, f"{len(wrong)} of 2000 reads were wrong, e.g. {wrong[:5]}"
