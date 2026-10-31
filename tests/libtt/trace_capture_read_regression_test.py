"""Results read back while another thread captures traces keep their values.

Reading a tiled result back untilizes it on device, a program, and then reads
it. While a trace is captured, the command queue skips the programs it is
given, or records them into the trace, and rejects reads. A read from another
thread then returned what an earlier buffer had left (in SGLang-TorchTPU, now
and then the token of the step before) or aborted the process with "Reads are
not supported during trace capture".
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

    # Tiled results, which a read untilizes on device. Reading the first one
    # before the captures start puts the untilize program in the program cache.
    tiled = jax.jit(lambda x: x @ jnp.eye(32, dtype=jnp.float32) + 1)
    results = [tiled(jax.device_put(np.full((32, 32), i, np.float32))) for i in range(1000)]
    np.testing.assert_array_equal(np.asarray(results[0]), 1)

    stop = threading.Event()

    def capture():
        while not stop.is_set():
            for f, x in programs:
                f(x)
                time.sleep(0.002)

    thread = threading.Thread(target=capture)
    thread.start()
    wrong = []
    try:
        for i, result in enumerate(results[1:], 1):
            y = np.asarray(result)
            if not (y == i + 1).all():
                wrong.append((i, np.unique(y)[:3].tolist()))
    finally:
        stop.set()
        thread.join()
    assert not wrong, f"{len(wrong)} of 999 reads were wrong, e.g. {wrong[:5]}"
