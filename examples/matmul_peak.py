#!/usr/bin/env python3
import argparse
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Run a bf16 JAX matmul on libtt and report TFLOP/s.")
    parser.add_argument("--m", type=int, default=512)
    parser.add_argument("--k", type=int, default=512)
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--validate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rtol", type=float, default=0.05)
    parser.add_argument("--atol", type=float, default=0.5)
    return parser.parse_args()


def require_tiled(name, value):
    if value % 32 != 0:
        raise SystemExit(f"{name} must be a multiple of 32, got {value}")


def make_inputs(m, k, n):
    row_pattern = (np.arange(m, dtype=np.float32) % 13 - 6).reshape(m, 1)
    k_pattern = (np.arange(k, dtype=np.float32) % 17 - 8).reshape(1, k)
    a_host = (row_pattern + k_pattern) / np.float32(16.0)

    k_pattern = (np.arange(k, dtype=np.float32) % 11 - 5).reshape(k, 1)
    col_pattern = (np.arange(n, dtype=np.float32) % 19 - 9).reshape(1, n)
    b_host = (k_pattern - col_pattern) / np.float32(16.0)
    return a_host.astype(np.float32), b_host.astype(np.float32)


def validate_output(name, out, expected, rtol, atol):
    if not np.allclose(out, expected, rtol=rtol, atol=atol):
        diff = np.abs(out - expected)
        max_index = np.unravel_index(np.argmax(np.nan_to_num(diff, nan=np.inf)), diff.shape)
        raise SystemExit(
            f"{name} validation failed: "
            f"max_abs={float(diff[max_index])} at {max_index}, "
            f"got={float(out[max_index])}, expected={float(expected[max_index])}"
        )


def main():
    args = parse_args()
    require_tiled("M", args.m)
    require_tiled("K", args.k)
    require_tiled("N", args.n)

    devices = jax.devices()
    if not devices or devices[0].platform != "tt":
        raise SystemExit(f"expected tt backend, got devices={devices}")
    print(f"device: {devices[0]}")
    print(f"shape: ({args.m}, {args.k}) @ ({args.k}, {args.n}) -> ({args.m}, {args.n})")

    a_host, b_host = make_inputs(args.m, args.k, args.n)
    a = jnp.asarray(a_host, dtype=jnp.bfloat16)
    b = jnp.asarray(b_host, dtype=jnp.bfloat16)

    matmul = jax.jit(lambda x, y: x @ y)
    result = matmul(a, b).block_until_ready()
    for _ in range(args.warmup):
        result = matmul(a, b).block_until_ready()

    times = []
    for _ in range(args.iters):
        start = time.perf_counter()
        result = matmul(a, b).block_until_ready()
        times.append(time.perf_counter() - start)

    if args.validate:
        out = np.asarray(result, dtype=np.float32)
        expected = a_host @ b_host
        validate_output("timed result", out, expected, args.rtol, args.atol)

        a_check = jnp.asarray(-a_host, dtype=jnp.bfloat16)
        check = matmul(a_check, b).block_until_ready()
        validate_output(
            "fresh result",
            np.asarray(check, dtype=np.float32),
            -expected,
            args.rtol,
            args.atol,
        )

    ops = 2 * args.m * args.k * args.n
    best = min(times)
    median = statistics.median(times)
    peak_tflops = ops / best / 1e12
    median_tflops = ops / median / 1e12
    print(f"best_ms: {best * 1e3:.3f}")
    print(f"median_ms: {median * 1e3:.3f}")
    print(f"peak_tflops: {peak_tflops:.6f}")
    print(f"median_tflops: {median_tflops:.6f}")
    print(f"peak_gflops: {peak_tflops * 1e3:.3f}")
    print(f"median_gflops: {median_tflops * 1e3:.3f}")


if __name__ == "__main__":
    main()
