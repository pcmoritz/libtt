#!/usr/bin/env python3
import argparse
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Run a bf16 JAX matmul on libtt and report TFLOP/s.")
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--validate", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def require_tiled(name, value):
    if value % 32 != 0:
        raise SystemExit(f"{name} must be a multiple of 32, got {value}")


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

    a_host = np.ones((args.m, args.k), dtype=np.float32)
    b_host = np.ones((args.k, args.n), dtype=np.float32)
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
        expected = np.float32(args.k)
        if not np.all(out == expected):
            max_abs = float(np.max(np.abs(out - expected)))
            raise SystemExit(f"validation failed: expected all {expected}, max_abs={max_abs}")

    ops = 2 * args.m * args.k * args.n
    best = min(times)
    median = statistics.median(times)
    print(f"best_ms: {best * 1e3:.3f}")
    print(f"median_ms: {median * 1e3:.3f}")
    print(f"peak_tflops: {ops / best / 1e12:.3f}")
    print(f"median_tflops: {ops / median / 1e12:.3f}")


if __name__ == "__main__":
    main()
