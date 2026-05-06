#!/usr/bin/env python3
import argparse

import jax
import jax.numpy as jnp
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Run a small bf16 feedforward JAX model on libtt.")
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--in-features", type=int, default=256)
    parser.add_argument("--hidden-features", type=int, default=512)
    parser.add_argument("--out-features", type=int, default=128)
    parser.add_argument("--validate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rtol", type=float, default=0.05)
    parser.add_argument("--atol", type=float, default=0.5)
    return parser.parse_args()


def require_tiled(name, value):
    if value % 32 != 0:
        raise SystemExit(f"{name} must be a multiple of 32, got {value}")


def make_inputs(batch, in_features, hidden_features, out_features):
    x_rows = (np.arange(batch, dtype=np.float32) % 13 - 6).reshape(batch, 1)
    x_cols = (np.arange(in_features, dtype=np.float32) % 17 - 8).reshape(1, in_features)
    x = (x_rows + x_cols) / np.float32(16.0)

    w1_rows = (np.arange(in_features, dtype=np.float32) % 11 - 5).reshape(in_features, 1)
    w1_cols = (np.arange(hidden_features, dtype=np.float32) % 19 - 9).reshape(1, hidden_features)
    w1 = (w1_rows - w1_cols) / np.float32(32.0)

    w2_rows = (np.arange(hidden_features, dtype=np.float32) % 7 - 3).reshape(hidden_features, 1)
    w2_cols = (np.arange(out_features, dtype=np.float32) % 23 - 11).reshape(1, out_features)
    w2 = (w2_rows + w2_cols) / np.float32(32.0)
    return x, w1, w2


def feedforward(x, w1, w2):
    return jax.nn.relu(x @ w1) @ w2


def validate_output(out, expected, rtol, atol):
    if not np.allclose(out, expected, rtol=rtol, atol=atol):
        diff = np.abs(out - expected)
        max_index = np.unravel_index(np.argmax(np.nan_to_num(diff, nan=np.inf)), diff.shape)
        raise SystemExit(
            "validation failed: "
            f"max_abs={float(diff[max_index])} at {max_index}, "
            f"got={float(out[max_index])}, expected={float(expected[max_index])}"
        )


def main():
    args = parse_args()
    require_tiled("batch", args.batch)
    require_tiled("in_features", args.in_features)
    require_tiled("hidden_features", args.hidden_features)
    require_tiled("out_features", args.out_features)

    devices = jax.devices()
    if not devices or devices[0].platform != "tt":
        raise SystemExit(f"expected tt backend, got devices={devices}")
    print(f"device: {devices[0]}")

    x_host, w1_host, w2_host = make_inputs(
        args.batch, args.in_features, args.hidden_features, args.out_features
    )
    x = jnp.asarray(x_host, dtype=jnp.bfloat16)
    w1 = jnp.asarray(w1_host, dtype=jnp.bfloat16)
    w2 = jnp.asarray(w2_host, dtype=jnp.bfloat16)

    compiled = jax.jit(feedforward)
    result = compiled(x, w1, w2).block_until_ready()

    if args.validate:
        expected = np.maximum(x_host @ w1_host, 0) @ w2_host
        validate_output(np.asarray(result, dtype=np.float32), expected, args.rtol, args.atol)

    print(f"shape: ({args.batch}, {args.in_features}) -> ({args.batch}, {args.out_features})")
    print("validation: ok" if args.validate else "validation: skipped")


if __name__ == "__main__":
    main()
