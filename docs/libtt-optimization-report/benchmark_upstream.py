#!/usr/bin/env python3
"""Collect the upstream tt-inference-server baseline used by the report.

The client uses one persistent loopback HTTP connection.  It records the
wall-clock interval from sending each request through reading the complete
OpenAI-compatible response, then validates that exactly 128 tokens were
generated.  The first two runs are labeled as warm-ups; the next 32 are the
analysis window.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse


PROMPT = "The capital of France is"
MODEL = "Qwen/Qwen3-8B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser.parse_args()


def connection(base_url: str, timeout: float) -> tuple[http.client.HTTPConnection, str]:
    parsed = urlparse(base_url)
    if parsed.scheme != "http" or not parsed.hostname:
        raise ValueError("--base-url must be an http:// URL")
    port = parsed.port or 80
    prefix = parsed.path.rstrip("/")
    return http.client.HTTPConnection(parsed.hostname, port, timeout=timeout), prefix


def get_json(base_url: str, path: str, timeout: float) -> dict:
    conn, prefix = connection(base_url, timeout)
    try:
        conn.request("GET", prefix + path)
        response = conn.getresponse()
        body = response.read()
        if response.status != 200:
            raise RuntimeError(f"GET {path} returned HTTP {response.status}: {body!r}")
        return json.loads(body)
    finally:
        conn.close()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if list(args.output_dir.glob("run_*.json")):
        raise RuntimeError(f"refusing to mix samples in non-empty {args.output_dir}")

    request_payload = {
        "model": MODEL,
        "prompt": PROMPT,
        "temperature": 0,
        "max_tokens": args.tokens,
    }
    encoded = json.dumps(request_payload, separators=(",", ":")).encode()
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "endpoint": "/v1/completions",
        "request": request_payload,
        "warmups": args.warmups,
        "retained_samples": args.samples,
        "timing_scope": "client wall clock: request send through complete response read",
        "vllm_api_version": get_json(args.base_url, "/version", args.timeout),
        "models": get_json(args.base_url, "/v1/models", args.timeout),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    conn, prefix = connection(args.base_url, args.timeout)
    total = args.warmups + args.samples
    try:
        for run_index in range(1, total + 1):
            started_utc = datetime.now(timezone.utc).isoformat()
            start_ns = time.perf_counter_ns()
            conn.request(
                "POST",
                prefix + "/v1/completions",
                body=encoded,
                headers={"Content-Type": "application/json"},
            )
            http_response = conn.getresponse()
            body = http_response.read()
            end_ns = time.perf_counter_ns()
            latency_s = (end_ns - start_ns) / 1e9
            if http_response.status != 200:
                raise RuntimeError(
                    f"run {run_index}: HTTP {http_response.status}: {body.decode(errors='replace')}"
                )

            response_payload = json.loads(body)
            completion_tokens = response_payload["usage"]["completion_tokens"]
            if completion_tokens != args.tokens:
                raise RuntimeError(
                    f"run {run_index}: expected {args.tokens} completion tokens, got {completion_tokens}"
                )
            completion_text = response_payload["choices"][0]["text"]
            phase = "warmup" if run_index <= args.warmups else "retained"
            record = {
                "schema_version": 1,
                "run_index": run_index,
                "phase": phase,
                "started_utc": started_utc,
                "client_latency_s": latency_s,
                "throughput_tokens_s": completion_tokens / latency_s,
                "completion_text_sha256_12": hashlib.sha256(
                    completion_text.encode()
                ).hexdigest()[:12],
                "request": request_payload,
                "response": response_payload,
            }
            output_path = args.output_dir / f"run_{run_index:02d}.json"
            output_path.write_text(json.dumps(record, indent=2) + "\n")
            print(
                f"{run_index:02d}/{total} {phase:8s} "
                f"{latency_s:.6f} s {completion_tokens / latency_s:.6f} tok/s",
                flush=True,
            )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
