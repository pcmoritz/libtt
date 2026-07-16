#!/usr/bin/env python3
"""Collect the streaming tt-inference-server baseline used by the report.

The client uses one persistent loopback HTTP connection and records both time
to first token and token arrival times.  This gives TTIS the same measurement
scope as the current libtt benchmark: pure decode is measured between the
first and last generated token, while streaming end-to-end throughput includes
TTFT.  The first two requests are warm-ups; the next 32 form the analysis
window.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import statistics
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


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


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
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    encoded = json.dumps(request_payload, separators=(",", ":")).encode()
    manifest = {
        "schema_version": 2,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "endpoint": "/v1/completions",
        "request": request_payload,
        "warmups": args.warmups,
        "retained_samples": args.samples,
        "timing_scope": "loopback streaming client token-arrival clock",
        "definitions": {
            "ttft_s": "request send to first non-empty completion chunk",
            "total_s": "request send to final non-empty completion chunk",
            "decode_tps": "(completion_tokens - 1) / (last_token_time - first_token_time)",
            "e2e_tps": "completion_tokens / total_s",
            "itl_s": "arrival-time delta between consecutive non-empty completion chunks",
        },
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
                headers={
                    "Content-Type": "application/json",
                    "Accept": "text/event-stream",
                },
            )
            response = conn.getresponse()
            if response.status != 200:
                body = response.read().decode(errors="replace")
                raise RuntimeError(f"run {run_index}: HTTP {response.status}: {body}")

            chunk_times_ns: list[int] = []
            chunk_texts: list[str] = []
            usage: dict | None = None
            finish_reason: str | None = None
            while True:
                raw_line = response.readline()
                if not raw_line:
                    break
                line = raw_line.decode().strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    # Drain the terminating HTTP chunk before reusing the
                    # persistent connection for the next serial request.
                    response.read()
                    break
                event = json.loads(data)
                if event.get("usage"):
                    usage = event["usage"]
                choices = event.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                if choice.get("finish_reason") is not None:
                    finish_reason = choice["finish_reason"]
                text = choice.get("text") or ""
                if text:
                    chunk_times_ns.append(time.perf_counter_ns())
                    chunk_texts.append(text)

            if usage is None or not chunk_times_ns:
                raise RuntimeError(f"run {run_index}: incomplete streaming response")
            completion_tokens = int(usage["completion_tokens"])
            if completion_tokens != args.tokens:
                raise RuntimeError(
                    f"run {run_index}: expected {args.tokens} tokens, "
                    f"got {completion_tokens}"
                )
            if len(chunk_times_ns) != completion_tokens:
                raise RuntimeError(
                    f"run {run_index}: expected one non-empty chunk per token; "
                    f"got {len(chunk_times_ns)} chunks for {completion_tokens} tokens"
                )

            ttft_s = (chunk_times_ns[0] - start_ns) / 1e9
            total_s = (chunk_times_ns[-1] - start_ns) / 1e9
            decode_elapsed_s = (chunk_times_ns[-1] - chunk_times_ns[0]) / 1e9
            itls_s = [
                (right - left) / 1e9
                for left, right in zip(chunk_times_ns, chunk_times_ns[1:])
            ]
            phase = "warmup" if run_index <= args.warmups else "retained"
            record = {
                "schema_version": 2,
                "run_index": run_index,
                "phase": phase,
                "started_utc": started_utc,
                "prompt_tokens": usage["prompt_tokens"],
                "completion_tokens": completion_tokens,
                "stream_chunks": len(chunk_times_ns),
                "finish_reason": finish_reason,
                "ttft_s": ttft_s,
                "total_s": total_s,
                "decode_elapsed_s": decode_elapsed_s,
                "decode_tps": (completion_tokens - 1) / decode_elapsed_s,
                "e2e_tps": completion_tokens / total_s,
                "mean_itl_s": statistics.mean(itls_s),
                "median_itl_s": statistics.median(itls_s),
                "p95_itl_s": percentile(itls_s, 0.95),
                "min_itl_s": min(itls_s),
                "max_itl_s": max(itls_s),
                "completion_text_sha256_12": hashlib.sha256(
                    "".join(chunk_texts).encode()
                ).hexdigest()[:12],
                "chunk_texts": chunk_texts,
                "inter_token_latencies_s": itls_s,
            }
            output_path = args.output_dir / f"run_{run_index:02d}.json"
            output_path.write_text(json.dumps(record, indent=2) + "\n")
            print(
                f"{run_index:02d}/{total} {phase:8s} "
                f"TTFT={ttft_s:.6f}s total={total_s:.6f}s "
                f"decode={record['decode_tps']:.6f} tok/s",
                flush=True,
            )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
