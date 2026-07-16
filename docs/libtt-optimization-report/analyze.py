#!/usr/bin/env python3
"""Reproduce the statistics and SVG figures in the libtt optimization report.

The benchmark driver intentionally records two warm-up requests before the
32-request analysis window.  This script consumes the raw SGLang JSON files,
checks the retained outputs, and writes publication-ready CSV/SVG artifacts.
It also analyzes the separately collected upstream tt-inference-server
baseline and the current SwiGLU-blocking/down-projection experiments.  Those
experiments are not inserted into the older cumulative libtt sequence because
they use a direct streaming decode clock.
"""

from __future__ import annotations

import csv
import glob
import hashlib
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path

from scipy import stats


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
FIGURE_DIR = HERE / "figures"
N_WARMUP = 2
N_SAMPLES = 32
TOKENS = 128
UPSTREAM_RAW_DIR = Path("/tmp/libtt-ttis-baseline-20260715")
LATEST_MAIN_STREAMING_DIR = Path("/tmp/libtt-prefill-rebench-20260715")
SWIGLU_BLOCK_2_DIR = Path("/tmp/libtt-swiglu-width2-final-20260715")
SWIGLU_BLOCK_4_DIR = Path("/tmp/libtt-down-110-final-20260715/disabled")
DOWN_PROJECTION_110_DIR = Path(
    "/tmp/libtt-down-110-final-fixed-20260715/enabled"
)


@dataclass(frozen=True)
class Variant:
    label: str
    commit: str
    optimization: str
    plot_label: str
    raw_dir: Path


VARIANTS = [
    Variant(
        "V0",
        "7482967",
        "Documented serving baseline",
        "baseline",
        Path("/tmp/libtt-report-bench/v0_7482967"),
    ),
    Variant(
        "V1",
        "9978a9b",
        "Decode foundation bundle",
        "decode foundation",
        Path("/tmp/libtt-report-bench/v1_9978a9b"),
    ),
    Variant(
        "V2",
        "10459b5",
        "Expanded-SiLU recognition",
        "SiLU graph",
        Path("/tmp/libtt-report-bench/v2_10459b5"),
    ),
    Variant(
        "V3",
        "3fe072b",
        "QKV projection fusion",
        "QKV graph",
        Path("/tmp/libtt-report-bench/v3_3fe072b"),
    ),
    Variant(
        "V4",
        "83baa8d",
        "Two-way shared-LHS matmul fusion",
        "shared LHS",
        Path("/tmp/libtt-report-bench/v4_83baa8d"),
    ),
    Variant(
        "V5",
        "e534690",
        "Decode RMSNorm runtime sharding",
        "RMSNorm shard",
        Path("/tmp/libtt-branch-bench-20260711/main"),
    ),
    Variant(
        "V6",
        "a718685",
        "SiLU fused into binary multiply",
        "SiLU multiply",
        Path("/tmp/libtt-branch-bench-20260711/fused-silu-multiply"),
    ),
    Variant(
        "V7",
        "ce99831",
        "True matmul-SwiGLU epilogue",
        "Dst SwiGLU",
        Path("/tmp/libtt-branch-bench-20260711/matmul-swiglu-epilogue"),
    ),
    Variant(
        "V8",
        "caa5428",
        "Fallback-free prefill-capable epilogue",
        "one path",
        Path("/tmp/libtt-branch-bench-20260711/prefill-bf16-no-fallback"),
    ),
    Variant(
        "V9",
        "627a32d",
        "Latest main with traced short-prompt prefill",
        "prefill trace",
        Path("/tmp/libtt-report-main-627a32d/raw"),
    ),
]


def retained_json(variant: Variant) -> list[tuple[Path, dict]]:
    paths = [Path(p) for p in sorted(glob.glob(str(variant.raw_dir / "run_*.json")))]
    retained = paths[N_WARMUP : N_WARMUP + N_SAMPLES]
    if len(retained) != N_SAMPLES:
        raise RuntimeError(f"{variant.label}: expected {N_SAMPLES} retained files, found {len(retained)}")
    rows = [(path, json.loads(path.read_text())) for path in retained]
    for path, payload in rows:
        meta = payload["meta_info"]
        if meta["completion_tokens"] != TOKENS or len(payload["output_ids"]) != TOKENS:
            raise RuntimeError(f"{path}: request did not return {TOKENS} tokens")
    return rows


def token_hash(output_ids: list[int]) -> str:
    encoded = ",".join(map(str, output_ids)).encode()
    return hashlib.sha256(encoded).hexdigest()[:12]


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def esc(value: object) -> str:
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def write_throughput_svg(summaries: list[dict]) -> None:
    width, height = 980, 520
    left, right, top, bottom = 90, 25, 40, 105
    plot_w, plot_h = width - left - right, height - top - bottom
    y_min, y_max = 15.0, 27.5

    def x(i: int) -> float:
        return left + i * plot_w / (len(summaries) - 1)

    def y(v: float) -> float:
        return top + (y_max - v) * plot_h / (y_max - y_min)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Inter,Helvetica,Arial,sans-serif;fill:#172033}.axis{stroke:#718096;stroke-width:1}.grid{stroke:#dbe4ee;stroke-width:1}.line{fill:none;stroke:#006d77;stroke-width:4}.ci{stroke:#e29578;stroke-width:3}.dot{fill:#006d77;stroke:white;stroke-width:2}.label{font-size:14px}.small{font-size:12px;fill:#526174}.title{font-size:21px;font-weight:700}</style>',
        '<text x="90" y="27" class="title">Qwen3-8B end-to-end generation throughput</text>',
    ]
    for tick in range(15, 28):
        yy = y(tick)
        parts.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{width-right}" y2="{yy:.1f}" class="grid"/>')
        parts.append(f'<text x="{left-12}" y="{yy+5:.1f}" text-anchor="end" class="label">{tick}</text>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" class="axis"/>')
    parts.append(f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" class="axis"/>')
    points = " ".join(f'{x(i):.1f},{y(row["mean_tps"]):.1f}' for i, row in enumerate(summaries))
    parts.append(f'<polyline points="{points}" class="line"/>')
    for i, row in enumerate(summaries):
        xx = x(i)
        lo, hi = y(row["ci_low_tps"]), y(row["ci_high_tps"])
        parts.extend([
            f'<line x1="{xx:.1f}" y1="{hi:.1f}" x2="{xx:.1f}" y2="{lo:.1f}" class="ci"/>',
            f'<line x1="{xx-6:.1f}" y1="{hi:.1f}" x2="{xx+6:.1f}" y2="{hi:.1f}" class="ci"/>',
            f'<line x1="{xx-6:.1f}" y1="{lo:.1f}" x2="{xx+6:.1f}" y2="{lo:.1f}" class="ci"/>',
            f'<circle cx="{xx:.1f}" cy="{y(row["mean_tps"]):.1f}" r="7" class="dot"/>',
            f'<text x="{xx:.1f}" y="{height-bottom+28}" text-anchor="middle" class="label">{esc(row["variant"])}</text>',
            f'<text x="{xx:.1f}" y="{height-bottom+47}" text-anchor="middle" class="small">{esc(row["plot_label"])}</text>',
        ])
    parts.append(f'<text transform="translate(23 {top+plot_h/2}) rotate(-90)" text-anchor="middle" class="label">tokens/s (mean and 95% t interval)</text>')
    parts.append('<text x="90" y="505" class="small">32 retained requests per revision; two compile/warm-up requests excluded; 128 generated tokens/request.</text>')
    parts.append('</svg>')
    (FIGURE_DIR / "throughput.svg").write_text("\n".join(parts) + "\n")


def write_incremental_svg(summaries: list[dict]) -> None:
    rows = summaries[1:]
    width, height = 980, 500
    left, right, top, bottom = 90, 25, 45, 100
    plot_w, plot_h = width - left - right, height - top - bottom
    y_min, y_max = -3.0, 24.0
    bar_slot = plot_w / len(rows)

    def y(v: float) -> float:
        return top + (y_max - v) * plot_h / (y_max - y_min)

    zero = y(0)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Inter,Helvetica,Arial,sans-serif;fill:#172033}.axis{stroke:#718096;stroke-width:1.2}.grid{stroke:#dbe4ee;stroke-width:1}.pos{fill:#006d77}.neg{fill:#c84b31}.neutral{fill:#8395a7}.label{font-size:14px}.small{font-size:12px;fill:#526174}.value{font-size:13px;font-weight:700}.title{font-size:21px;font-weight:700}</style>',
        '<text x="90" y="29" class="title">Incremental effect of each cumulative revision</text>',
    ]
    for tick in (-2, 0, 4, 8, 12, 16, 20, 24):
        yy = y(tick)
        parts.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{width-right}" y2="{yy:.1f}" class="grid"/>')
        parts.append(f'<text x="{left-12}" y="{yy+5:.1f}" text-anchor="end" class="label">{tick}%</text>')
    parts.append(f'<line x1="{left}" y1="{zero:.1f}" x2="{width-right}" y2="{zero:.1f}" class="axis"/>')
    for i, row in enumerate(rows):
        value = row["incremental_speedup_pct"]
        xx = left + i * bar_slot + bar_slot * 0.18
        bw = bar_slot * 0.64
        yy = y(value)
        rect_y, rect_h = min(yy, zero), abs(zero - yy)
        klass = "neutral" if row["holm_adjusted_p"] >= 0.05 else ("pos" if value >= 0 else "neg")
        value_y = yy - 8 if value >= 0 else yy + 18
        parts.extend([
            f'<rect x="{xx:.1f}" y="{rect_y:.1f}" width="{bw:.1f}" height="{max(rect_h, 1):.1f}" rx="3" class="{klass}"/>',
            f'<text x="{xx+bw/2:.1f}" y="{value_y:.1f}" text-anchor="middle" class="value">{value:+.2f}%</text>',
            f'<text x="{xx+bw/2:.1f}" y="{height-bottom+29}" text-anchor="middle" class="label">{esc(row["variant"])}</text>',
            f'<text x="{xx+bw/2:.1f}" y="{height-bottom+48}" text-anchor="middle" class="small">{esc(row["plot_label"])}</text>',
        ])
    parts.append(f'<text transform="translate(23 {top+plot_h/2}) rotate(-90)" text-anchor="middle" class="label">throughput change vs. preceding revision</text>')
    parts.append('<text x="90" y="482" class="small">Teal/red: Holm-adjusted p&lt;0.05; gray: not statistically distinguishable from the preceding revision.</text>')
    parts.append('</svg>')
    (FIGURE_DIR / "incremental-speedup.svg").write_text("\n".join(parts) + "\n")


def write_upstream_comparison_svg(summaries: list[dict], upstream: dict) -> None:
    rows = [
        {
            "label": "libtt V0",
            "detail": summaries[0]["commit"],
            "mean": summaries[0]["mean_tps"],
            "low": summaries[0]["ci_low_tps"],
            "high": summaries[0]["ci_high_tps"],
            "class": "baseline",
        },
        {
            "label": f'libtt {summaries[-1]["variant"]}',
            "detail": summaries[-1]["commit"],
            "mean": summaries[-1]["mean_tps"],
            "low": summaries[-1]["ci_low_tps"],
            "high": summaries[-1]["ci_high_tps"],
            "class": "libtt",
        },
        {
            "label": "tt-inference-server",
            "detail": "v0.10.0",
            "mean": upstream["mean_tps"],
            "low": upstream["ci_low_tps"],
            "high": upstream["ci_high_tps"],
            "class": "upstream",
        },
    ]
    width, height = 980, 510
    left, right, top, bottom = 100, 30, 45, 120
    plot_w, plot_h = width - left - right, height - top - bottom
    y_min, y_max = 0.0, 27.0
    slot = plot_w / len(rows)

    def y(value: float) -> float:
        return top + (y_max - value) * plot_h / (y_max - y_min)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Inter,Helvetica,Arial,sans-serif;fill:#172033}.axis{stroke:#718096;stroke-width:1.2}.grid{stroke:#dbe4ee;stroke-width:1}.baseline{fill:#8395a7}.libtt{fill:#006d77}.upstream{fill:#e29578}.ci{stroke:#172033;stroke-width:3}.label{font-size:14px}.small{font-size:12px;fill:#526174}.value{font-size:16px;font-weight:700}.title{font-size:21px;font-weight:700}</style>',
        '<text x="100" y="29" class="title">Same-prompt 128-token serving comparison</text>',
    ]
    for tick in (0, 5, 10, 15, 20, 25):
        yy = y(tick)
        parts.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{width-right}" y2="{yy:.1f}" class="grid"/>')
        parts.append(f'<text x="{left-12}" y="{yy+5:.1f}" text-anchor="end" class="label">{tick}</text>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" class="axis"/>')
    parts.append(f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" class="axis"/>')
    for i, row in enumerate(rows):
        center = left + (i + 0.5) * slot
        bar_w = slot * 0.52
        bar_y = y(row["mean"])
        ci_top, ci_bottom = y(row["high"]), y(row["low"])
        parts.extend([
            f'<rect x="{center-bar_w/2:.1f}" y="{bar_y:.1f}" width="{bar_w:.1f}" height="{height-bottom-bar_y:.1f}" rx="4" class="{row["class"]}"/>',
            f'<line x1="{center:.1f}" y1="{ci_top:.1f}" x2="{center:.1f}" y2="{ci_bottom:.1f}" class="ci"/>',
            f'<line x1="{center-7:.1f}" y1="{ci_top:.1f}" x2="{center+7:.1f}" y2="{ci_top:.1f}" class="ci"/>',
            f'<line x1="{center-7:.1f}" y1="{ci_bottom:.1f}" x2="{center+7:.1f}" y2="{ci_bottom:.1f}" class="ci"/>',
            f'<text x="{center:.1f}" y="{bar_y-13:.1f}" text-anchor="middle" class="value">{row["mean"]:.3f}</text>',
            f'<text x="{center:.1f}" y="{height-bottom+29}" text-anchor="middle" class="label">{esc(row["label"])}</text>',
            f'<text x="{center:.1f}" y="{height-bottom+48}" text-anchor="middle" class="small">{esc(row["detail"])}</text>',
        ])
    parts.append(f'<text transform="translate(25 {top+plot_h/2}) rotate(-90)" text-anchor="middle" class="label">tokens/s (mean and 95% t interval)</text>')
    parts.append('<text x="100" y="468" class="small">32 retained requests each. libtt: server-reported request latency; upstream: loopback client wall clock.</text>')
    parts.append('<text x="100" y="488" class="small">The upstream interval therefore includes a small amount of HTTP and response-serialization overhead.</text>')
    parts.append('</svg>')
    (FIGURE_DIR / "upstream-comparison.svg").write_text("\n".join(parts) + "\n")


def analyze_upstream(summaries: list[dict]) -> dict:
    paths = sorted(UPSTREAM_RAW_DIR.glob("run_*.json"))
    records = [json.loads(path.read_text()) for path in paths]
    retained = [(path, row) for path, row in zip(paths, records) if row["phase"] == "retained"]
    if len(records) != N_WARMUP + N_SAMPLES or len(retained) != N_SAMPLES:
        raise RuntimeError(
            f"upstream: expected {N_WARMUP + N_SAMPLES} total and {N_SAMPLES} retained files, "
            f"found {len(records)} and {len(retained)}"
        )

    samples: list[dict] = []
    throughputs: list[float] = []
    latencies: list[float] = []
    hashes: set[str] = set()
    for sample_index, (path, record) in enumerate(retained, 1):
        request = record["request"]
        response = record["response"]
        completion_tokens = response["usage"]["completion_tokens"]
        if (
            request["model"] != "Qwen/Qwen3-8B"
            or request["prompt"] != "The capital of France is"
            or request["temperature"] != 0
            or request["max_tokens"] != TOKENS
            or completion_tokens != TOKENS
        ):
            raise RuntimeError(f"{path}: request does not match the report workload")
        latency = float(record["client_latency_s"])
        throughput = TOKENS / latency
        output_hash = text_hash(response["choices"][0]["text"])
        latencies.append(latency)
        throughputs.append(throughput)
        hashes.add(output_hash)
        samples.append({
            "implementation": "upstream tt-inference-server",
            "release": "v0.10.0",
            "sample": sample_index,
            "source_file": path.name,
            "client_latency_s": f"{latency:.9f}",
            "throughput_tokens_s": f"{throughput:.9f}",
            "prompt_tokens": response["usage"]["prompt_tokens"],
            "completion_tokens": completion_tokens,
            "completion_text_sha256_12": output_hash,
        })
    if len(hashes) != 1:
        raise RuntimeError(f"upstream: retained completions are not deterministic: {hashes}")

    mean = statistics.mean(throughputs)
    ci_low, ci_high = stats.t.interval(
        0.95, len(throughputs) - 1, loc=mean, scale=stats.sem(throughputs)
    )
    v0 = summaries[0]
    latest = summaries[-1]
    v0_values = [float(row["throughput_tokens_s"]) for row in samples_for_variant("V0")]
    latest_values = [
        float(row["throughput_tokens_s"])
        for row in samples_for_variant(latest["variant"])
    ]
    summary = {
        "implementation": "upstream tt-inference-server",
        "release": "v0.10.0",
        "server_commit": "4be69a67c7183bf76052d4a6f64a42ac93b71ac5",
        "container_image": "0.10.0-e867533-22be241",
        "tt_metal_commit": "e867533",
        "vllm_commit": "22be241",
        "n": len(throughputs),
        "mean_tps": mean,
        "stddev_tps": statistics.stdev(throughputs),
        "median_tps": statistics.median(throughputs),
        "min_tps": min(throughputs),
        "max_tps": max(throughputs),
        "ci_low_tps": ci_low,
        "ci_high_tps": ci_high,
        "mean_latency_s": statistics.mean(latencies),
        "stddev_latency_s": statistics.stdev(latencies),
        "upstream_speedup_vs_libtt_v0_pct": 100.0 * (mean / v0["mean_tps"] - 1.0),
        "upstream_speedup_vs_libtt_latest_pct": 100.0
        * (mean / latest["mean_tps"] - 1.0),
        "welch_p_vs_libtt_v0": float(stats.ttest_ind(throughputs, v0_values, equal_var=False).pvalue),
        "welch_p_vs_libtt_latest": float(
            stats.ttest_ind(throughputs, latest_values, equal_var=False).pvalue
        ),
        "completion_text_sha256_12": next(iter(hashes)),
    }
    with (DATA_DIR / "upstream-tt-inference-samples.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(samples[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(samples)
    with (DATA_DIR / "upstream-tt-inference-summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary), lineterminator="\n")
        writer.writeheader()
        writer.writerow(summary)
    manifest = json.loads((UPSTREAM_RAW_DIR / "manifest.json").read_text())
    if "server_version" in manifest:
        # The /version endpoint belongs to the embedded vLLM API server.
        manifest["vllm_api_version"] = manifest.pop("server_version")
    manifest["tt_inference_server"] = {
        "release": "v0.10.0",
        "commit": "4be69a67c7183bf76052d4a6f64a42ac93b71ac5",
        "container_image": "ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.10.0-e867533-22be241",
        "prefix_caching": "disabled with --no-enable-prefix-caching",
        "hardware": "Blackhole P150",
    }
    (DATA_DIR / "upstream-tt-inference-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return summary


def analyze_latest_main_streaming() -> None:
    metric_names = (
        "ttft_s",
        "total_s",
        "decode_tps",
        "mean_itl_s",
        "p95_itl_s",
    )
    by_config: dict[str, dict[str, list[float]]] = {}
    sample_rows: list[dict] = []
    hashes: dict[str, set[str]] = {}

    for config, trace_decode_only in (("baseline", True), ("optimized", False)):
        paths = sorted((LATEST_MAIN_STREAMING_DIR / config).glob("run_*.json"))
        records = [json.loads(path.read_text()) for path in paths]
        retained = [
            (path, record)
            for path, record in zip(paths, records)
            if record["phase"] == "retained"
        ]
        if len(records) != N_WARMUP + N_SAMPLES or len(retained) != N_SAMPLES:
            raise RuntimeError(
                f"latest-main streaming {config}: expected {N_WARMUP + N_SAMPLES} "
                f"total and {N_SAMPLES} retained files, found {len(records)} and "
                f"{len(retained)}"
            )

        values = {metric: [] for metric in metric_names}
        hashes[config] = set()
        for sample_index, (path, record) in enumerate(retained, 1):
            if (
                record["completion_tokens"] != TOKENS
                or record["stream_chunks"] != TOKENS
            ):
                raise RuntimeError(f"{path}: incomplete streaming response")
            hashes[config].add(record["completion_text_sha256_12"])
            for metric in metric_names:
                values[metric].append(float(record[metric]))
            sample_rows.append(
                {
                    "configuration": config,
                    "trace_decode_only": str(trace_decode_only).lower(),
                    "sample": sample_index,
                    "source_file": path.name,
                    "ttft_s": f'{record["ttft_s"]:.9f}',
                    "total_s": f'{record["total_s"]:.9f}',
                    "decode_tps": f'{record["decode_tps"]:.9f}',
                    "mean_itl_s": f'{record["mean_itl_s"]:.9f}',
                    "p95_itl_s": f'{record["p95_itl_s"]:.9f}',
                    "completion_text_sha256_12": record[
                        "completion_text_sha256_12"
                    ],
                }
            )
        if len(hashes[config]) != 1:
            raise RuntimeError(
                f"latest-main streaming {config}: non-deterministic output {hashes[config]}"
            )
        by_config[config] = values

    if hashes["baseline"] != hashes["optimized"]:
        raise RuntimeError(
            "latest-main streaming trace configurations returned different completions"
        )

    summary_rows: list[dict] = []
    for config, trace_decode_only in (("baseline", True), ("optimized", False)):
        row: dict[str, object] = {
            "configuration": config,
            "trace_decode_only": str(trace_decode_only).lower(),
            "n": N_SAMPLES,
        }
        for metric in metric_names:
            values = by_config[config][metric]
            mean = statistics.mean(values)
            ci_low, ci_high = stats.t.interval(
                0.95,
                len(values) - 1,
                loc=mean,
                scale=stats.sem(values),
            )
            row[f"mean_{metric}"] = mean
            row[f"stddev_{metric}"] = statistics.stdev(values)
            row[f"ci_low_{metric}"] = ci_low
            row[f"ci_high_{metric}"] = ci_high
        row["completion_text_sha256_12"] = next(iter(hashes[config]))
        if config == "optimized":
            baseline_ttft = statistics.mean(by_config["baseline"]["ttft_s"])
            optimized_ttft = statistics.mean(by_config["optimized"]["ttft_s"])
            baseline_decode = statistics.mean(by_config["baseline"]["decode_tps"])
            optimized_decode = statistics.mean(by_config["optimized"]["decode_tps"])
            row["ttft_change_vs_baseline_pct"] = 100.0 * (
                optimized_ttft / baseline_ttft - 1.0
            )
            row["ttft_speedup_vs_baseline"] = baseline_ttft / optimized_ttft
            row["ttft_welch_p"] = float(
                stats.ttest_ind(
                    by_config["optimized"]["ttft_s"],
                    by_config["baseline"]["ttft_s"],
                    equal_var=False,
                ).pvalue
            )
            row["decode_change_vs_baseline_pct"] = 100.0 * (
                optimized_decode / baseline_decode - 1.0
            )
            row["decode_welch_p"] = float(
                stats.ttest_ind(
                    by_config["optimized"]["decode_tps"],
                    by_config["baseline"]["decode_tps"],
                    equal_var=False,
                ).pvalue
            )
        else:
            row["ttft_change_vs_baseline_pct"] = math.nan
            row["ttft_speedup_vs_baseline"] = math.nan
            row["ttft_welch_p"] = math.nan
            row["decode_change_vs_baseline_pct"] = math.nan
            row["decode_welch_p"] = math.nan
        summary_rows.append(row)

    with (DATA_DIR / "latest-main-streaming-samples.csv").open(
        "w", newline=""
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=list(sample_rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(sample_rows)
    with (DATA_DIR / "latest-main-streaming-summary.csv").open(
        "w", newline=""
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=list(summary_rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(summary_rows)


def analyze_current_kernel_experiments() -> None:
    """Analyze same-build streaming A/B tests for the current kernel patches."""

    configurations = (
        {
            "experiment": "swiglu_k_blocking",
            "configuration": "k_block_2",
            "raw_dir": SWIGLU_BLOCK_2_DIR,
            "swiglu_in0_block_w": 2,
            "down_projection_110_cores": False,
            "baseline": None,
        },
        {
            "experiment": "swiglu_k_blocking",
            "configuration": "k_block_4",
            "raw_dir": SWIGLU_BLOCK_4_DIR,
            "swiglu_in0_block_w": 4,
            "down_projection_110_cores": False,
            "baseline": "k_block_2",
        },
        {
            "experiment": "down_projection",
            "configuration": "110_core_specialization",
            "raw_dir": DOWN_PROJECTION_110_DIR,
            "swiglu_in0_block_w": 4,
            "down_projection_110_cores": True,
            "baseline": "k_block_4",
        },
    )
    metric_names = ("ttft_s", "total_s", "decode_tps", "e2e_tps")
    values_by_configuration: dict[str, dict[str, list[float]]] = {}
    hashes: dict[str, set[str]] = {}
    sample_rows: list[dict] = []

    for config in configurations:
        paths = sorted(config["raw_dir"].glob("run_*.json"))
        records = [json.loads(path.read_text()) for path in paths]
        retained = [
            (path, record)
            for path, record in zip(paths, records)
            if record["phase"] == "retained"
        ][:N_SAMPLES]
        if len(retained) != N_SAMPLES:
            raise RuntimeError(
                f'{config["configuration"]}: expected {N_SAMPLES} retained '
                f"files, found {len(retained)}"
            )

        values = {metric: [] for metric in metric_names}
        hashes[config["configuration"]] = set()
        for sample_index, (path, record) in enumerate(retained, 1):
            if (
                record["completion_tokens"] != TOKENS
                or record["stream_chunks"] != TOKENS
            ):
                raise RuntimeError(f"{path}: incomplete streaming response")
            e2e_tps = TOKENS / float(record["total_s"])
            row_values = {
                "ttft_s": float(record["ttft_s"]),
                "total_s": float(record["total_s"]),
                "decode_tps": float(record["decode_tps"]),
                "e2e_tps": e2e_tps,
            }
            for metric, value in row_values.items():
                values[metric].append(value)
            hashes[config["configuration"]].add(
                record["completion_text_sha256_12"]
            )
            sample_rows.append(
                {
                    "experiment": config["experiment"],
                    "configuration": config["configuration"],
                    "sample": sample_index,
                    "source_file": path.name,
                    "swiglu_in0_block_w": config["swiglu_in0_block_w"],
                    "down_projection_110_cores": str(
                        config["down_projection_110_cores"]
                    ).lower(),
                    "ttft_s": f'{row_values["ttft_s"]:.9f}',
                    "total_s": f'{row_values["total_s"]:.9f}',
                    "decode_tps": f'{row_values["decode_tps"]:.9f}',
                    "e2e_tps": f'{row_values["e2e_tps"]:.9f}',
                    "completion_text_sha256_12": record[
                        "completion_text_sha256_12"
                    ],
                }
            )
        if len(hashes[config["configuration"]]) != 1:
            raise RuntimeError(
                f'{config["configuration"]}: non-deterministic output '
                f'{hashes[config["configuration"]]}'
            )
        values_by_configuration[config["configuration"]] = values

    summary_rows: list[dict] = []
    for config in configurations:
        configuration = config["configuration"]
        row: dict[str, object] = {
            "experiment": config["experiment"],
            "configuration": configuration,
            "n": N_SAMPLES,
            "swiglu_in0_block_w": config["swiglu_in0_block_w"],
            "down_projection_110_cores": str(
                config["down_projection_110_cores"]
            ).lower(),
        }
        for metric in metric_names:
            values = values_by_configuration[configuration][metric]
            mean = statistics.mean(values)
            ci_low, ci_high = stats.t.interval(
                0.95,
                len(values) - 1,
                loc=mean,
                scale=stats.sem(values),
            )
            row[f"mean_{metric}"] = mean
            row[f"stddev_{metric}"] = statistics.stdev(values)
            row[f"ci_low_{metric}"] = ci_low
            row[f"ci_high_{metric}"] = ci_high
        row["completion_text_sha256_12"] = next(iter(hashes[configuration]))
        baseline = config["baseline"]
        if baseline is None:
            row["baseline_configuration"] = ""
            row["decode_change_pct"] = math.nan
            row["decode_welch_p"] = math.nan
            row["e2e_change_pct"] = math.nan
            row["e2e_welch_p"] = math.nan
        else:
            row["baseline_configuration"] = baseline
            for metric in ("decode", "e2e"):
                metric_name = f"{metric}_tps"
                current_values = values_by_configuration[configuration][
                    metric_name
                ]
                baseline_values = values_by_configuration[baseline][metric_name]
                row[f"{metric}_change_pct"] = 100.0 * (
                    statistics.mean(current_values)
                    / statistics.mean(baseline_values)
                    - 1.0
                )
                row[f"{metric}_welch_p"] = float(
                    stats.ttest_ind(
                        current_values, baseline_values, equal_var=False
                    ).pvalue
                )
        summary_rows.append(row)

    with (DATA_DIR / "current-kernel-samples.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(sample_rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(sample_rows)
    with (DATA_DIR / "current-kernel-summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(summary_rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(summary_rows)


def samples_for_variant(variant: str) -> list[dict]:
    with (DATA_DIR / "samples.csv").open(newline="") as f:
        return [row for row in csv.DictReader(f) if row["variant"] == variant]


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    samples: list[dict] = []
    summaries: list[dict] = []
    previous: list[float] | None = None
    baseline_mean: float | None = None

    for variant in VARIANTS:
        retained = retained_json(variant)
        throughputs: list[float] = []
        hashes: set[str] = set()
        for sample_index, (path, payload) in enumerate(retained, 1):
            latency = float(payload["meta_info"]["e2e_latency"])
            throughput = TOKENS / latency
            output_hash = token_hash(payload["output_ids"])
            throughputs.append(throughput)
            hashes.add(output_hash)
            samples.append({
                "variant": variant.label,
                "commit": variant.commit,
                "sample": sample_index,
                "source_file": path.name,
                "e2e_latency_s": f"{latency:.9f}",
                "throughput_tokens_s": f"{throughput:.9f}",
                "completion_tokens": payload["meta_info"]["completion_tokens"],
                "output_sha256_12": output_hash,
            })
        if len(hashes) != 1:
            raise RuntimeError(f"{variant.label}: retained requests are not token-deterministic: {hashes}")

        mean = statistics.mean(throughputs)
        stddev = statistics.stdev(throughputs)
        ci_low, ci_high = stats.t.interval(
            0.95, len(throughputs) - 1, loc=mean, scale=stats.sem(throughputs)
        )
        if baseline_mean is None:
            baseline_mean = mean
        if previous is None:
            incremental = math.nan
            p_value = math.nan
        else:
            incremental = 100.0 * (mean / statistics.mean(previous) - 1.0)
            p_value = float(stats.ttest_ind(throughputs, previous, equal_var=False).pvalue)
        summaries.append({
            "variant": variant.label,
            "commit": variant.commit,
            "optimization": variant.optimization,
            "plot_label": variant.plot_label,
            "n": len(throughputs),
            "mean_tps": mean,
            "stddev_tps": stddev,
            "median_tps": statistics.median(throughputs),
            "min_tps": min(throughputs),
            "max_tps": max(throughputs),
            "ci_low_tps": ci_low,
            "ci_high_tps": ci_high,
            "incremental_speedup_pct": incremental,
            "cumulative_speedup_pct": 100.0 * (mean / baseline_mean - 1.0),
            "welch_p_value": p_value,
            "output_sha256_12": next(iter(hashes)),
        })
        previous = throughputs

    # Holm's step-down correction controls family-wise error across the nine
    # planned adjacent-revision comparisons while preserving the raw Welch p.
    tested = sorted(summaries[1:], key=lambda row: row["welch_p_value"])
    running_max = 0.0
    for rank, row in enumerate(tested):
        adjusted = min(1.0, (len(tested) - rank) * row["welch_p_value"])
        running_max = max(running_max, adjusted)
        row["holm_adjusted_p"] = running_max
    summaries[0]["holm_adjusted_p"] = math.nan

    with (DATA_DIR / "samples.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(samples[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(samples)
    with (DATA_DIR / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summaries)

    write_throughput_svg(summaries)
    write_incremental_svg(summaries)
    upstream = analyze_upstream(summaries)
    write_upstream_comparison_svg(summaries, upstream)
    analyze_latest_main_streaming()
    analyze_current_kernel_experiments()


if __name__ == "__main__":
    main()
