#!/usr/bin/env python3
"""Reproduce the statistics and SVG figures in the libtt optimization report.

The benchmark driver intentionally records two warm-up requests before the
32-request analysis window.  This script consumes the raw SGLang JSON files,
checks the retained outputs, and writes publication-ready CSV/SVG artifacts.
It also analyzes the separately collected upstream tt-inference-server
baseline and the current SwiGLU-blocking/down-projection experiments.  The
final optimized configuration is included as the last stage of the cumulative
sequence using streaming end-to-end throughput; the report identifies the
clock change at that boundary.  A separate 2026-07-16 dataset decomposes the
foundation bundle with repeated baseline/full windows and leave-one-concept-out
builds.
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
UPSTREAM_RAW_DIR = Path("/tmp/libtt-ttis-streaming-20260715-retry")
LATEST_MAIN_STREAMING_DIR = Path("/tmp/libtt-prefill-rebench-20260715")
SWIGLU_BLOCK_2_DIR = Path("/tmp/libtt-swiglu-width2-final-20260715")
SWIGLU_BLOCK_4_DIR = Path("/tmp/libtt-down-110-final-20260715/disabled")
DOWN_PROJECTION_110_DIR = Path(
    "/tmp/libtt-down-110-final-fixed-20260715/enabled"
)
FOUNDATION_RAW_DIR = Path("/tmp/libtt-foundation-bench-20260716")


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

FINAL_VARIANT = Variant(
    "V10",
    "37d5460",
    "SwiGLU blocking and 110-core down projection",
    "MLP kernels",
    DOWN_PROJECTION_110_DIR,
)


@dataclass(frozen=True)
class FoundationConfiguration:
    name: str
    group: str
    omitted_concept: str


FOUNDATION_CONFIGURATIONS = (
    FoundationConfiguration(
        "baseline_compat",
        "baseline",
        "all foundation performance concepts",
    ),
    FoundationConfiguration(
        "baseline_b",
        "baseline",
        "all foundation performance concepts",
    ),
    FoundationConfiguration("full_a", "full foundation", "none"),
    FoundationConfiguration("full_b", "full foundation", "none"),
    FoundationConfiguration(
        "no_rmsnorm",
        "without RMSNorm recognition",
        "JAX RMSNorm recognition",
    ),
    FoundationConfiguration(
        "no_silu",
        "without SiLU call lowering",
        "SiLU call lowering",
    ),
    FoundationConfiguration(
        "no_rope",
        "without rank-3 RoPE fusion",
        "rank-3 decode RoPE fusion",
    ),
    FoundationConfiguration(
        "no_kv_dtype",
        "without KV-cache result typing",
        "KV-cache result typing",
    ),
    FoundationConfiguration(
        "no_bf8_activation",
        "without BF8 activation lowering",
        "BF8 activation lowering",
    ),
    FoundationConfiguration(
        "no_layout_admission",
        "without decode layout admission",
        "decode layout admission",
    ),
)


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


def append_final_streaming_stage(
    samples: list[dict],
    summaries: list[dict],
    baseline_mean: float,
    previous: list[float],
) -> list[float]:
    """Append the final kernel configuration using streaming E2E throughput."""

    paths = sorted(FINAL_VARIANT.raw_dir.glob("run_*.json"))
    records = [(path, json.loads(path.read_text())) for path in paths]
    retained = [
        (path, record)
        for path, record in records
        if record["phase"] == "retained"
    ][:N_SAMPLES]
    if len(retained) != N_SAMPLES:
        raise RuntimeError(
            f"{FINAL_VARIANT.label}: expected {N_SAMPLES} retained files, "
            f"found {len(retained)}"
        )

    throughputs: list[float] = []
    hashes: set[str] = set()
    for sample_index, (path, record) in enumerate(retained, 1):
        if (
            record["completion_tokens"] != TOKENS
            or record["stream_chunks"] != TOKENS
        ):
            raise RuntimeError(f"{path}: incomplete streaming response")
        latency = float(record["total_s"])
        throughput = TOKENS / latency
        output_hash = record["completion_text_sha256_12"]
        throughputs.append(throughput)
        hashes.add(output_hash)
        samples.append(
            {
                "variant": FINAL_VARIANT.label,
                "commit": FINAL_VARIANT.commit,
                "sample": sample_index,
                "source_file": path.name,
                "e2e_latency_s": f"{latency:.9f}",
                "throughput_tokens_s": f"{throughput:.9f}",
                "completion_tokens": record["completion_tokens"],
                "output_sha256_12": output_hash,
            }
        )
    if len(hashes) != 1:
        raise RuntimeError(
            f"{FINAL_VARIANT.label}: retained requests are not deterministic: "
            f"{hashes}"
        )

    mean = statistics.mean(throughputs)
    stddev = statistics.stdev(throughputs)
    ci_low, ci_high = stats.t.interval(
        0.95,
        len(throughputs) - 1,
        loc=mean,
        scale=stats.sem(throughputs),
    )
    summaries.append(
        {
            "variant": FINAL_VARIANT.label,
            "commit": FINAL_VARIANT.commit,
            "optimization": FINAL_VARIANT.optimization,
            "plot_label": FINAL_VARIANT.plot_label,
            "n": len(throughputs),
            "mean_tps": mean,
            "stddev_tps": stddev,
            "median_tps": statistics.median(throughputs),
            "min_tps": min(throughputs),
            "max_tps": max(throughputs),
            "ci_low_tps": ci_low,
            "ci_high_tps": ci_high,
            "incremental_speedup_pct": 100.0
            * (mean / statistics.mean(previous) - 1.0),
            "cumulative_speedup_pct": 100.0 * (mean / baseline_mean - 1.0),
            "welch_p_value": float(
                stats.ttest_ind(
                    throughputs,
                    previous,
                    equal_var=False,
                ).pvalue
            ),
            "output_sha256_12": next(iter(hashes)),
        }
    )
    return throughputs


def esc(value: object) -> str:
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def write_throughput_svg(summaries: list[dict]) -> None:
    width, height = 980, 520
    left, right, top, bottom = 90, 25, 40, 105
    plot_w, plot_h = width - left - right, height - top - bottom
    y_min, y_max = 15.0, 29.0

    def x(i: int) -> float:
        return left + i * plot_w / (len(summaries) - 1)

    def y(v: float) -> float:
        return top + (y_max - v) * plot_h / (y_max - y_min)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Helvetica,Arial,sans-serif;fill:#111}.axis{stroke:#555;stroke-width:1}.grid{stroke:#d0d0d0;stroke-width:1}.line{fill:none;stroke:#111;stroke-width:3}.ci{stroke:#666;stroke-width:2}.dot{fill:#111;stroke:white;stroke-width:1.5}.label{font-size:14px}.small{font-size:12px;fill:#444}.title{font-size:21px;font-weight:700}</style>',
        '<text x="90" y="27" class="title">Qwen3-8B end-to-end generation throughput</text>',
    ]
    for tick in range(15, 30):
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
    parts.append('<text x="90" y="505" class="small">32 retained requests per stage; two compile/warm-up requests excluded; V10 uses the streaming E2E clock.</text>')
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
        '<style>text{font-family:Helvetica,Arial,sans-serif;fill:#111}.axis{stroke:#555;stroke-width:1.2}.grid{stroke:#d0d0d0;stroke-width:1}.pos{fill:#333}.neg{fill:#aaa}.label{font-size:14px}.small{font-size:12px;fill:#444}.value{font-size:13px;font-weight:700}.title{font-size:21px;font-weight:700}</style>',
        '<text x="90" y="29" class="title">Incremental throughput change</text>',
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
        klass = "pos" if value >= 0 else "neg"
        value_y = yy - 8 if value >= 0 else yy + 18
        parts.extend([
            f'<rect x="{xx:.1f}" y="{rect_y:.1f}" width="{bw:.1f}" height="{max(rect_h, 1):.1f}" rx="3" class="{klass}"/>',
            f'<text x="{xx+bw/2:.1f}" y="{value_y:.1f}" text-anchor="middle" class="value">{value:+.2f}%</text>',
            f'<text x="{xx+bw/2:.1f}" y="{height-bottom+29}" text-anchor="middle" class="label">{esc(row["variant"])}</text>',
            f'<text x="{xx+bw/2:.1f}" y="{height-bottom+48}" text-anchor="middle" class="small">{esc(row["plot_label"])}</text>',
        ])
    parts.append(f'<text transform="translate(23 {top+plot_h/2}) rotate(-90)" text-anchor="middle" class="label">throughput change vs. preceding stage</text>')
    parts.append('<text x="90" y="482" class="small">Bars show the measured throughput change from the preceding stage.</text>')
    parts.append('</svg>')
    (FIGURE_DIR / "incremental-speedup.svg").write_text("\n".join(parts) + "\n")


def write_upstream_comparison_svg(comparison: dict) -> None:
    groups = (
        ("Pure decode", "decode_tps"),
        ("Streaming E2E", "e2e_tps"),
    )
    implementations = (
        ("libtt", "37d5460", comparison["libtt"], "libtt"),
        ("tt-inference-server", "v0.10.0", comparison["ttis"], "upstream"),
    )
    width, height = 980, 510
    left, right, top, bottom = 100, 30, 65, 115
    plot_w, plot_h = width - left - right, height - top - bottom
    y_min, y_max = 0.0, 30.0
    group_slot = plot_w / len(groups)
    bar_w = 112
    gap = 24

    def y(value: float) -> float:
        return top + (y_max - value) * plot_h / (y_max - y_min)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Helvetica,Arial,sans-serif;fill:#111}.axis{stroke:#555;stroke-width:1.2}.grid{stroke:#d0d0d0;stroke-width:1}.libtt{fill:#333}.upstream{fill:#aaa}.ci{stroke:#111;stroke-width:2.5}.label{font-size:14px}.small{font-size:12px;fill:#444}.value{font-size:15px;font-weight:700}.title{font-size:21px;font-weight:700}.legend{font-size:13px}</style>',
        '<text x="100" y="29" class="title">Same-clock Qwen3-8B streaming comparison</text>',
        '<rect x="608" y="16" width="16" height="16" rx="2" class="libtt"/>',
        '<text x="631" y="29" class="legend">libtt 37d5460</text>',
        '<rect x="760" y="16" width="16" height="16" rx="2" class="upstream"/>',
        '<text x="783" y="29" class="legend">TTIS v0.10.0</text>',
    ]
    for tick in (0, 5, 10, 15, 20, 25, 30):
        yy = y(tick)
        parts.append(
            f'<line x1="{left}" y1="{yy:.1f}" x2="{width-right}" '
            f'y2="{yy:.1f}" class="grid"/>'
        )
        parts.append(
            f'<text x="{left-12}" y="{yy+5:.1f}" text-anchor="end" '
            f'class="label">{tick}</text>'
        )
    parts.append(
        f'<line x1="{left}" y1="{top}" x2="{left}" '
        f'y2="{height-bottom}" class="axis"/>'
    )
    parts.append(
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" '
        f'y2="{height-bottom}" class="axis"/>'
    )
    for group_index, (group_label, metric) in enumerate(groups):
        group_center = left + (group_index + 0.5) * group_slot
        centers = (
            group_center - (bar_w + gap) / 2,
            group_center + (bar_w + gap) / 2,
        )
        for center, (_, _, values, klass) in zip(centers, implementations):
            metric_values = values[metric]
            bar_y = y(metric_values["mean"])
            ci_top = y(metric_values["ci_high"])
            ci_bottom = y(metric_values["ci_low"])
            parts.extend(
                [
                    f'<rect x="{center-bar_w/2:.1f}" y="{bar_y:.1f}" '
                    f'width="{bar_w}" height="{height-bottom-bar_y:.1f}" '
                    f'rx="4" class="{klass}"/>',
                    f'<line x1="{center:.1f}" y1="{ci_top:.1f}" '
                    f'x2="{center:.1f}" y2="{ci_bottom:.1f}" class="ci"/>',
                    f'<line x1="{center-7:.1f}" y1="{ci_top:.1f}" '
                    f'x2="{center+7:.1f}" y2="{ci_top:.1f}" class="ci"/>',
                    f'<line x1="{center-7:.1f}" y1="{ci_bottom:.1f}" '
                    f'x2="{center+7:.1f}" y2="{ci_bottom:.1f}" class="ci"/>',
                    f'<text x="{center:.1f}" y="{bar_y-12:.1f}" '
                    f'text-anchor="middle" class="value">'
                    f'{metric_values["mean"]:.3f}</text>',
                ]
            )
        parts.append(
            f'<text x="{group_center:.1f}" y="{height-bottom+29}" '
            f'text-anchor="middle" class="label">{esc(group_label)}</text>'
        )
    parts.append(
        f'<text transform="translate(25 {top+plot_h/2}) rotate(-90)" '
        'text-anchor="middle" class="label">tokens/s '
        '(mean and 95% t interval)</text>'
    )
    parts.append(
        '<text x="100" y="462" class="small">32 retained serial requests '
        'per implementation; two warm-ups; 128 generated tokens/request.</text>'
    )
    parts.append(
        '<text x="100" y="482" class="small">Both use the same loopback '
        'streaming clock: 127 inter-token intervals for decode and request-to-last-token for E2E.</text>'
    )
    parts.append('</svg>')
    (FIGURE_DIR / "upstream-comparison.svg").write_text(
        "\n".join(parts) + "\n"
    )


def analyze_upstream() -> dict:
    """Analyze TTIS and current libtt with the same streaming client clock."""

    manifest = json.loads((UPSTREAM_RAW_DIR / "manifest.json").read_text())
    request = manifest["request"]
    if (
        request["model"] != "Qwen/Qwen3-8B"
        or request["prompt"] != "The capital of France is"
        or request["temperature"] != 0
        or request["max_tokens"] != TOKENS
        or request.get("stream") is not True
    ):
        raise RuntimeError("upstream manifest does not match the report workload")

    metric_names = ("decode_tps", "e2e_tps", "ttft_s", "total_s")
    values: dict[str, dict[str, list[float]]] = {
        "ttis": {metric: [] for metric in metric_names},
        "libtt": {metric: [] for metric in metric_names},
    }
    hashes: dict[str, set[str]] = {"ttis": set(), "libtt": set()}
    samples: list[dict] = []

    ttis_paths = sorted(UPSTREAM_RAW_DIR.glob("run_*.json"))
    ttis_records = [json.loads(path.read_text()) for path in ttis_paths]
    ttis_retained = [
        (path, record)
        for path, record in zip(ttis_paths, ttis_records)
        if record["phase"] == "retained"
    ]
    if (
        len(ttis_records) != N_WARMUP + N_SAMPLES
        or len(ttis_retained) != N_SAMPLES
    ):
        raise RuntimeError(
            f"upstream: expected {N_WARMUP + N_SAMPLES} total and "
            f"{N_SAMPLES} retained files, found {len(ttis_records)} and "
            f"{len(ttis_retained)}"
        )

    for sample_index, (path, record) in enumerate(ttis_retained, 1):
        if (
            record["completion_tokens"] != TOKENS
            or record["stream_chunks"] != TOKENS
            or record["prompt_tokens"] != 5
        ):
            raise RuntimeError(f"{path}: incomplete or mismatched streaming response")
        row_values = {
            "decode_tps": float(record["decode_tps"]),
            "e2e_tps": TOKENS / float(record["total_s"]),
            "ttft_s": float(record["ttft_s"]),
            "total_s": float(record["total_s"]),
        }
        for metric, value in row_values.items():
            values["ttis"][metric].append(value)
        hashes["ttis"].add(record["completion_text_sha256_12"])
        samples.append(
            {
                "implementation": "upstream tt-inference-server",
                "release": "v0.10.0",
                "sample": sample_index,
                "source_file": path.name,
                "ttft_s": f'{row_values["ttft_s"]:.9f}',
                "total_s": f'{row_values["total_s"]:.9f}',
                "decode_tps": f'{row_values["decode_tps"]:.9f}',
                "e2e_tps": f'{row_values["e2e_tps"]:.9f}',
                "prompt_tokens": record["prompt_tokens"],
                "completion_tokens": record["completion_tokens"],
                "completion_text_sha256_12": record[
                    "completion_text_sha256_12"
                ],
            }
        )

    libtt_paths = sorted(DOWN_PROJECTION_110_DIR.glob("run_*.json"))
    libtt_records = [json.loads(path.read_text()) for path in libtt_paths]
    libtt_retained = [
        record for record in libtt_records if record["phase"] == "retained"
    ][:N_SAMPLES]
    if len(libtt_retained) != N_SAMPLES:
        raise RuntimeError(
            f"libtt comparison: expected {N_SAMPLES} retained files, "
            f"found {len(libtt_retained)}"
        )
    for record in libtt_retained:
        if (
            record["completion_tokens"] != TOKENS
            or record["stream_chunks"] != TOKENS
        ):
            raise RuntimeError("libtt comparison contains an incomplete response")
        row_values = {
            "decode_tps": float(record["decode_tps"]),
            "e2e_tps": TOKENS / float(record["total_s"]),
            "ttft_s": float(record["ttft_s"]),
            "total_s": float(record["total_s"]),
        }
        for metric, value in row_values.items():
            values["libtt"][metric].append(value)
        hashes["libtt"].add(record["completion_text_sha256_12"])

    if len(hashes["ttis"]) != 1 or len(hashes["libtt"]) != 1:
        raise RuntimeError(f"non-deterministic comparison outputs: {hashes}")

    described: dict[str, dict[str, dict[str, float]]] = {}
    for implementation in ("ttis", "libtt"):
        described[implementation] = {}
        for metric in metric_names:
            metric_values = values[implementation][metric]
            mean = statistics.mean(metric_values)
            ci_low, ci_high = stats.t.interval(
                0.95,
                len(metric_values) - 1,
                loc=mean,
                scale=stats.sem(metric_values),
            )
            described[implementation][metric] = {
                "mean": mean,
                "stddev": statistics.stdev(metric_values),
                "median": statistics.median(metric_values),
                "min": min(metric_values),
                "max": max(metric_values),
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
            }

    comparison: dict[str, float] = {}
    for metric in ("decode_tps", "e2e_tps"):
        comparison[f"libtt_{metric}_speedup_pct"] = 100.0 * (
            described["libtt"][metric]["mean"]
            / described["ttis"][metric]["mean"]
            - 1.0
        )
        comparison[f"{metric}_welch_p"] = float(
            stats.ttest_ind(
                values["libtt"][metric],
                values["ttis"][metric],
                equal_var=False,
            ).pvalue
        )
    for metric in ("ttft_s", "total_s"):
        comparison[f"libtt_{metric}_reduction_pct"] = 100.0 * (
            1.0
            - described["libtt"][metric]["mean"]
            / described["ttis"][metric]["mean"]
        )
        comparison[f"{metric}_welch_p"] = float(
            stats.ttest_ind(
                values["libtt"][metric],
                values["ttis"][metric],
                equal_var=False,
            ).pvalue
        )

    summary: dict[str, object] = {
        "implementation": "upstream tt-inference-server",
        "release": "v0.10.0",
        "server_commit": "4be69a67c7183bf76052d4a6f64a42ac93b71ac5",
        "container_image": "0.10.0-e867533-22be241",
        "tt_metal_commit": "e867533",
        "vllm_commit": "22be241",
        "n": N_SAMPLES,
    }
    for metric in metric_names:
        for statistic, value in described["ttis"][metric].items():
            summary[f"{statistic}_{metric}"] = value
    summary.update(comparison)
    summary["libtt_commit"] = "37d5460"
    summary["completion_text_sha256_12"] = next(iter(hashes["ttis"]))

    with (DATA_DIR / "upstream-tt-inference-samples.csv").open(
        "w", newline=""
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=list(samples[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(samples)
    with (DATA_DIR / "upstream-tt-inference-summary.csv").open(
        "w", newline=""
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=list(summary), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerow(summary)

    manifest["schema_version"] = 2
    manifest["timing_scope"] = "loopback streaming client token-arrival clock"
    manifest["definitions"] = {
        "ttft_s": "request send to first non-empty completion chunk",
        "total_s": "request send to final non-empty completion chunk",
        "decode_tps": "127 / (last token arrival - first token arrival)",
        "e2e_tps": "128 / (last token arrival - request send)",
    }
    manifest["tt_inference_server"] = {
        "release": "v0.10.0",
        "commit": "4be69a67c7183bf76052d4a6f64a42ac93b71ac5",
        "container_image": "ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.10.0-e867533-22be241",
        "prefix_caching": "disabled with --no-enable-prefix-caching",
        "hardware": "Blackhole P150",
    }
    manifest["libtt_comparison"] = {
        "branch": "agent/qwen3-swiglu-blocking-down-projection",
        "commit": "37d5460",
        "source_directory": str(DOWN_PROJECTION_110_DIR),
        "collector": "same loopback streaming token-arrival clock",
        "retained_samples": N_SAMPLES,
    }
    (DATA_DIR / "upstream-tt-inference-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return {
        "ttis": described["ttis"],
        "libtt": described["libtt"],
        "comparison": comparison,
    }


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


def write_foundation_ablation_svg(summary_rows: list[dict]) -> None:
    rows = [
        row
        for row in summary_rows
        if row["group"] not in ("baseline", "full foundation")
    ]
    width, height = 980, 500
    left, right, top, bottom = 310, 40, 55, 55
    plot_w, plot_h = width - left - right, height - top - bottom
    x_min, x_max = -17.0, 2.0
    row_h = plot_h / len(rows)

    def x(value: float) -> float:
        return left + (value - x_min) * plot_w / (x_max - x_min)

    zero = x(0.0)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Helvetica,Arial,sans-serif;fill:#111}.axis{stroke:#555;stroke-width:1.2}.grid{stroke:#d0d0d0;stroke-width:1}.loss{fill:#444}.gain{fill:#aaa}.label{font-size:14px}.small{font-size:12px;fill:#444}.value{font-size:13px;font-weight:700}.inverse{fill:#fff}.title{font-size:21px;font-weight:700}</style>',
        '<text x="40" y="31" class="title">Foundation concept leave-one-out results</text>',
    ]
    for tick in (-16, -12, -8, -4, 0):
        xx = x(tick)
        parts.append(
            f'<line x1="{xx:.1f}" y1="{top}" x2="{xx:.1f}" '
            f'y2="{height-bottom}" class="grid"/>'
        )
        parts.append(
            f'<text x="{xx:.1f}" y="{height-bottom+24}" '
            f'text-anchor="middle" class="label">{tick}%</text>'
        )
    parts.append(
        f'<line x1="{zero:.1f}" y1="{top}" x2="{zero:.1f}" '
        f'y2="{height-bottom}" class="axis"/>'
    )
    for index, row in enumerate(rows):
        value = float(row["throughput_change_vs_full_pct"])
        yy = top + index * row_h + row_h * 0.2
        bar_h = row_h * 0.6
        value_x = x(value)
        rect_x = min(zero, value_x)
        rect_w = max(abs(zero - value_x), 1.0)
        klass = "loss" if value < 0 else "gain"
        inverse = value < -10.0
        anchor = "start" if inverse or value >= 0 else "end"
        text_x = value_x + 8 if inverse or value >= 0 else value_x - 8
        value_class = "value inverse" if inverse else "value"
        parts.extend(
            [
                f'<text x="{left-15}" y="{yy+bar_h*0.68:.1f}" '
                f'text-anchor="end" class="label">{esc(row["omitted_concept"])}</text>',
                f'<rect x="{rect_x:.1f}" y="{yy:.1f}" width="{rect_w:.1f}" '
                f'height="{bar_h:.1f}" rx="3" class="{klass}"/>',
                f'<text x="{text_x:.1f}" y="{yy+bar_h*0.68:.1f}" '
                f'text-anchor="{anchor}" class="{value_class}">{value:+.2f}%</text>',
            ]
        )
    parts.append(
        '<text x="310" y="486" class="small">Change in mean throughput when the concept is omitted; 32 retained requests per omission.</text>'
    )
    parts.append("</svg>")
    (FIGURE_DIR / "foundation-ablation.svg").write_text(
        "\n".join(parts) + "\n"
    )


def analyze_foundation_ablation() -> None:
    values_by_group: dict[str, list[float]] = {}
    hashes_by_group: dict[str, set[str]] = {}
    sample_rows: list[dict] = []

    for config in FOUNDATION_CONFIGURATIONS:
        paths = sorted((FOUNDATION_RAW_DIR / config.name).glob("run_*.json"))
        if len(paths) != N_WARMUP + N_SAMPLES:
            raise RuntimeError(
                f"foundation {config.name}: expected {N_WARMUP + N_SAMPLES} "
                f"files, found {len(paths)}"
            )
        retained = paths[N_WARMUP:]
        config_hashes: set[str] = set()
        values_by_group.setdefault(config.group, [])
        hashes_by_group.setdefault(config.group, set())
        for sample_index, path in enumerate(retained, 1):
            payload = json.loads(path.read_text())
            meta = payload["meta_info"]
            if (
                meta["completion_tokens"] != TOKENS
                or len(payload["output_ids"]) != TOKENS
            ):
                raise RuntimeError(f"{path}: incomplete foundation response")
            latency = float(meta["e2e_latency"])
            throughput = TOKENS / latency
            output_hash = token_hash(payload["output_ids"])
            config_hashes.add(output_hash)
            values_by_group[config.group].append(throughput)
            hashes_by_group[config.group].add(output_hash)
            sample_rows.append(
                {
                    "configuration": config.name,
                    "group": config.group,
                    "omitted_concept": config.omitted_concept,
                    "sample": sample_index,
                    "source_file": path.name,
                    "e2e_latency_s": f"{latency:.9f}",
                    "throughput_tokens_s": f"{throughput:.9f}",
                    "completion_tokens": meta["completion_tokens"],
                    "output_sha256_12": output_hash,
                }
            )
        if len(config_hashes) != 1:
            raise RuntimeError(
                f"foundation {config.name}: non-deterministic output "
                f"{config_hashes}"
            )

    full_values = values_by_group["full foundation"]
    baseline_values = values_by_group["baseline"]
    full_mean = statistics.mean(full_values)
    baseline_mean = statistics.mean(baseline_values)
    group_order = list(dict.fromkeys(c.group for c in FOUNDATION_CONFIGURATIONS))
    omitted_by_group = {
        config.group: config.omitted_concept
        for config in FOUNDATION_CONFIGURATIONS
    }
    summary_rows: list[dict] = []
    for group in group_order:
        values = values_by_group[group]
        mean = statistics.mean(values)
        ci_low, ci_high = stats.t.interval(
            0.95,
            len(values) - 1,
            loc=mean,
            scale=stats.sem(values),
        )
        hashes = hashes_by_group[group]
        summary_rows.append(
            {
                "group": group,
                "omitted_concept": omitted_by_group[group],
                "n": len(values),
                "mean_tps": mean,
                "stddev_tps": statistics.stdev(values),
                "median_tps": statistics.median(values),
                "min_tps": min(values),
                "max_tps": max(values),
                "ci_low_tps": ci_low,
                "ci_high_tps": ci_high,
                "throughput_change_vs_full_pct": 100.0
                * (mean / full_mean - 1.0),
                "marginal_tps_loss_vs_full": full_mean - mean,
                "marginal_loss_share_of_total_gain_pct": 100.0
                * (full_mean - mean)
                / (full_mean - baseline_mean),
                "full_speedup_over_configuration_pct": 100.0
                * (full_mean / mean - 1.0),
                "speedup_vs_baseline_pct": 100.0
                * (mean / baseline_mean - 1.0),
                "output_sha256_12": ";".join(sorted(hashes)),
            }
        )

    with (DATA_DIR / "foundation-ablation-samples.csv").open(
        "w", newline=""
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=list(sample_rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(sample_rows)
    with (DATA_DIR / "foundation-ablation-summary.csv").open(
        "w", newline=""
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=list(summary_rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    manifest = {
        "schema_version": 1,
        "benchmark_date": "2026-07-16",
        "hardware": "Tenstorrent Blackhole P150",
        "firmware_observed_at_startup": "19.6.0",
        "libtt_foundation_commit": "9978a9b2017de067d0892f67811e5bb7ffc3cc7e",
        "functional_baseline_commit": "7482967",
        "baseline_compatibility_note": (
            "The baseline keeps the build-only NoC public-UMD include patch "
            "from 9978a9b; all six performance concepts are removed."
        ),
        "sglang_jax": {
            "path": "/home/pcmoritz/sglang-jax",
            "commit": "24eb823ed97e58ef83ab04b33cab8283ed003acb",
            "dirty_files_preserved": [
                "python/sgl_jax/srt/managers/tp_worker.py",
                "python/sgl_jax/srt/model_executor/model_runner.py",
                "python/sgl_jax/srt/layers/attention/tt_sdpa.py.bak",
                "python/sgl_jax/srt/utils/jax_utils.py.ttxla.bak",
                "python/sgl_jax/srt/utils/mesh_utils.py.ttxla.bak",
                "python/sgl_jax/srt/utils/weight_utils.py.ttxla.bak",
                "python/uv.lock",
                "tt-deps",
                "tt-metal-deps/",
            ],
        },
        "request": {
            "text": "The capital of France is",
            "temperature": 0,
            "max_new_tokens": TOKENS,
        },
        "sampling": {
            "warmups_per_configuration": N_WARMUP,
            "retained_per_configuration": N_SAMPLES,
            "retained_full_total": len(full_values),
            "retained_baseline_total": len(baseline_values),
            "trace_decode_only": True,
            "timing": "SGLang server-reported end-to-end latency",
        },
        "concepts": {
            "JAX RMSNorm recognition": [
                "tt_mlir_fuse_jax_rms_norm.patch"
            ],
            "SiLU call lowering": ["tt_mlir_lower_silu_call.patch"],
            "rank-3 decode RoPE fusion": [
                "tt_mlir_fuse_rank3_rope_decode.patch"
            ],
            "KV-cache result typing": [
                "tt_mlir_kv_cache_dtype_return_types.patch"
            ],
            "BF8 activation lowering": [
                "tt_xla_enable_bf8_activation_dtype_lowering.patch",
                "tt_mlir_single_chip_activation_dtype_lowering.patch",
            ],
            "decode layout admission": [
                "sdpa_decode_allow_l1_interleaved_q.patch",
                "layernorm_allow_single_core_height_sharded.patch",
            ],
        },
        "build_only_patch": "noc_debugging_use_public_umd_include.patch",
        "raw_directory": str(FOUNDATION_RAW_DIR),
    }
    (DATA_DIR / "foundation-ablation-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    write_foundation_ablation_svg(summary_rows)


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

    assert baseline_mean is not None
    assert previous is not None
    previous = append_final_streaming_stage(
        samples,
        summaries,
        baseline_mean,
        previous,
    )

    # Holm's step-down correction controls family-wise error across the
    # adjacent-stage comparisons while preserving the raw Welch p.
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
    upstream = analyze_upstream()
    write_upstream_comparison_svg(upstream)
    analyze_latest_main_streaming()
    analyze_current_kernel_experiments()
    analyze_foundation_ablation()


if __name__ == "__main__":
    main()
