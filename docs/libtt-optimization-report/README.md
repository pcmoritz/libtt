# libtt optimization report

This directory contains the report source, benchmark data, generated figures,
and publication artifacts.

## Contents

- `report.md`: canonical report source;
- `report.html`, `report.tex`, and `report.pdf`: generated report artifacts;
- `data/samples.csv` and `data/summary.csv`: 32-sample cumulative optimization
  measurements, including the final current-stack measurement;
- `data/foundation-ablation-*`: 64-sample baseline/complete-set comparison and
  32-sample leave-one-feature-out measurements;
- `data/current-kernel-samples.csv` and
  `data/current-kernel-summary.csv`: blocking and down-projection A/B data;
- `data/current-kernel-manifest.json`: current experiment provenance;
- `data/down-projection-profile-summary.csv`: per-operation profile summary;
- `data/latest-main-streaming-*`: direct prefill trace A/B data;
- `data/upstream-tt-inference-*`: matched streaming libtt and TTIS observations,
  same-clock comparison statistics, and manifest;
- `analyze.py`: statistics and SVG generation;
- `benchmark_upstream.py`: matched streaming benchmark collector;
- `figures/*.svg`: vector figures; and
- `Makefile` and `style.css`: report build files.

## Regenerate statistics and figures

The raw benchmark directories referenced by `analyze.py` must be available
under `/tmp`, including `/tmp/libtt-foundation-bench-20260716` for the
feature-attribution measurements. From the repository root, run:

```bash
/home/pcmoritz/sglang-jax/.venv/bin/python \
  docs/libtt-optimization-report/analyze.py
```

## Feature attribution data

The feature-attribution experiment uses the Qwen3-8B server command from the
repository README with `SGLANG_TT_TRACE_DECODE_ONLY=true`, which excludes the
later fixed-shape prefill trace. Each server run records two warmups followed
by 32 retained requests. The raw directory has two complete-feature-set runs,
two functional-baseline runs, and one run for each leave-one-feature-out
configuration. The baseline keeps the build-only NoC public-include patch.
Exact feature groups and source provenance are recorded in
`data/foundation-ablation-manifest.json`.

## Current libtt and TTIS comparison

The current comparison uses Qwen3-8B, a 128-token greedy completion, two
warmups, and 32 retained requests for each stack. Prefix caching is disabled.
Both measurements use the same persistent loopback streaming client and the
same Blackhole P150, with a device reset between stacks.

Collect the libtt samples after starting the SGLang-JAX server with the command
from the repository README:

```bash
/home/pcmoritz/sglang-jax/.venv/bin/python \
  docs/libtt-optimization-report/benchmark_upstream.py \
  --base-url http://127.0.0.1:31000 \
  --output-dir /tmp/libtt-report-rebased-20260717 \
  --warmups 2 --samples 32 --tokens 128 --timeout 1200 \
  --skip-server-metadata
```

Collect the TTIS samples after starting the official v0.10.0 server with
prefix caching disabled:

```bash
/home/pcmoritz/sglang-jax/.venv/bin/python \
  docs/libtt-optimization-report/benchmark_upstream.py \
  --base-url http://127.0.0.1:8000 \
  --output-dir /tmp/libtt-ttis-streaming-20260717 \
  --warmups 2 --samples 32 --tokens 128 --timeout 1200
```

The exact source revisions, binary digest, server image, request, and summary
statistics are recorded in `data/upstream-tt-inference-manifest.json`.

## Build the report

Build the self-contained HTML report:

```bash
make -C docs/libtt-optimization-report html
```

Build the LaTeX source:

```bash
make -C docs/libtt-optimization-report tex
```

Build the typeset PDF with vector figures:

```bash
make -C docs/libtt-optimization-report pdf
```

The PDF build requires Pandoc, XeLaTeX, and `rsvg-convert`. It uses `booktabs`,
`microtype`, and vector SVG conversion. Generated HTML, LaTeX, and PDF files are
stored with the source.
