# libtt optimization report

This directory contains the report source, benchmark data, generated figures,
and publication artifacts.

## Contents

- `report.md`: canonical report source;
- `report.html`, `report.tex`, and `report.pdf`: generated report artifacts;
- `data/samples.csv` and `data/summary.csv`: 32-sample optimization sequence,
  including the final streaming end-to-end MLP-kernel stage;
- `data/foundation-ablation-*`: fresh 64-sample baseline/full comparison and
  32-sample leave-one-concept-out foundation measurements;
- `data/current-kernel-samples.csv` and
  `data/current-kernel-summary.csv`: blocking and down-projection A/B data;
- `data/current-kernel-manifest.json`: current experiment provenance;
- `data/down-projection-profile-summary.csv`: per-operation profile summary;
- `data/latest-main-streaming-*`: direct prefill trace A/B data;
- `data/upstream-tt-inference-*`: streaming TTIS observations, same-clock
  comparison statistics, and manifest;
- `analyze.py`: statistics and SVG generation;
- `benchmark_upstream.py`: upstream TTIS collector;
- `figures/*.svg`: vector figures; and
- `Makefile` and `style.css`: report build files.

## Regenerate statistics and figures

The raw benchmark directories referenced by `analyze.py` must be available
under `/tmp`, including `/tmp/libtt-foundation-bench-20260716` for the
foundation decomposition. From the repository root, run:

```bash
/home/pcmoritz/sglang-jax/.venv/bin/python \
  docs/libtt-optimization-report/analyze.py
```

## Foundation decomposition

The foundation experiment uses commit `9978a9b` and the Qwen3-8B server
command from the repository README, with
`SGLANG_TT_TRACE_DECODE_ONLY=true` to match the original V0/V1 timing scope.
Each server run records two warmups followed by 32 retained requests. The raw
directory has two full-foundation runs, two functional-baseline runs, and one
run for each leave-one-concept-out build. The baseline keeps the build-only NoC
public-include patch. Exact patch groups and source provenance are recorded in
`data/foundation-ablation-manifest.json`.

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
versioned with the source.
