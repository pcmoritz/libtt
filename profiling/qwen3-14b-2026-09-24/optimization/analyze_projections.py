"""Map Qwen3-14B projections using QKV-head creation as an execution anchor.

Each layer has QKV, attention output, up/gate, and down projections. The
five-matmul interval between adjacent QKV anchors identifies the LM head.
Only complete 40-layer, 161-matmul cycles are included. Fused SwiGLU markers,
when present, must occupy all expected up/gate positions. Callback records
are deduplicated by chip/start/end. Category timings are inclusive.
"""
import argparse
from collections import defaultdict
import csv
import gzip
import json
from pathlib import Path
import statistics


def analyze(directory, window_ms):
    chips = defaultdict(dict)
    for path in sorted(directory.glob('*.csv*')):
        opener = gzip.open if path.suffix == '.gz' else open
        with opener(path, 'rt') as stream:
            for row in csv.DictReader(stream):
                chips[int(row['chip_id'])][(int(row['start']), int(row['end']))] = row
    report = {}
    for chip, unique in chips.items():
        all_rows = sorted(unique.values(), key=lambda r: int(r['start']))
        frequency = statistics.median((int(r['end']) - int(r['start'])) / float(r['duration_ns'])
                                      for r in all_rows if float(r['duration_ns']) > 0)
        cutoff = max(int(r['end']) for r in all_rows) - window_ms * 1e6 * frequency
        rows = [r for r in all_rows if 'matmul' in r['kernels']]
        # QKV-head creation immediately follows each layer's QKV projection.
        # A five-matmul interval between QKV projections contains the LM head.
        qkv_starts = []
        last_matmul = None
        matmul_index = -1
        for row in all_rows:
            if 'matmul' in row['kernels']:
                matmul_index += 1
                last_matmul = matmul_index
            if 'nlp_create_qkv_heads_decode' in row['kernels'] and last_matmul is not None:
                qkv_starts.append(last_matmul)
        heads = [b - 1 for a, b in zip(qkv_starts, qkv_starts[1:]) if b - a == 5]
        cycles = []
        boundaries = []
        for a, b in zip(heads, heads[1:]):
            cycle = rows[a + 1:b + 1]
            if len(cycle) != 161 or int(cycle[0]['start']) < cutoff:
                continue
            positions = [i for i, r in enumerate(cycle) if 'matmul_swiglu.cpp' in r['kernels']]
            if positions and positions != list(range(2, 160, 4)):
                continue
            cycles.append(cycle)
            boundaries.append((int(rows[a]["end"]), int(rows[b]["end"])))
        samples = defaultdict(list)
        sums = defaultdict(list)
        for cycle in cycles:
            totals = defaultdict(float)
            for i, row in enumerate(cycle):
                label = ('qkv', 'attention_output', 'up_gate', 'down')[i % 4] if i < 160 else 'lm_head'
                duration_us = float(row['duration_ns']) / 1000
                samples[label].append(duration_us)
                totals[label] += duration_us / 1000
            for label, value in totals.items():
                sums[label].append(value)
        category_totals = defaultdict(list)
        spans, coverage = [], []
        for start, end in boundaries:
            totals = defaultdict(float)
            intervals = []
            for row in all_rows:
                left, right = max(start, int(row['start'])), min(end, int(row['end']))
                if right <= left:
                    continue
                kernels = row['kernels']
                category = kernels.split('/operations/', 1)[-1].split('/device/', 1)[0]
                totals[category] += (right - left) / frequency / 1e6
                intervals.append((left, right))
            covered, previous = 0, start
            for left, right in sorted(intervals):
                covered += max(0, right - max(left, previous))
                previous = max(previous, right)
            spans.append((end - start) / frequency / 1e6)
            coverage.append(covered / frequency / 1e6)
            for category, total in totals.items():
                category_totals[category].append(total)
        report[str(chip)] = {
            'complete_decode_iterations': len(cycles),
            'window_ms': window_ms,
            'median_lm_head_to_lm_head_ms': statistics.median(spans) if spans else None,
            'median_program_covered_ms': statistics.median(coverage) if coverage else None,
            'program_category_ms_per_token': dict(sorted(
                ((k, statistics.median(v)) for k, v in category_totals.items()),
                key=lambda item: -item[1])),
            'max_reported_dropped': max(int(r['dropped']) for r in all_rows if int(r['end']) >= cutoff),
            'projections': {name: {'median_kernel_us': statistics.median(values),
                                    'median_total_ms_per_token': statistics.median(sums[name]),
                                    'sample_count': len(values)} for name, values in samples.items()},
        }
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--window-ms', type=float, default=500)
    args = parser.parse_args()
    print(json.dumps(analyze(args.directory, args.window_ms), indent=2))
