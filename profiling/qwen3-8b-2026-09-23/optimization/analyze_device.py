"""Summarize the final 500 ms of a diagnostic device profile, deduplicating callbacks."""
import argparse
import collections
import csv
import gzip
import json
from pathlib import Path
import statistics

parser = argparse.ArgumentParser()
parser.add_argument('directory', nargs='?', type=Path, default=Path(__file__).parent / 'device-profile')
args = parser.parse_args()
records = {}
for path in sorted(args.directory.glob('*.csv*')):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as handle:
        for row in csv.DictReader(handle):
            if row['chip_id'] == '0':
                records[(row['start'], row['end'])] = row
frequency = statistics.median(
    (int(r['end']) - int(r['start'])) / float(r['duration_ns'])
    for r in records.values() if float(r['duration_ns']) > 0
)
end = max(int(r['end']) for r in records.values())
rows = [r for r in records.values() if int(r['start']) > end - 500_000_000 * frequency]
groups = collections.defaultdict(list)
for row in rows:
    kernels = row['kernels']
    group = next((name for name in ('matmul', 'all_gather', 'reduce_scatter', 'sdpa') if name in kernels), None)
    if group is None:
        group = 'rmsnorm' if 'layernorm' in kernels or 'rmsnorm' in kernels else kernels.split(';')[0]
    groups[group].append(float(row['duration_ns']) / 1e6)
summary = {
    'window_ms': 500,
    'chip_id': 0,
    'estimated_frequency_ghz': frequency,
    'max_reported_dropped': max(int(r['dropped']) for r in rows),
    'unique_records': len(rows),
    'variant': 'route-aware link discovery + 8 KiB packets; diagnostic build',
    'groups': {
        name: {'count': len(values), 'total_ms': sum(values), 'median_us': statistics.median(values) * 1000}
        for name, values in sorted(groups.items(), key=lambda pair: -sum(pair[1]))
    },
}
print(json.dumps(summary, indent=2))
