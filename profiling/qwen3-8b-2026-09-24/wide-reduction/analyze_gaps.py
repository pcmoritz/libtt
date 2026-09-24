"""Attribute positive gaps between device programs to their adjacent operations.

Durations outside programs are not necessarily CPU-only time. Run on one TP
profile directory at a time; duplicate callback records are removed.
"""
import argparse
from collections import defaultdict
import csv
import gzip
import json
from pathlib import Path
import statistics


def analyze(directory, chip, window_ms):
    unique = {}
    for path in directory.glob("*.csv*"):
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt") as stream:
            for row in csv.DictReader(stream):
                if int(row["chip_id"]) == chip:
                    unique[int(row["start"]), int(row["end"])] = row
    rows = sorted(unique.values(), key=lambda row: int(row["start"]))
    frequency = statistics.median(
        (int(row["end"]) - int(row["start"])) / float(row["duration_ns"])
        for row in rows if float(row["duration_ns"]) > 0
    )
    cutoff = max(int(row["end"]) for row in rows) - window_ms * 1e6 * frequency
    rows = [row for row in rows if int(row["start"]) > cutoff]
    gaps = defaultdict(list)

    def category(row):
        return row["kernels"].split("/operations/", 1)[-1].split("/device/", 1)[0]

    for previous, current in zip(rows, rows[1:]):
        gap_us = (int(current["start"]) - int(previous["end"])) / frequency / 1000
        if gap_us > 0:
            gaps[category(previous), category(current)].append(gap_us)
    return [
        {"from": key[0], "to": key[1], "count": len(values),
         "median_gap_us": statistics.median(values),
         "total_gap_ms_in_window": sum(values) / 1000}
        for key, values in sorted(gaps.items(), key=lambda item: -sum(item[1]))
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--chip", type=int, default=0)
    parser.add_argument("--window-ms", type=float, default=500)
    args = parser.parse_args()
    print(json.dumps(analyze(args.directory, args.chip, args.window_ms), indent=2))
