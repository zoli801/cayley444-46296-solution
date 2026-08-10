#!/usr/bin/env python3
"""Emit exhaustive length-11/12 checkpoint windows for selected solution rows."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from cube444 import load_submission


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ids", required=True, help="comma-separated state IDs")
    parser.add_argument("--lengths", default="11,12")
    args = parser.parse_args()

    paths = load_submission(args.submission)
    ids = [value for value in args.ids.split(",") if value]
    lengths = sorted({int(value) for value in args.lengths.split(",") if value})
    if not ids or not lengths or any(length < 1 or length > 12 for length in lengths):
        raise ValueError("IDs and window lengths 1..12 are required")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("initial_state_id", "start", "end"))
        count = 0
        for state_id in ids:
            path_length = len(paths[state_id])
            for length in lengths:
                for start in range(path_length - length + 1):
                    writer.writerow((state_id, start, start + length))
                    count += 1
    print(f"ids={len(ids)} windows={count} output={args.output}")


if __name__ == "__main__":
    main()
