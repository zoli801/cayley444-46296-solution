#!/usr/bin/env python3
"""Retain rows whose optimizer metadata reports a strict shortening."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from merge_candidates import atomic_write_text, csv_text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--old-column", default="old_length")
    parser.add_argument("--new-column", default="new_length")
    args = parser.parse_args()

    with args.input.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        required = {args.old_column, args.new_column}
        if not required.issubset(fieldnames):
            raise ValueError(f"missing columns {sorted(required - set(fieldnames))}")
        rows = [
            tuple(row[name] for name in fieldnames)
            for row in reader
            if int(row[args.new_column]) < int(row[args.old_column])
        ]

    atomic_write_text(args.output, csv_text(fieldnames, rows))
    print(f"input={args.input} shortened_rows={len(rows)} output={args.output}")


if __name__ == "__main__":
    main()
