#!/usr/bin/env python3
"""Create and exactly validate a submission with selected candidate-row overrides."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from cube444 import Puzzle, load_submission, load_tests, split_path, validate_submission
from merge_candidates import atomic_write_text, csv_text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--select",
        action="append",
        required=True,
        metavar="ID:KERNEL_VERSION",
        help="candidate row to select; may be repeated",
    )
    parser.add_argument("--puzzle-info", type=Path, default=Path("data/puzzle_info.json"))
    parser.add_argument("--test", type=Path, default=Path("data/test.csv"))
    args = parser.parse_args()

    selections: dict[str, str] = {}
    for value in args.select:
        try:
            state_id, version = value.split(":", 1)
        except ValueError as error:
            raise ValueError(f"invalid --select {value!r}; expected ID:KERNEL_VERSION") from error
        if state_id in selections:
            raise ValueError(f"duplicate selected ID {state_id}")
        selections[state_id] = version

    puzzle = Puzzle.load(args.puzzle_info)
    tests = load_tests(args.test)
    submission = dict(load_submission(args.baseline))
    baseline_result = validate_submission(puzzle, tests, submission)
    if not baseline_result.valid:
        raise ValueError(f"invalid baseline: {'; '.join(baseline_result.failures[:5])}")

    matches: dict[str, tuple[str, ...]] = {}
    with args.candidates.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"initial_state_id", "path", "kernel_version"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"candidate CSV must contain {sorted(required)}")
        for row in reader:
            state_id = row["initial_state_id"]
            if selections.get(state_id) != row["kernel_version"]:
                continue
            if state_id in matches:
                raise ValueError(
                    f"selection {state_id}:{row['kernel_version']} matched multiple rows"
                )
            matches[state_id] = split_path(row["path"])

    missing = selections.keys() - matches.keys()
    if missing:
        raise ValueError(f"unmatched selections: {','.join(sorted(missing, key=int))}")
    submission.update(matches)
    result = validate_submission(puzzle, tests, submission)
    if not result.valid:
        raise ValueError(f"invalid output: {'; '.join(result.failures[:5])}")

    atomic_write_text(
        args.output,
        csv_text(
            ("initial_state_id", "path"),
            (
                (state_id, ".".join(submission[state_id]))
                for state_id in sorted(tests, key=int)
            ),
        ),
    )
    print(
        f"puzzles={result.puzzle_count} score={result.score} valid={result.valid} "
        + " ".join(
            f"id{state_id}={len(matches[state_id])}" for state_id in sorted(matches, key=int)
        )
    )


if __name__ == "__main__":
    main()
