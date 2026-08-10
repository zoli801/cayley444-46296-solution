#!/usr/bin/env python3
"""Build replay-valid full submissions from ranked alternate solution paths."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from cube444 import Puzzle, load_submission, load_tests, split_path, validate_submission


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_csv", type=Path)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--slots", type=int, default=4)
    parser.add_argument("--max-slack", type=int, default=2)
    parser.add_argument("--min-baseline-length", type=int, default=0)
    parser.add_argument("--puzzle-info", type=Path, default=Path("data/puzzle_info.json"))
    parser.add_argument("--test", type=Path, default=Path("data/test.csv"))
    args = parser.parse_args()

    puzzle = Puzzle.load(args.puzzle_info)
    tests = load_tests(args.test)
    baseline = load_submission(args.baseline)
    baseline_validation = validate_submission(puzzle, tests, baseline)
    if not baseline_validation.valid:
        raise ValueError("invalid baseline: " + "; ".join(baseline_validation.failures[:5]))

    alternatives: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    rejected = 0
    with args.candidate_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"initial_state_id", "path"}.issubset(reader.fieldnames or ()):
            raise ValueError("candidate CSV needs initial_state_id,path columns")
        for row in reader:
            state_id = row["initial_state_id"]
            if state_id not in tests:
                rejected += 1
                continue
            moves = split_path(row["path"])
            incumbent = baseline[state_id]
            if (
                len(incumbent) < args.min_baseline_length
                or len(moves) > len(incumbent) + args.max_slack
                or moves == incumbent
                or puzzle.replay(tests[state_id], moves) != puzzle.central_state
            ):
                rejected += 1
                continue
            alternatives[state_id].add(moves)

    ranked = {
        state_id: sorted(paths, key=lambda moves: (len(moves), moves))
        for state_id, paths in alternatives.items()
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for slot in range(args.slots):
        selected = dict(baseline)
        changed_ids: list[str] = []
        for state_id, paths in ranked.items():
            if slot < len(paths):
                selected[state_id] = paths[slot]
                changed_ids.append(state_id)
        output = args.output_dir / f"slot_{slot}.csv"
        with output.open("w", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(("initial_state_id", "path"))
            for state_id in tests:
                writer.writerow((state_id, ".".join(selected[state_id])))
        validation = validate_submission(puzzle, tests, selected)
        if not validation.valid:
            output.unlink(missing_ok=True)
            raise RuntimeError("generated invalid slot: " + "; ".join(validation.failures[:5]))
        (args.output_dir / f"slot_{slot}_ids.txt").write_text(
            ",".join(sorted(changed_ids, key=int)) + "\n"
        )
        print(
            f"slot={slot} changed={len(changed_ids)} score={validation.score} "
            f"output={output}"
        )
    print(
        f"candidate_ids={len(ranked)} alternate_paths={sum(map(len, ranked.values()))} "
        f"rejected={rejected}"
    )


if __name__ == "__main__":
    main()
