#!/usr/bin/env python3
"""Map legacy Santa 2023 cube paths to CayleyPy test IDs by exact state.

The legacy files identify a puzzle by a dataset-local integer.  A solving word
uniquely determines its initial state: start at the solved state and replay the
inverse word.  Matching that state against ``test.csv`` avoids relying on a
fragile hard-coded ID offset.  Every emitted row is replay-validated.

The output intentionally permits several rows for one CayleyPy ID so it can be
used as a trajectory portfolio by ``make_trajectory_bridge_walks.py``.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from cube444 import Puzzle, load_tests, split_path
from merge_candidates import atomic_write_text, csv_text


def inverse_move(move: str) -> str:
    return move[1:] if move.startswith("-") else f"-{move}"


def initial_state_for_solution(
    puzzle: Puzzle, moves: tuple[str, ...]
) -> tuple[int, ...]:
    return puzzle.replay(
        puzzle.central_state,
        (inverse_move(move) for move in reversed(moves)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--puzzle-info", type=Path, default=Path("data/puzzle_info.json"))
    parser.add_argument("--test", type=Path, default=Path("data/test.csv"))
    parser.add_argument("--minimum-legacy-id", type=int, default=150)
    parser.add_argument("--maximum-legacy-id", type=int, default=199)
    args = parser.parse_args()

    puzzle = Puzzle.load(args.puzzle_info)
    tests = load_tests(args.test)
    state_to_id: dict[tuple[int, ...], str] = {}
    for state_id, state in tests.items():
        if state in state_to_id:
            raise ValueError(
                f"test IDs {state_to_id[state]} and {state_id} have the same state"
            )
        state_to_id[state] = state_id

    rows: list[tuple[str, str, str, str]] = []
    unmatched = 0
    for input_path in args.inputs:
        with input_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if not {"id", "moves"}.issubset(reader.fieldnames or ()):
                raise ValueError(f"{input_path}: expected id,moves columns")
            for row_number, row in enumerate(reader, start=2):
                try:
                    legacy_id = int(row["id"])
                except ValueError as error:
                    raise ValueError(
                        f"{input_path}:{row_number}: non-integer id {row['id']!r}"
                    ) from error
                if not args.minimum_legacy_id <= legacy_id <= args.maximum_legacy_id:
                    continue
                moves = split_path(row["moves"])
                unknown = [move for move in moves if move not in puzzle.generators]
                if unknown:
                    raise ValueError(
                        f"{input_path}:{row_number}: unknown move {unknown[0]!r}"
                    )
                initial = initial_state_for_solution(puzzle, moves)
                state_id = state_to_id.get(initial)
                if state_id is None:
                    unmatched += 1
                    continue
                if puzzle.replay(tests[state_id], moves) != puzzle.central_state:
                    raise AssertionError(
                        f"{input_path}:{row_number}: reconstructed path failed replay"
                    )
                rows.append(
                    (state_id, ".".join(moves), input_path.name, str(legacy_id))
                )

    # Preserve all distinct trajectories while making the artifact deterministic.
    rows = sorted(set(rows), key=lambda row: (int(row[0]), len(split_path(row[1])), row))
    atomic_write_text(
        args.output,
        csv_text(
            ("initial_state_id", "path", "legacy_source", "legacy_id"),
            rows,
        ),
    )
    mapped_ids = {row[0] for row in rows}
    print(
        f"inputs={len(args.inputs)} rows={len(rows)} mapped_ids={len(mapped_ids)} "
        f"unmatched={unmatched} output={args.output}"
    )


if __name__ == "__main__":
    main()
