#!/usr/bin/env python3
"""Exact local replay and scoring for the CayleyPy 4x4x4 competition."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


State = tuple[int, ...]
Permutation = tuple[int, ...]


@dataclass(frozen=True)
class Puzzle:
    central_state: State
    generators: dict[str, Permutation]

    @classmethod
    def load(cls, path: Path) -> "Puzzle":
        raw = json.loads(path.read_text())
        central = tuple(int(value) for value in raw["central_state"])
        generators = {
            name: tuple(int(index) for index in permutation)
            for name, permutation in raw["generators"].items()
        }
        width = len(central)
        expected = set(range(width))
        for name, permutation in generators.items():
            if len(permutation) != width or set(permutation) != expected:
                raise ValueError(f"{name!r} is not a permutation of 0..{width - 1}")
        return cls(central, generators)

    @staticmethod
    def apply_permutation(state: Sequence[int], permutation: Permutation) -> State:
        return tuple(state[index] for index in permutation)

    def apply(self, state: Sequence[int], move: str) -> State:
        try:
            permutation = self.generators[move]
        except KeyError as error:
            raise ValueError(f"unknown move {move!r}") from error
        return self.apply_permutation(state, permutation)

    def replay(self, initial: Sequence[int], moves: Iterable[str]) -> State:
        state = tuple(initial)
        for move in moves:
            state = self.apply(state, move)
        return state


def split_path(path: str) -> tuple[str, ...]:
    value = path.strip()
    if not value:
        return ()
    moves = tuple(value.split("."))
    if any(not move for move in moves):
        raise ValueError(f"malformed path {path!r}")
    return moves


def load_tests(path: Path) -> dict[str, State]:
    tests: dict[str, State] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"initial_state_id", "initial_state"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"test CSV must contain {sorted(required)}")
        for row in reader:
            state_id = row["initial_state_id"]
            if state_id in tests:
                raise ValueError(f"duplicate test id {state_id}")
            tests[state_id] = tuple(int(value) for value in row["initial_state"].split(","))
    return tests


def load_submission(path: Path) -> dict[str, tuple[str, ...]]:
    solutions: dict[str, tuple[str, ...]] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"initial_state_id", "path"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"submission CSV must contain {sorted(required)}")
        for row in reader:
            state_id = row["initial_state_id"]
            if state_id in solutions:
                raise ValueError(f"duplicate submission id {state_id}")
            solutions[state_id] = split_path(row["path"])
    return solutions


@dataclass(frozen=True)
class ValidationResult:
    score: int
    puzzle_count: int
    failures: tuple[str, ...]
    lengths: dict[str, int]

    @property
    def valid(self) -> bool:
        return not self.failures


def validate_submission(
    puzzle: Puzzle,
    tests: dict[str, State],
    solutions: dict[str, tuple[str, ...]],
) -> ValidationResult:
    failures: list[str] = []
    missing = tests.keys() - solutions.keys()
    extra = solutions.keys() - tests.keys()
    if missing:
        failures.append(f"missing ids: {','.join(sorted(missing, key=int))}")
    if extra:
        failures.append(f"extra ids: {','.join(sorted(extra, key=int))}")

    lengths: dict[str, int] = {}
    for state_id, initial in tests.items():
        moves = solutions.get(state_id)
        if moves is None:
            continue
        lengths[state_id] = len(moves)
        try:
            final = puzzle.replay(initial, moves)
        except ValueError as error:
            failures.append(f"id {state_id}: {error}")
            continue
        if final != puzzle.central_state:
            mismatch = sum(a != b for a, b in zip(final, puzzle.central_state))
            failures.append(f"id {state_id}: unsolved ({mismatch} mismatched stickers)")

    return ValidationResult(
        score=sum(lengths.values()),
        puzzle_count=len(tests),
        failures=tuple(failures),
        lengths=lengths,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission", type=Path)
    parser.add_argument("--puzzle-info", type=Path, default=Path("data/puzzle_info.json"))
    parser.add_argument("--test", type=Path, default=Path("data/test.csv"))
    args = parser.parse_args()

    result = validate_submission(
        Puzzle.load(args.puzzle_info),
        load_tests(args.test),
        load_submission(args.submission),
    )
    print(f"puzzles={result.puzzle_count} score={result.score} valid={result.valid}")
    if result.failures:
        for failure in result.failures[:20]:
            print(failure)
        if len(result.failures) > 20:
            print(f"... {len(result.failures) - 20} more failures")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
