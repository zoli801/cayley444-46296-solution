#!/usr/bin/env python3
"""Construct valid walks exposing ordered checkpoint pairs across solutions.

For a current path P and alternatives A_i, the emitted solving walk is

    P, inv(A_1), A_1, ..., inv(A_k), A_k, inv(P), P

Each inverse leg returns from solved to the common initial state.  Therefore
all forward trajectories occur in one valid solving walk, and an exact
checkpoint splicer can bridge from P into any A_i and from every A_i back
into P.  Reversing the alternative order in a second run covers the remaining
ordered alternative pairs without the memory cost of duplicating every path.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from cube444 import Puzzle, load_submission, load_tests, validate_submission
from merge_candidates import (
    IngestStats,
    atomic_write_text,
    csv_text,
    ingest_source,
    iter_csv_sources,
)


def inverse_move(move: str) -> str:
    return move[1:] if move.startswith("-") else f"-{move}"


def inverse_path(path: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(inverse_move(move) for move in reversed(path))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--puzzle-info", type=Path, default=Path("data/puzzle_info.json"))
    parser.add_argument("--test", type=Path, default=Path("data/test.csv"))
    parser.add_argument("--max-slack", type=int, default=10)
    parser.add_argument(
        "--max-alternatives-per-id",
        type=int,
        help="greedily retain at most this many paths by checkpoint-state novelty",
    )
    parser.add_argument("--reverse-alternatives", action="store_true")
    parser.add_argument(
        "--layout",
        choices=("both", "primary-to-alternative", "alternative-to-primary"),
        default="both",
        help=(
            "checkpoint ordering; the directional layouts are cheaper for "
            "deep exact splicing and require one retained alternative per ID"
        ),
    )
    args = parser.parse_args()

    puzzle = Puzzle.load(args.puzzle_info)
    tests = load_tests(args.test)
    current = load_submission(args.current)
    current_result = validate_submission(puzzle, tests, current)
    if not current_result.valid:
        raise ValueError("current submission is invalid: " + "; ".join(current_result.failures))

    stats = IngestStats()
    replay_cache: dict[tuple[str, tuple[str, ...]], str | None] = {}
    alternatives: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    for source in iter_csv_sources(
        args.current,
        args.inputs,
        stats,
        max_member_bytes=64 * 1024 * 1024,
        max_zip_depth=3,
    ):
        candidates, _ = ingest_source(source, puzzle, tests, stats, replay_cache)
        for candidate in candidates:
            if candidate.length <= len(current[candidate.state_id]) + args.max_slack:
                alternatives[candidate.state_id].add(candidate.moves)

    walks: dict[str, tuple[str, ...]] = {}
    alternative_count = 0
    multi_count = 0
    for state_id in tests:
        primary = current[state_id]
        others = sorted(
            (path for path in alternatives[state_id] if path != primary),
            key=lambda path: (len(path), path),
            reverse=args.reverse_alternatives,
        )
        if (
            args.max_alternatives_per_id is not None
            and len(others) > args.max_alternatives_per_id
        ):
            def checkpoints(path: tuple[str, ...]) -> set[tuple[int, ...]]:
                state = tests[state_id]
                result = {state}
                for move in path:
                    state = puzzle.apply(state, move)
                    result.add(state)
                return result

            covered = checkpoints(primary)
            remaining = [(path, checkpoints(path)) for path in others]
            selected: list[tuple[str, ...]] = []
            while remaining and len(selected) < args.max_alternatives_per_id:
                best_index = max(
                    range(len(remaining)),
                    key=lambda index: (
                        len(remaining[index][1] - covered),
                        -len(remaining[index][0]),
                        tuple(remaining[index][0]),
                    ),
                )
                path, states = remaining.pop(best_index)
                selected.append(path)
                covered.update(states)
            others = selected
        alternative_count += len(others)
        if not others:
            walks[state_id] = primary
            continue
        multi_count += 1
        if args.layout == "alternative-to-primary":
            if len(others) != 1:
                raise ValueError(
                    "alternative-to-primary layout requires exactly one retained "
                    "alternative; use --max-alternatives-per-id 1"
                )
            walk = list(others[0])
            walk.extend(inverse_path(primary))
            walk.extend(primary)
        else:
            walk = list(primary)
            for path in others:
                walk.extend(inverse_path(path))
                walk.extend(path)
            if args.layout == "both":
                # Put the current trajectory after all alternatives as well,
                # exposing the A_i-prefix -> P-suffix bridge direction too.
                walk.extend(inverse_path(primary))
                walk.extend(primary)
        walks[state_id] = tuple(walk)

    ordered_ids = sorted(tests, key=int)
    atomic_write_text(
        args.output,
        csv_text(
            ("initial_state_id", "path"),
            ((state_id, ".".join(walks[state_id])) for state_id in ordered_ids),
        ),
    )
    result = validate_submission(puzzle, tests, load_submission(args.output))
    if not result.valid:
        raise AssertionError("bridge walk failed replay: " + "; ".join(result.failures))
    print(
        f"current={current_result.score} alternatives={alternative_count} "
        f"multi={multi_count} walk_score={result.score} valid={result.valid}"
    )


if __name__ == "__main__":
    main()
