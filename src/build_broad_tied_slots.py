#!/usr/bin/env python3
"""Build checkpoint-diverse tied slots from replay-valid local CSV corpora."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from dataclasses import asdict
import json
from pathlib import Path

from cube444 import Puzzle, load_submission, load_tests, validate_submission
from merge_candidates import (
    IngestStats,
    atomic_write_text,
    csv_text,
    ingest_source,
    iter_csv_sources,
)


def trajectory_states(
    puzzle: Puzzle,
    initial: tuple[int, ...],
    moves: tuple[str, ...],
) -> frozenset[tuple[int, ...]]:
    state = initial
    states = {state}
    for move in moves:
        state = puzzle.apply(state, move)
        states.add(state)
    return frozenset(states)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--exclude-slot-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--slots", type=int, default=4)
    parser.add_argument("--min-baseline-length", type=int, default=44)
    parser.add_argument("--puzzle-info", type=Path, default=Path("data/puzzle_info.json"))
    parser.add_argument("--test", type=Path, default=Path("data/test.csv"))
    parser.add_argument("--max-member-mib", type=int, default=64)
    parser.add_argument("--max-zip-depth", type=int, default=3)
    args = parser.parse_args()

    puzzle = Puzzle.load(args.puzzle_info)
    tests = load_tests(args.test)
    baseline = load_submission(args.baseline)
    baseline_result = validate_submission(puzzle, tests, baseline)
    if not baseline_result.valid:
        raise ValueError("invalid baseline: " + "; ".join(baseline_result.failures[:5]))

    excluded: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    excluded_files = sorted(args.exclude_slot_dir.glob("slot_*.csv"))
    for path in excluded_files:
        submission = load_submission(path)
        result = validate_submission(puzzle, tests, submission)
        if not result.valid:
            raise ValueError(f"invalid exclusion slot {path}: {result.failures[:1]}")
        for state_id, moves in submission.items():
            excluded[state_id].add(moves)

    stats = IngestStats()
    replay_cache: dict[tuple[str, tuple[str, ...]], str | None] = {}
    alternatives: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    for source in iter_csv_sources(
        args.baseline,
        args.inputs,
        stats,
        max_member_bytes=args.max_member_mib * 1024 * 1024,
        max_zip_depth=args.max_zip_depth,
    ):
        candidates, _ = ingest_source(source, puzzle, tests, stats, replay_cache)
        for candidate in candidates:
            incumbent = baseline[candidate.state_id]
            if (
                candidate.baseline
                or len(incumbent) < args.min_baseline_length
                or candidate.length != len(incumbent)
                or candidate.moves == incumbent
                or candidate.moves in excluded[candidate.state_id]
            ):
                continue
            alternatives[candidate.state_id].add(candidate.moves)

    # Greedily choose paths with the most checkpoints not already exposed by
    # the incumbent, kernel-history slots, or an earlier broad-corpus slot.
    selected_by_slot: list[dict[str, tuple[str, ...]]] = [dict() for _ in range(args.slots)]
    novelty_by_slot = [0] * args.slots
    for state_id in sorted(alternatives, key=int):
        initial = tests[state_id]
        covered = set(trajectory_states(puzzle, initial, baseline[state_id]))
        for path in excluded[state_id]:
            covered.update(trajectory_states(puzzle, initial, path))
        remaining = {
            path: trajectory_states(puzzle, initial, path)
            for path in alternatives[state_id]
        }
        for slot_index in range(args.slots):
            if not remaining:
                break
            path = min(
                remaining,
                key=lambda item: (-len(remaining[item] - covered), item),
            )
            states = remaining.pop(path)
            novelty_by_slot[slot_index] += len(states - covered)
            covered.update(states)
            selected_by_slot[slot_index][state_id] = path

    args.output_dir.mkdir(parents=True, exist_ok=True)
    slot_reports: list[dict[str, object]] = []
    ordered_ids = sorted(tests, key=int)
    for slot_index, selected in enumerate(selected_by_slot):
        submission = dict(baseline)
        submission.update(selected)
        result = validate_submission(puzzle, tests, submission)
        if not result.valid or result.score != baseline_result.score:
            raise AssertionError(
                f"slot {slot_index} failed validation: score={result.score} "
                + "; ".join(result.failures[:5])
            )
        output = args.output_dir / f"slot_{slot_index}.csv"
        atomic_write_text(
            output,
            csv_text(
                ("initial_state_id", "path"),
                ((state_id, ".".join(submission[state_id])) for state_id in ordered_ids),
            ),
        )
        atomic_write_text(
            args.output_dir / f"slot_{slot_index}_ids.txt",
            ",".join(sorted(selected, key=int)) + "\n",
        )
        slot_reports.append(
            {
                "slot": slot_index,
                "changed_ids": len(selected),
                "new_checkpoint_count": novelty_by_slot[slot_index],
                "output": str(output),
                "score": result.score,
            }
        )
        print(
            f"slot={slot_index} changed={len(selected)} "
            f"new_checkpoints={novelty_by_slot[slot_index]} score={result.score}",
            flush=True,
        )

    report = {
        "baseline": str(args.baseline),
        "baseline_score": baseline_result.score,
        "candidate_ids": len(alternatives),
        "distinct_tied_paths": sum(map(len, alternatives.values())),
        "excluded_slot_files": [str(path) for path in excluded_files],
        "ingest": asdict(stats),
        "slots": slot_reports,
    }
    atomic_write_text(
        args.output_dir / "report.json",
        json.dumps(report, indent=2, sort_keys=True) + "\n",
    )
    print(
        f"candidate_ids={report['candidate_ids']} "
        f"distinct_tied_paths={report['distinct_tied_paths']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
