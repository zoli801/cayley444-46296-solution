#!/usr/bin/env python3
"""Shorten a diverse, validated candidate corpus with one twsearch run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from cube444 import Puzzle, load_submission, load_tests, validate_submission
from merge_candidates import IngestStats, ingest_source, iter_csv_sources
from twsearch_optimize_submission import decode_path, encode_path, run_twsearch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-slack", type=int, default=4)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument(
        "--inputs-only",
        action="store_true",
        help="optimize only input candidates, excluding baseline rows from the batch",
    )
    parser.add_argument(
        "--min-baseline-length",
        type=int,
        default=0,
        help="only optimize candidates for rows at least this long in the baseline",
    )
    parser.add_argument(
        "--max-baseline-length",
        type=int,
        help="optionally exclude rows longer than this in the baseline",
    )
    parser.add_argument("--memory-mb", type=int, default=2048)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and count the filtered unique candidates without running twsearch",
    )
    parser.add_argument(
        "--definition", type=Path, default=Path("work/cayleypy444_cubies.tws")
    )
    parser.add_argument(
        "--twsearch", type=Path, default=Path("work/twsearch-cpp/build/bin/twsearch")
    )
    parser.add_argument(
        "--puzzle-info", type=Path, default=Path("data/puzzle_info.json")
    )
    parser.add_argument("--test", type=Path, default=Path("data/test.csv"))
    args = parser.parse_args()

    puzzle = Puzzle.load(args.puzzle_info)
    tests = load_tests(args.test)
    baseline = load_submission(args.baseline)
    validation = validate_submission(puzzle, tests, baseline)
    if not validation.valid:
        raise ValueError("invalid baseline: " + "; ".join(validation.failures[:5]))

    stats = IngestStats()
    replay_cache: dict[tuple[str, tuple[str, ...]], str | None] = {}
    unique: dict[tuple[str, tuple[str, ...]], None] = {}
    for source in iter_csv_sources(
        args.baseline,
        args.inputs,
        stats,
        max_member_bytes=64 * 1024 * 1024,
        max_zip_depth=3,
    ):
        candidates, _ = ingest_source(source, puzzle, tests, stats, replay_cache)
        for candidate in candidates:
            if args.inputs_only and candidate.baseline:
                continue
            baseline_length = len(baseline[candidate.state_id])
            if (
                baseline_length >= args.min_baseline_length
                and (
                    args.max_baseline_length is None
                    or baseline_length <= args.max_baseline_length
                )
                and candidate.length <= baseline_length + args.max_slack
            ):
                unique[(candidate.state_id, candidate.moves)] = None

    ordered = sorted(unique, key=lambda item: (int(item[0]), len(item[1]), item[1]))
    if args.dry_run:
        print(
            f"candidates={len(ordered)} baseline={validation.score} "
            f"inputs_only={args.inputs_only} min_baseline_length={args.min_baseline_length} "
            f"max_slack={args.max_slack}"
        )
        return
    encoded = [encode_path(".".join(moves)) for _, moves in ordered]
    output_lines, elapsed = run_twsearch(
        args.twsearch,
        args.definition,
        encoded,
        "shorten",
        args.max_depth,
        args.memory_mb,
        args.threads,
    )
    bases = {name for name in puzzle.generators if not name.startswith("-")}
    rewritten: list[tuple[str, str, int, int]] = []
    shortened = 0
    for (state_id, old_moves), line in zip(ordered, output_lines):
        path = decode_path(line, bases)
        new_moves = tuple(path.split(".")) if path else ()
        final = puzzle.replay(tests[state_id], new_moves)
        if final != puzzle.central_state:
            raise RuntimeError(f"twsearch output failed replay for id {state_id}")
        if len(new_moves) > len(old_moves):
            raise RuntimeError(f"twsearch length increased for id {state_id}")
        shortened += len(new_moves) < len(old_moves)
        rewritten.append((state_id, path, len(old_moves), len(new_moves)))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("initial_state_id", "path", "old_length", "new_length"))
        writer.writerows(rewritten)

    best = dict(baseline)
    for state_id, path, _, _ in rewritten:
        moves = tuple(path.split(".")) if path else ()
        if len(moves) < len(best[state_id]):
            best[state_id] = moves
    old_score = validation.score
    new_score = sum(len(best[state_id]) for state_id in tests)
    report = {
        "baseline": str(args.baseline),
        "baseline_score": old_score,
        "candidate_count": len(ordered),
        "elapsed_seconds": elapsed,
        "max_depth": args.max_depth,
        "max_slack": args.max_slack,
        "inputs_only": args.inputs_only,
        "min_baseline_length": args.min_baseline_length,
        "max_baseline_length": args.max_baseline_length,
        "merged_score_floor": new_score,
        "potential_gain": old_score - new_score,
        "shortened_candidate_count": shortened,
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        f"candidates={len(ordered)} shortened={shortened} seconds={elapsed:.3f} "
        f"baseline={old_score} merged_floor={new_score} gain={old_score-new_score}"
    )


if __name__ == "__main__":
    main()
