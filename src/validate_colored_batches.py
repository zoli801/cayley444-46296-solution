#!/usr/bin/env python3
"""Follow exact-colored batch outputs, replay them, and merge improvements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from cube444 import Puzzle, load_submission, load_tests, validate_submission
from merge_candidates import atomic_write_text, csv_text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--start-batch", type=int, default=0)
    parser.add_argument("--stop-batch", type=int, required=True)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--puzzle-info", type=Path, default=Path("data/puzzle_info.json"))
    parser.add_argument("--test", type=Path, default=Path("data/test.csv"))
    args = parser.parse_args()

    puzzle = Puzzle.load(args.puzzle_info)
    tests = load_tests(args.test)
    current = load_submission(args.baseline)
    baseline_result = validate_submission(puzzle, tests, current)
    if not baseline_result.valid:
        raise ValueError("baseline is invalid: " + "; ".join(baseline_result.failures[:5]))

    events: list[dict[str, object]] = []
    ordered_ids = sorted(tests, key=int)
    current_score = baseline_result.score
    for index in range(args.start_batch, args.stop_batch):
        output = args.batch_dir / f"batch_{index:03d}.csv"
        log = args.batch_dir / f"batch_{index:03d}.log"
        relations = args.batch_dir / f"batch_{index:03d}_relations.tsv"
        while not (output.is_file() and log.is_file() and relations.is_file()):
            time.sleep(args.poll_seconds)
        candidate = load_submission(output)
        result = validate_submission(puzzle, tests, candidate)
        if not result.valid:
            raise RuntimeError(
                f"batch {index:03d} invalid: " + "; ".join(result.failures[:5])
            )
        improved_ids = [
            state_id
            for state_id in ordered_ids
            if len(candidate[state_id]) < len(current[state_id])
        ]
        gain = sum(
            len(current[state_id]) - len(candidate[state_id])
            for state_id in improved_ids
        )
        for state_id in improved_ids:
            current[state_id] = candidate[state_id]
        current_score -= gain
        relation_count = max(0, sum(1 for _ in relations.open()) - 1)
        atomic_write_text(
            args.output,
            csv_text(
                ("initial_state_id", "path"),
                ((state_id, ".".join(current[state_id])) for state_id in ordered_ids),
            ),
        )
        merged_result = validate_submission(puzzle, tests, load_submission(args.output))
        if not merged_result.valid or merged_result.score != current_score:
            raise AssertionError(f"merged output failed after batch {index:03d}")
        event = {
            "batch": index,
            "candidate_score": result.score,
            "gain": gain,
            "improved_ids": improved_ids,
            "relation_count": relation_count,
            "score": current_score,
        }
        events.append(event)
        atomic_write_text(
            args.report,
            json.dumps(
                {
                    "baseline": str(args.baseline),
                    "baseline_score": baseline_result.score,
                    "batches_validated": len(events),
                    "events": events,
                    "gain": baseline_result.score - current_score,
                    "output": str(args.output),
                    "score": current_score,
                },
                indent=2,
                sort_keys=True,
            ) + "\n",
        )
        print(
            f"batch={index:03d} valid=True relations={relation_count} "
            f"gain={gain} score={current_score} ids={','.join(improved_ids) or '-'}",
            flush=True,
        )


if __name__ == "__main__":
    main()
