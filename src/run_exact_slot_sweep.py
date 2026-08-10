#!/usr/bin/env python3
"""Sequentially shorten tied full submissions and merge strict gains."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

from cube444 import Puzzle, load_submission, load_tests, validate_submission
from merge_candidates import atomic_write_text, csv_text


def write_submission(path: Path, solutions: dict[str, tuple[str, ...]]) -> None:
    atomic_write_text(
        path,
        csv_text(
            ("initial_state_id", "path"),
            (
                (state_id, ".".join(solutions[state_id]))
                for state_id in sorted(solutions, key=int)
            ),
        ),
    )


def write_json(path: Path, payload: object) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("slots", nargs="+", type=Path)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--binary", type=Path, default=Path("work/exact_state_splice_fast"))
    parser.add_argument("--puzzle-info", type=Path, default=Path("data/puzzle_info.json"))
    parser.add_argument("--test", type=Path, default=Path("data/test.csv"))
    parser.add_argument("--radius", type=int, default=6)
    parser.add_argument(
        "--signature-index",
        action="store_true",
        help="use the exact collision-safe sampled-state index",
    )
    args = parser.parse_args()

    puzzle = Puzzle.load(args.puzzle_info)
    tests = load_tests(args.test)
    baseline = load_submission(args.baseline)
    baseline_result = validate_submission(puzzle, tests, baseline)
    if not baseline_result.valid:
        raise ValueError("baseline is invalid: " + "; ".join(baseline_result.failures[:5]))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    incumbent_path = args.output_dir / "incumbent.csv"
    incumbent = (
        load_submission(incumbent_path) if incumbent_path.is_file() else dict(baseline)
    )
    incumbent_result = validate_submission(puzzle, tests, incumbent)
    if not incumbent_result.valid:
        raise ValueError("saved incumbent is invalid")
    initial_score = incumbent_result.score

    completed = 0
    for index, slot_path in enumerate(args.slots):
        stem = args.output_dir / f"slot_{index}"
        report_path = stem.with_suffix(".json")
        if report_path.is_file():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if report.get("valid") and report.get("slot") == str(slot_path):
                completed += 1
                continue

        slot = load_submission(slot_path)
        slot_result = validate_submission(puzzle, tests, slot)
        if not slot_result.valid:
            raise ValueError(
                f"{slot_path} is invalid: " + "; ".join(slot_result.failures[:5])
            )

        candidate_path = stem.with_suffix(".csv")
        log_path = stem.with_suffix(".log")
        command = [
            str(args.binary),
            "--quiet",
            "--input",
            str(slot_path),
            "--output",
            str(candidate_path),
            "--puzzle-info",
            str(args.puzzle_info),
            "--test",
            str(args.test),
            "--radius",
            str(args.radius),
        ]
        if args.signature_index:
            command.append("--signature-index")
        print(
            f"slot={index} input={slot_path} input_score={slot_result.score} "
            f"incumbent={incumbent_result.score}",
            flush=True,
        )
        begin = time.monotonic()
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
                text=True,
            )
        elapsed = time.monotonic() - begin
        if process.returncode:
            raise RuntimeError(
                f"slot {index} optimizer failed with code {process.returncode}; "
                f"see {log_path}"
            )

        candidate = load_submission(candidate_path)
        candidate_result = validate_submission(puzzle, tests, candidate)
        if not candidate_result.valid:
            raise RuntimeError(
                f"slot {index} replay failed: "
                + "; ".join(candidate_result.failures[:5])
            )
        improvements: list[dict[str, int | str]] = []
        for state_id in tests:
            old_length = len(incumbent[state_id])
            new_length = len(candidate[state_id])
            if new_length >= old_length:
                continue
            incumbent[state_id] = candidate[state_id]
            improvements.append(
                {
                    "initial_state_id": state_id,
                    "old_length": old_length,
                    "new_length": new_length,
                    "gain": old_length - new_length,
                }
            )
        incumbent_result = validate_submission(puzzle, tests, incumbent)
        if not incumbent_result.valid:
            raise AssertionError("merged incumbent failed replay")
        write_submission(incumbent_path, incumbent)
        report = {
            "slot": str(slot_path),
            "candidate": str(candidate_path),
            "candidate_score": candidate_result.score,
            "elapsed_seconds": elapsed,
            "improvements": improvements,
            "gain": sum(int(item["gain"]) for item in improvements),
            "incumbent_score": incumbent_result.score,
            "valid": True,
            "log": str(log_path),
        }
        write_json(report_path, report)
        completed += 1
        print(
            f"slot={index} valid=True seconds={elapsed:.1f} "
            f"gain={report['gain']} ids="
            f"{','.join(str(item['initial_state_id']) for item in improvements)} "
            f"incumbent={incumbent_result.score}",
            flush=True,
        )

    write_json(
        args.output_dir / "summary.json",
        {
            "baseline": str(args.baseline),
            "slot_count": len(args.slots),
            "completed": completed,
            "starting_score": initial_score,
            "final_score": incumbent_result.score,
            "gain": initial_score - incumbent_result.score,
            "valid": incumbent_result.valid,
            "incumbent": str(incumbent_path),
        },
    )
    print(
        f"done completed={completed} score={incumbent_result.score} "
        f"gain={initial_score-incumbent_result.score} valid={incumbent_result.valid}",
        flush=True,
    )


if __name__ == "__main__":
    main()
