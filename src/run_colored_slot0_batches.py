#!/usr/bin/env python3
"""Run resumable exact colored radius-8 batches on selected slot-0 paths."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import time

from cube444 import (
    Puzzle,
    load_submission,
    load_tests,
    split_path,
    validate_submission,
)
from merge_candidates import atomic_write_text, csv_text


@dataclass(frozen=True)
class Target:
    state_id: str
    length: int


def parse_ids(value: str) -> set[str]:
    return {item for item in value.split(",") if item}


def atomic_json(path: Path, payload: object) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slot", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument(
        "--target-reference",
        type=Path,
        help="submission used only to select slot paths (defaults to baseline)",
    )
    parser.add_argument(
        "--slot-delta",
        type=int,
        default=0,
        help="select rows with len(slot) == len(target reference) + this value",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--binary", type=Path, default=Path("/private/tmp/colored_mitm8_changed")
    )
    parser.add_argument("--puzzle-info", type=Path, default=Path("data/puzzle_info.json"))
    parser.add_argument("--test", type=Path, default=Path("data/test.csv"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--min-length", type=int, default=44)
    parser.add_argument(
        "--max-length",
        type=int,
        help="optionally exclude baseline rows longer than this",
    )
    parser.add_argument("--exclude-ids", default="")
    parser.add_argument(
        "--id-parity",
        choices=("all", "even", "odd"),
        default="all",
        help="optionally restrict targets by numeric state-ID parity",
    )
    parser.add_argument(
        "--id-modulus",
        type=int,
        default=1,
        help="partition IDs by ID modulo this positive value",
    )
    parser.add_argument(
        "--id-remainder",
        type=int,
        default=0,
        help="retain IDs whose remainder modulo --id-modulus equals this value",
    )
    parser.add_argument(
        "--include-identical-reference",
        action="store_true",
        help="scan slot rows even when they equal the target-reference rows",
    )
    parser.add_argument("--max-batches", type=int)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.id_modulus < 1 or not 0 <= args.id_remainder < args.id_modulus:
        parser.error("require positive --id-modulus and 0 <= --id-remainder < modulus")

    puzzle = Puzzle.load(args.puzzle_info)
    tests = load_tests(args.test)
    baseline = load_submission(args.baseline)
    target_reference_path = args.target_reference or args.baseline
    target_reference = load_submission(target_reference_path)
    slot = load_submission(args.slot)
    for label, submission in (
        ("baseline", baseline),
        ("target reference", target_reference),
        ("slot", slot),
    ):
        result = validate_submission(puzzle, tests, submission)
        if not result.valid:
            raise ValueError(f"{label} is invalid: {'; '.join(result.failures[:5])}")

    excluded = parse_ids(args.exclude_ids)
    parity = {"all": None, "even": 0, "odd": 1}[args.id_parity]
    targets = [
        Target(state_id, len(slot[state_id]))
        for state_id in tests
        if state_id not in excluded
        and (parity is None or int(state_id) % 2 == parity)
        and int(state_id) % args.id_modulus == args.id_remainder
        and (
            args.include_identical_reference
            or slot[state_id] != target_reference[state_id]
        )
        and len(slot[state_id])
        == len(target_reference[state_id]) + args.slot_delta
        and len(baseline[state_id]) >= args.min_length
        and (
            args.max_length is None
            or len(baseline[state_id]) <= args.max_length
        )
    ]
    targets.sort(key=lambda item: (-item.length, int(item.state_id)))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    completed: set[str] = set()
    for report_path in sorted(args.output_dir.glob("batch_*.json")):
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if (
                report.get("valid")
                and report.get("slot") == str(args.slot)
                and report.get("target_reference", str(args.baseline))
                == str(target_reference_path)
                and int(report.get("slot_delta", 0)) == args.slot_delta
                and report.get("id_parity", "all") == args.id_parity
                and int(report.get("id_modulus", 1)) == args.id_modulus
                and int(report.get("id_remainder", 0)) == args.id_remainder
                and bool(report.get("include_identical_reference", False))
                == args.include_identical_reference
            ):
                completed.update(map(str, report.get("ids", ())))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    pending = [target for target in targets if target.state_id not in completed]

    incumbent_path = args.output_dir / "incumbent.csv"
    incumbent = (
        load_submission(incumbent_path) if incumbent_path.is_file() else dict(baseline)
    )
    incumbent_result = validate_submission(puzzle, tests, incumbent)
    if not incumbent_result.valid:
        raise ValueError("saved incumbent is invalid")
    starting_score = incumbent_result.score

    tied_paths: dict[str, tuple[str, ...]] = {}
    tied_path_file = args.output_dir / "tied_paths.csv"
    if tied_path_file.is_file():
        with tied_path_file.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if not {"initial_state_id", "path"}.issubset(reader.fieldnames or ()):
                raise ValueError("saved tied_paths.csv has unexpected columns")
            for row in reader:
                tied_paths[row["initial_state_id"]] = split_path(row["path"])

    print(
        f"targets={len(targets)} completed={len(completed)} pending={len(pending)} "
        f"baseline_score={starting_score}",
        flush=True,
    )
    batch_count = 0
    for offset in range(0, len(pending), args.batch_size):
        if args.max_batches is not None and batch_count >= args.max_batches:
            break
        group = pending[offset : offset + args.batch_size]
        batch_index = 0
        while (args.output_dir / f"batch_{batch_index:04d}.json").exists():
            batch_index += 1
        stem = args.output_dir / f"batch_{batch_index:04d}"
        candidate_path = stem.with_suffix(".csv")
        relations_path = stem.with_suffix(".tsv")
        log_path = stem.with_suffix(".log")
        ids = [item.state_id for item in group]
        command = [
            str(args.binary),
            str(args.puzzle_info),
            str(args.slot),
            str(args.test),
            str(candidate_path),
            str(relations_path),
            str(min(item.length for item in group)),
            str(max(item.length for item in group)),
            ",".join(ids),
        ]
        print(
            f"batch={batch_index} ids={','.join(ids)} lengths="
            f"{min(item.length for item in group)}-{max(item.length for item in group)}",
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
                f"batch {batch_index} failed with code {process.returncode}; see {log_path}"
            )

        candidate = load_submission(candidate_path)
        validation = validate_submission(puzzle, tests, candidate)
        if not validation.valid:
            raise RuntimeError(
                f"batch {batch_index} replay failed: {'; '.join(validation.failures[:5])}"
            )
        improvements: list[dict[str, int | str]] = []
        ties: list[dict[str, int | str]] = []
        outcomes: list[dict[str, int | str]] = []
        for target in group:
            old_length = len(incumbent[target.state_id])
            new_length = len(candidate[target.state_id])
            outcomes.append(
                {
                    "initial_state_id": target.state_id,
                    "slot_length": len(slot[target.state_id]),
                    "reference_length": len(target_reference[target.state_id]),
                    "incumbent_length": old_length,
                    "candidate_length": new_length,
                    "slot_gain": len(slot[target.state_id]) - new_length,
                }
            )
            if new_length < old_length:
                incumbent[target.state_id] = candidate[target.state_id]
                improvements.append(
                    {
                        "initial_state_id": target.state_id,
                        "old_length": old_length,
                        "new_length": new_length,
                        "gain": old_length - new_length,
                    }
                )
            elif (
                new_length == old_length
                and candidate[target.state_id] != incumbent[target.state_id]
            ):
                tied_paths[target.state_id] = candidate[target.state_id]
                ties.append(
                    {
                        "initial_state_id": target.state_id,
                        "length": new_length,
                    }
                )
        incumbent_validation = validate_submission(puzzle, tests, incumbent)
        if not incumbent_validation.valid:
            raise RuntimeError("cumulative incumbent failed replay")
        atomic_write_text(
            incumbent_path,
            csv_text(
                ("initial_state_id", "path"),
                (
                    (state_id, ".".join(incumbent[state_id]))
                    for state_id in sorted(tests, key=int)
                ),
            ),
        )
        atomic_write_text(
            tied_path_file,
            csv_text(
                ("initial_state_id", "path"),
                (
                    (state_id, ".".join(tied_paths[state_id]))
                    for state_id in sorted(tied_paths, key=int)
                ),
            ),
        )
        report = {
            "slot": str(args.slot),
            "baseline": str(args.baseline),
            "target_reference": str(target_reference_path),
            "slot_delta": args.slot_delta,
            "id_parity": args.id_parity,
            "id_modulus": args.id_modulus,
            "id_remainder": args.id_remainder,
            "include_identical_reference": args.include_identical_reference,
            "ids": ids,
            "lengths": [item.length for item in group],
            "elapsed_seconds": elapsed,
            "candidate_score": validation.score,
            "valid": True,
            "improvements": improvements,
            "ties": ties,
            "outcomes": outcomes,
            "incumbent_score": incumbent_validation.score,
            "candidate": str(candidate_path),
            "relations": str(relations_path),
            "log": str(log_path),
        }
        atomic_json(stem.with_suffix(".json"), report)
        print(
            f"batch={batch_index} valid=True seconds={elapsed:.1f} "
            f"gains={sum(int(item['gain']) for item in improvements)} "
            f"improved_ids={','.join(str(item['initial_state_id']) for item in improvements)} "
            f"tied_ids={','.join(str(item['initial_state_id']) for item in ties)} "
            f"incumbent={incumbent_validation.score}",
            flush=True,
        )
        batch_count += 1

    final = validate_submission(puzzle, tests, incumbent)
    atomic_json(
        args.output_dir / "summary.json",
        {
            "slot": str(args.slot),
            "baseline": str(args.baseline),
            "target_reference": str(target_reference_path),
            "slot_delta": args.slot_delta,
            "id_parity": args.id_parity,
            "id_modulus": args.id_modulus,
            "id_remainder": args.id_remainder,
            "include_identical_reference": args.include_identical_reference,
            "target_count": len(targets),
            "completed_before_run": len(completed),
            "batches_this_run": batch_count,
            "remaining_after_run": max(0, len(pending) - batch_count * args.batch_size),
            "starting_score": starting_score,
            "final_score": final.score,
            "gain": starting_score - final.score,
            "tied_path_count": len(tied_paths),
            "valid": final.valid,
            "incumbent": str(incumbent_path),
        },
    )
    print(
        f"done batches={batch_count} score={final.score} "
        f"gain={starting_score-final.score} valid={final.valid}",
        flush=True,
    )


if __name__ == "__main__":
    main()
