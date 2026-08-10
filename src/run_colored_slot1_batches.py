#!/usr/bin/env python3
"""Run exact colored MITM on distinct tied paths from a history slot."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import subprocess
import time


@dataclass(frozen=True)
class Batch:
    index: int
    ids: tuple[str, ...]


def load_paths(path: Path) -> dict[str, str]:
    with path.open(newline="") as handle:
        return {
            row["initial_state_id"]: row["path"]
            for row in csv.DictReader(handle)
        }


def path_length(path: str) -> int:
    return 0 if not path else path.count(".") + 1


def select_ids(slot: Path, baseline: Path, min_length: int) -> list[str]:
    slot_paths = load_paths(slot)
    baseline_paths = load_paths(baseline)
    selected = [
        state_id
        for state_id, baseline_path in baseline_paths.items()
        if state_id in slot_paths
        and path_length(baseline_path) >= min_length
        and path_length(slot_paths[state_id]) == path_length(baseline_path)
        and slot_paths[state_id] != baseline_path
    ]
    selected.sort(
        key=lambda state_id: (-path_length(baseline_paths[state_id]), int(state_id))
    )
    return selected


def run_batch(
    batch: Batch,
    *,
    binary: Path,
    puzzle_info: Path,
    slot: Path,
    test: Path,
    output_dir: Path,
    min_length: int,
    max_length: int,
) -> tuple[int, int, float, Path]:
    output = output_dir / f"batch_{batch.index:03d}.csv"
    relations = output_dir / f"batch_{batch.index:03d}_relations.tsv"
    log = output_dir / f"batch_{batch.index:03d}.log"
    command = [
        str(binary),
        str(puzzle_info),
        str(slot),
        str(test),
        str(output),
        str(relations),
        str(min_length),
        str(max_length),
        ",".join(batch.ids),
    ]
    begin = time.monotonic()
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    elapsed = time.monotonic() - begin
    log.write_text(
        "command=" + " ".join(command) + "\n"
        + f"returncode={result.returncode}\nelapsed_seconds={elapsed:.6f}\n"
        + "stdout:\n" + result.stdout
        + "stderr:\n" + result.stderr
    )
    return batch.index, result.returncode, elapsed, log


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slot", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--puzzle-info", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-length", type=int, default=44)
    parser.add_argument("--max-length", type=int, default=999)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--start-batch", type=int, default=0)
    parser.add_argument("--stop-batch", type=int)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = select_ids(args.slot, args.baseline, args.min_length)
    batches = [
        Batch(index // args.batch_size, tuple(selected[index:index + args.batch_size]))
        for index in range(0, len(selected), args.batch_size)
    ]
    stop = len(batches) if args.stop_batch is None else args.stop_batch
    batches = [batch for batch in batches if args.start_batch <= batch.index < stop]
    print(
        f"selected_distinct_tied={len(selected)} total_batches="
        f"{(len(selected) + args.batch_size - 1) // args.batch_size} "
        f"scheduled_batches={len(batches)} workers={args.workers}",
        flush=True,
    )
    failures = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                run_batch,
                batch,
                binary=args.binary,
                puzzle_info=args.puzzle_info,
                slot=args.slot,
                test=args.test,
                output_dir=args.output_dir,
                min_length=args.min_length,
                max_length=args.max_length,
            ): batch
            for batch in batches
        }
        for future in as_completed(futures):
            index, returncode, elapsed, log = future.result()
            failures += returncode != 0
            print(
                f"batch={index:03d} returncode={returncode} "
                f"seconds={elapsed:.3f} log={log}",
                flush=True,
            )
    if failures:
        raise SystemExit(f"{failures} batch(es) failed")


if __name__ == "__main__":
    main()
