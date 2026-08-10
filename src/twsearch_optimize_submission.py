#!/usr/bin/env python3
"""Run twsearch sequence rewriting on a CayleyPy submission and validate it."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path

from cube444 import Puzzle, load_submission, load_tests, validate_submission


def score(rows: list[dict[str, str]]) -> int:
    return sum(len(row["path"].split(".")) if row["path"] else 0 for row in rows)


def encode_path(path: str) -> str:
    if not path:
        return ""
    result = []
    for move in path.split("."):
        result.append(move[1:] + "'" if move.startswith("-") else move)
    return " ".join(result)


def decode_move(token: str, bases: set[str]) -> list[str]:
    if token in bases:
        return [token]
    if token.endswith("'") and token[:-1] in bases:
        return ["-" + token[:-1]]
    # All supplied generators have order four.  twsearch may combine two
    # quarter turns into a derived half turn even though the competition has
    # no half-turn token; expand it back to two scored moves.
    for base in sorted(bases, key=len, reverse=True):
        suffix = token[len(base) :] if token.startswith(base) else ""
        if suffix in {"2", "2'"}:
            return [base, base]
    raise ValueError(f"cannot map twsearch move {token!r} to competition syntax")


def decode_path(line: str, bases: set[str]) -> str:
    result: list[str] = []
    for token in line.split():
        result.extend(decode_move(token, bases))
    return ".".join(result)


def run_twsearch(
    binary: Path,
    definition: Path,
    encoded_paths: list[str],
    stage: str,
    max_depth: int,
    memory_mb: int,
    threads: int,
) -> tuple[list[str], float]:
    if stage == "merge":
        options = ["--quiet", "--mergeseqs"]
    else:
        options = [
            "--quiet",
            "-q",
            "--shortenseqs",
            "--maxdepth",
            str(max_depth),
            "-M",
            str(memory_mb),
            "-t",
            str(threads),
            "--nowrite",
        ]
    command = [str(binary), *options, str(definition)]
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        input="\n".join(encoded_paths) + "\n",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    elapsed = time.perf_counter() - started
    if completed.returncode:
        raise RuntimeError(
            f"twsearch failed with status {completed.returncode}\n"
            f"stdout:\n{completed.stdout[-4000:]}\n"
            f"stderr:\n{completed.stderr[-4000:]}"
        )
    lines = completed.stdout.splitlines()
    if len(lines) != len(encoded_paths):
        raise RuntimeError(
            f"expected {len(encoded_paths)} twsearch result lines, got {len(lines)}; "
            "use a build whose --quiet suppresses shortenseqs diagnostics\n"
            f"output tail:\n{completed.stdout[-4000:]}"
        )
    return lines, elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("merge", "shorten"), required=True)
    parser.add_argument("--definition", type=Path, default=Path("work/cayleypy444_cubies.tws"))
    parser.add_argument(
        "--twsearch", type=Path, default=Path("work/twsearch-cpp/build/bin/twsearch")
    )
    parser.add_argument("--puzzle-info", type=Path, default=Path("data/puzzle_info.json"))
    parser.add_argument("--test", type=Path, default=Path("data/test.csv"))
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--memory-mb", type=int, default=512)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()

    with args.puzzle_info.open() as handle:
        raw = json.load(handle)
    bases = {name for name in raw["generators"] if not name.startswith("-")}
    with args.input.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or set(rows[0]) != {"initial_state_id", "path"}:
        raise ValueError("input must have exactly initial_state_id,path columns")

    old_score = score(rows)
    encoded = [encode_path(row["path"]) for row in rows]
    output_lines, elapsed = run_twsearch(
        args.twsearch,
        args.definition,
        encoded,
        args.stage,
        args.max_depth,
        args.memory_mb,
        args.threads,
    )
    rewritten = [
        {"initial_state_id": row["initial_state_id"], "path": decode_path(line, bases)}
        for row, line in zip(rows, output_lines)
    ]
    new_score = score(rewritten)
    if new_score > old_score:
        raise RuntimeError(f"twsearch increased score from {old_score} to {new_score}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["initial_state_id", "path"])
        writer.writeheader()
        writer.writerows(rewritten)

    validation = validate_submission(
        Puzzle.load(args.puzzle_info),
        load_tests(args.test),
        load_submission(args.output),
    )
    if not validation.valid:
        args.output.unlink(missing_ok=True)
        raise RuntimeError("rewritten submission is invalid: " + "; ".join(validation.failures[:5]))
    changed = sum(old["path"] != new["path"] for old, new in zip(rows, rewritten))
    print(
        f"stage={args.stage} puzzles={len(rows)} old_score={old_score} "
        f"new_score={new_score} saved={old_score - new_score} changed={changed} "
        f"seconds={elapsed:.3f} valid={validation.valid}"
    )


if __name__ == "__main__":
    main()
