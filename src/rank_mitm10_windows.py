#!/usr/bin/env python3
"""Rank 11/12-move incumbent windows for exact colour-state MITM.

The public scalar agents estimate distance to the common solved state.  For a
path window [i, j),

    (j-i) - (V(state_i) - V(state_j))

is a useful model-based estimate of wasted progress.  This script evaluates
all checkpoints in one batched pass, ranks windows by the median estimate of
several independently trained agents, and emits a deterministic TSV consumed
by ``colored_window_mitm10``.  It performs no network or upload operations.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cube444 import Puzzle, load_submission, load_tests
from scalar_beam444 import load_scalar_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--puzzle-info", type=Path, default=Path("data/puzzle_info.json"))
    parser.add_argument("--test", type=Path, default=Path("data/test.csv"))
    parser.add_argument("--lengths", default="11,12")
    parser.add_argument("--top", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--max-per-row",
        type=int,
        default=3,
        help="limit correlated windows from any one puzzle",
    )
    args = parser.parse_args()

    lengths = sorted({int(value) for value in args.lengths.split(",") if value.strip()})
    if not lengths or min(lengths) < 2 or max(lengths) > 12:
        parser.error("--lengths must be comma-separated integers in [2, 12]")
    if args.top <= 0 or args.batch_size <= 0 or args.max_per_row <= 0:
        parser.error("--top, --batch-size, and --max-per-row must be positive")

    puzzle = Puzzle.load(args.puzzle_info)
    tests = load_tests(args.test)
    baseline = load_submission(args.baseline)
    move_names = tuple(puzzle.generators)
    move_perms = {
        name: np.asarray(puzzle.generators[name], dtype=np.int16) for name in move_names
    }

    checkpoint_states: list[np.ndarray] = []
    offsets: dict[int, tuple[int, int]] = {}
    for puzzle_id in range(len(tests)):
        key = str(puzzle_id)
        path = baseline[key]
        state = np.asarray(tests[key], dtype=np.uint8)
        first = len(checkpoint_states)
        checkpoint_states.append(state.copy())
        for move in path:
            state = state[move_perms[move]]
            checkpoint_states.append(state.copy())
        if not np.array_equal(state, np.asarray(puzzle.central_state, dtype=np.uint8)):
            raise ValueError(f"baseline fails exact replay for id {puzzle_id}")
        offsets[puzzle_id] = (first, len(path) + 1)

    states = np.ascontiguousarray(np.stack(checkpoint_states))
    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    predictions: list[np.ndarray] = []
    for checkpoint in args.checkpoint:
        model, contract = load_scalar_model(checkpoint, device)
        if contract.state_size != states.shape[1]:
            raise ValueError(f"checkpoint {checkpoint} has incompatible state size")
        chunks: list[np.ndarray] = []
        with torch.inference_mode():
            for first in range(0, len(states), args.batch_size):
                batch = torch.from_numpy(states[first : first + args.batch_size]).to(device)
                chunks.append(model(batch).float().cpu().numpy())
        predictions.append(np.concatenate(chunks))
    values = np.stack(predictions, axis=1)

    ranked: list[dict[str, object]] = []
    for puzzle_id in range(len(tests)):
        key = str(puzzle_id)
        path_length = len(baseline[key])
        offset, count = offsets[puzzle_id]
        row_values = values[offset : offset + count]
        for length in lengths:
            for start in range(0, path_length - length + 1):
                end = start + length
                agent_slack = length - (row_values[start] - row_values[end])
                ranked.append(
                    {
                        "puzzle_id": puzzle_id,
                        "start": start,
                        "end": end,
                        "length": length,
                        "score": float(np.median(agent_slack)),
                        "mean_score": float(np.mean(agent_slack)),
                        "agent_scores": [float(value) for value in agent_slack],
                        "path_length": path_length,
                        "is_prefix": start == 0,
                        "is_suffix": end == path_length,
                    }
                )

    ranked.sort(
        key=lambda item: (
            -float(item["score"]),
            -float(item["mean_score"]),
            -int(item["length"]),
            int(item["puzzle_id"]),
            int(item["start"]),
        )
    )
    selected: list[dict[str, object]] = []
    per_row: dict[int, int] = {}
    for item in ranked:
        puzzle_id = int(item["puzzle_id"])
        if per_row.get(puzzle_id, 0) >= args.max_per_row:
            continue
        selected.append(item)
        per_row[puzzle_id] = per_row.get(puzzle_id, 0) + 1
        if len(selected) >= args.top:
            break

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("initial_state_id", "start", "end", "model_score"))
        for item in selected:
            writer.writerow(
                (
                    item["puzzle_id"],
                    item["start"],
                    item["end"],
                    f"{float(item['score']):.9f}",
                )
            )

    report = {
        "baseline": str(args.baseline),
        "checkpoints": [str(path) for path in args.checkpoint],
        "device": str(device),
        "state_count": len(states),
        "candidate_window_count": len(ranked),
        "selected_window_count": len(selected),
        "selected_puzzle_count": len(per_row),
        "lengths": lengths,
        "max_per_row": args.max_per_row,
        "selected": selected,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "report": str(args.report),
                "states": len(states),
                "windows": len(ranked),
                "selected": len(selected),
                "puzzles": len(per_row),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
