#!/usr/bin/env python3
"""Exact whole-cube frame optimizer over all 48 cube isometries.

The usual rotation-frame optimizer carries one of the 24 physical cube
rotations between maximal same-axis blocks.  The full generator normalizer has
48 target-compatible isometries: 24 proper rotations and 24 improper views.
This program explicitly tests every normalizer view.  A path is conjugated
into a view, optimized by the exact 24-state frame dynamic program, and then
conjugated back.  The chosen result must have the identical complete 96-point
permutation, and the final submission is replay-validated.

Improper and proper frames cannot be joined by a legal cube word, so testing
the 48 conjugated start/end views is the exact way to include reflections; it
does not pretend that a reflection itself is a legal zero-cost move.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from cube_tools import (
    Puzzle,
    compose,
    identity,
    inverse,
    load_submission,
    load_test,
    validate_submission,
    write_submission,
)
from neural_reconnect_search import build_rotations
from optimize_submission import normalize_commuting


Permutation = Tuple[int, ...]
PathTokens = Tuple[str, ...]


def permutation_power(permutation: Permutation, exponent: int) -> Permutation:
    result = identity(len(permutation))
    for _ in range(exponent % 4):
        result = compose(result, permutation)
    return result


def whole_rotation(
    axis: str, generators: Mapping[str, Permutation]
) -> Permutation:
    result = identity(len(next(iter(generators.values()))))
    for layer in range(4):
        result = compose(result, generators[f"{axis}{layer}"])
    return result


def build_orientation_group(
    rotations: Mapping[str, Permutation],
) -> Tuple[List[Permutation], Dict[Permutation, int]]:
    start = identity(len(next(iter(rotations.values()))))
    orientations = [start]
    indices = {start: 0}
    queue = deque([start])
    steps = list(rotations.values()) + [inverse(value) for value in rotations.values()]
    while queue:
        current = queue.popleft()
        for step in steps:
            following = compose(current, step)
            if following in indices:
                continue
            indices[following] = len(orientations)
            orientations.append(following)
            queue.append(following)
    if len(orientations) != 24:
        raise ValueError(f"expected 24 proper orientations, found {len(orientations)}")
    return orientations, indices


@dataclass(frozen=True)
class AxisBlock:
    axis: str
    powers: Tuple[int, int, int, int]


def axis_blocks(tokens: Sequence[str]) -> List[AxisBlock]:
    blocks: List[AxisBlock] = []
    index = 0
    while index < len(tokens):
        axis = tokens[index].lstrip("-")[0]
        powers = [0, 0, 0, 0]
        stop = index
        while stop < len(tokens) and tokens[stop].lstrip("-")[0] == axis:
            token = tokens[stop]
            layer = int(token.lstrip("-")[1:])
            powers[layer] = (powers[layer] + (-1 if token.startswith("-") else 1)) % 4
            stop += 1
        blocks.append(AxisBlock(axis, tuple(powers)))
        index = stop
    return blocks


def residual_tokens(block: AxisBlock, factored_power: int) -> PathTokens:
    result: List[str] = []
    for layer, original_power in enumerate(block.powers):
        power = (original_power - factored_power) % 4
        base = f"{block.axis}{layer}"
        if power == 1:
            result.append(base)
        elif power == 2:
            result.extend((base, base))
        elif power == 3:
            result.append(f"-{base}")
    return tuple(result)


class RotationFrameOptimizer:
    """Precomputed exact 24-state frame DP used inside each isometry view."""

    def __init__(self, generators: Mapping[str, Permutation]) -> None:
        self.generators = generators
        rotations = {axis: whole_rotation(axis, generators) for axis in "frd"}
        self.orientations, orientation_indices = build_orientation_group(rotations)
        rotation_powers = {
            axis: [permutation_power(rotation, exponent) for exponent in range(4)]
            for axis, rotation in rotations.items()
        }
        self.next_state: Dict[str, List[List[int]]] = {}
        for axis in "frd":
            self.next_state[axis] = [
                [
                    orientation_indices[compose(orientation, rotation_powers[axis][exponent])]
                    for exponent in range(4)
                ]
                for orientation in self.orientations
            ]

        generator_by_permutation = {
            tuple(value): name for name, value in generators.items()
        }
        self.conjugated: List[Dict[str, str]] = []
        for orientation in self.orientations:
            orientation_inverse = inverse(orientation)
            mapped: Dict[str, str] = {}
            for name, move in generators.items():
                conjugated = compose(compose(orientation, move), orientation_inverse)
                try:
                    mapped[name] = generator_by_permutation[tuple(conjugated)]
                except KeyError as exc:
                    raise ValueError(
                        f"proper orientation does not normalize generator {name}"
                    ) from exc
            self.conjugated.append(mapped)

    def optimize(self, tokens: Sequence[str]) -> PathTokens:
        blocks = axis_blocks(tokens)
        state_count = len(self.orientations)
        infinity = 1 << 30
        costs = [infinity] * state_count
        costs[0] = 0
        parents: List[List[Tuple[int, int] | None]] = []

        for block in blocks:
            transition_costs = [
                len(residual_tokens(block, exponent)) for exponent in range(4)
            ]
            following = [infinity] * state_count
            following_parents: List[Tuple[int, int] | None] = [None] * state_count
            for source, cost in enumerate(costs):
                if cost == infinity:
                    continue
                for exponent in range(4):
                    destination = self.next_state[block.axis][source][exponent]
                    candidate = cost + transition_costs[exponent]
                    if candidate < following[destination]:
                        following[destination] = candidate
                        following_parents[destination] = (source, exponent)
            costs = following
            parents.append(following_parents)

        choices = [0] * len(blocks)
        state = 0
        for block_index in range(len(blocks) - 1, -1, -1):
            parent = parents[block_index][state]
            if parent is None:
                raise RuntimeError("broken frame-DP backpointer")
            state, choices[block_index] = parent
        if state != 0:
            raise RuntimeError("frame DP did not start at identity")

        state = 0
        output: List[str] = []
        for block, exponent in zip(blocks, choices, strict=True):
            mapping = self.conjugated[state]
            output.extend(mapping[token] for token in residual_tokens(block, exponent))
            state = self.next_state[block.axis][state][exponent]
        if state != 0:
            raise RuntimeError("frame DP did not return to identity")
        return tuple(output)


def path_permutation(
    tokens: Sequence[str], generators: Mapping[str, Permutation]
) -> Permutation:
    result = identity(len(next(iter(generators.values()))))
    for token in tokens:
        result = compose(result, generators[token])
    return tuple(result)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--puzzle-info", required=True, type=Path)
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = monotonic()
    puzzle = Puzzle.load(args.puzzle_info)
    tests = load_test(args.test)
    original = load_submission(args.input)
    before = validate_submission(puzzle, tests, original)

    move_names = tuple(puzzle.generators)
    move_index = {name: index for index, name in enumerate(move_names)}
    views = build_rotations(puzzle, move_names, include_reflections=True)
    if len(views) != 48:
        raise RuntimeError(f"expected 48 verified isometries, found {len(views)}")
    frame_optimizer = RotationFrameOptimizer(puzzle.generators)

    optimized: Dict[str, List[str]] = {}
    wins: Counter[int] = Counter()
    view_lengths: Counter[int] = Counter()
    changes: List[dict] = []
    for instance_id in sorted(tests, key=int):
        source = tuple(original[instance_id])
        source_indices = np.asarray([move_index[name] for name in source], dtype=np.int16)
        source_permutation = path_permutation(source, puzzle.generators)
        best = source
        best_view = -1
        lengths: List[int] = []
        for view in views:
            viewed_indices = view.move_to_rotated[source_indices]
            viewed = tuple(move_names[int(index)] for index in viewed_indices)
            candidate_viewed = frame_optimizer.optimize(viewed)
            candidate_viewed_indices = np.asarray(
                [move_index[name] for name in candidate_viewed], dtype=np.int16
            )
            candidate_indices = view.move_from_rotated[candidate_viewed_indices]
            candidate = tuple(move_names[int(index)] for index in candidate_indices)
            candidate = tuple(normalize_commuting(candidate, puzzle.generators))
            lengths.append(len(candidate))
            if len(candidate) < len(best):
                if path_permutation(candidate, puzzle.generators) != source_permutation:
                    raise RuntimeError(
                        f"isometry/frame rewrite changed permutation for id {instance_id}"
                    )
                best = candidate
                best_view = view.index
        if len(set(lengths)) == 1:
            view_lengths[1] += 1
        else:
            view_lengths[len(set(lengths))] += 1
        optimized[instance_id] = list(best)
        if len(best) < len(source):
            wins[best_view] += 1
            changes.append(
                {
                    "id": instance_id,
                    "before": len(source),
                    "after": len(best),
                    "saved": len(source) - len(best),
                    "view": best_view,
                    "view_name": views[best_view].name,
                }
            )

    after = validate_submission(puzzle, tests, optimized)
    if after.total_moves > before.total_moves:
        raise RuntimeError("isometry-frame optimizer increased score")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    write_submission(
        args.output,
        ((instance_id, optimized[instance_id]) for instance_id in sorted(tests, key=int)),
    )
    report = {
        "status": "complete",
        "input": str(args.input),
        "output": str(args.output),
        "before": before.total_moves,
        "after": after.total_moves,
        "saved": before.total_moves - after.total_moves,
        "rows": after.rows,
        "isometries": len(views),
        "proper": 24,
        "improper": 24,
        "rows_by_distinct_view_lengths": dict(sorted(view_lengths.items())),
        "winning_views": dict(sorted(wins.items())),
        "changes": changes,
        "seconds": round(monotonic() - started, 3),
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report["sha256"] = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(json.dumps({key: value for key, value in report.items() if key != "changes"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
