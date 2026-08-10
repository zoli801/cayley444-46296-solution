#!/usr/bin/env python3
"""Find shorter solutions in the union of locally validated trajectories.

Every input path is replayed before it is admitted.  The resulting colored
cube states form a graph whose edges are legal quarter turns.  A single BFS
from the solved state then finds the shortest suffix available to every test
state, including paths assembled from pieces of different candidate runs.

With ``--induced-edges``, all one-move connections between collected states
are added, even when that edge did not occur in an input path.
"""

from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path
from typing import Iterable

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


def apply_bytes(state: bytes, permutation: tuple[int, ...]) -> bytes:
    return bytes(state[index] for index in permutation)


def add_edge(
    adjacency: list[set[tuple[int, str]]],
    source: int,
    target: int,
    move: str,
) -> None:
    adjacency[source].add((target, move))
    adjacency[target].add((source, inverse_move(move)))


def build_graph(
    puzzle: Puzzle,
    tests: dict[str, tuple[int, ...]],
    candidates: Iterable[tuple[str, tuple[str, ...]]],
    *,
    induced_edges: bool,
) -> tuple[list[bytes], list[set[tuple[int, str]]], dict[bytes, int]]:
    states: list[bytes] = []
    adjacency: list[set[tuple[int, str]]] = []
    indexes: dict[bytes, int] = {}

    def intern(state: bytes) -> int:
        existing = indexes.get(state)
        if existing is not None:
            return existing
        index = len(states)
        indexes[state] = index
        states.append(state)
        adjacency.append(set())
        return index

    for state_id, moves in candidates:
        state = bytes(tests[state_id])
        source = intern(state)
        for move in moves:
            target_state = apply_bytes(state, puzzle.generators[move])
            target = intern(target_state)
            add_edge(adjacency, source, target, move)
            state = target_state
            source = target
        if state != bytes(puzzle.central_state):
            raise AssertionError(f"validated candidate {state_id} no longer solves")

    if induced_edges:
        # The state set is fixed before this pass.  We add only connections
        # between already collected checkpoints, so this is not an expanding
        # brute-force search.
        frozen_count = len(states)
        moves = tuple(sorted(puzzle.generators.items()))
        for source in range(frozen_count):
            state = states[source]
            for move, permutation in moves:
                target_state = apply_bytes(state, permutation)
                target = indexes.get(target_state)
                if target is not None and source <= target:
                    add_edge(adjacency, source, target, move)

    return states, adjacency, indexes


def shortest_paths_to_goal(
    states: list[bytes],
    adjacency: list[set[tuple[int, str]]],
    goal: bytes,
    indexes: dict[bytes, int],
) -> tuple[list[int], list[tuple[int, str] | None]]:
    goal_index = indexes[goal]
    distance = [-1] * len(states)
    parent: list[tuple[int, str] | None] = [None] * len(states)
    distance[goal_index] = 0
    queue: deque[int] = deque([goal_index])
    while queue:
        current = queue.popleft()
        for neighbor, move_from_current in sorted(
            adjacency[current], key=lambda item: (item[1], item[0])
        ):
            if distance[neighbor] >= 0:
                continue
            distance[neighbor] = distance[current] + 1
            # The graph stores both directions.  Traversing current -> neighbor
            # means the solution direction is neighbor -> current.
            parent[neighbor] = (current, inverse_move(move_from_current))
            queue.append(neighbor)
    return distance, parent


def reconstruct(
    start: int,
    parent: list[tuple[int, str] | None],
) -> tuple[str, ...]:
    moves: list[str] = []
    current = start
    while parent[current] is not None:
        current, move = parent[current]
        moves.append(move)
    return tuple(moves)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path)
    parser.add_argument(
        "--baseline", type=Path, default=Path("work/submissions/baseline_46718.csv")
    )
    parser.add_argument(
        "--puzzle-info", type=Path, default=Path("data/puzzle_info.json")
    )
    parser.add_argument("--test", type=Path, default=Path("data/test.csv"))
    parser.add_argument(
        "--output", type=Path, default=Path("work/submissions/trajectory_graph.csv")
    )
    parser.add_argument(
        "--report", type=Path, default=Path("work/submissions/trajectory_graph.json")
    )
    parser.add_argument("--induced-edges", action="store_true")
    parser.add_argument(
        "--max-slack",
        type=int,
        help=(
            "discard candidates longer than baseline length plus this many moves; "
            "useful for excluding sample paths while retaining search diversity"
        ),
    )
    parser.add_argument("--max-member-mib", type=int, default=64)
    parser.add_argument("--max-zip-depth", type=int, default=3)
    args = parser.parse_args()

    puzzle = Puzzle.load(args.puzzle_info)
    tests = load_tests(args.test)
    baseline = load_submission(args.baseline)
    baseline_result = validate_submission(puzzle, tests, baseline)
    if not baseline_result.valid:
        raise ValueError("baseline is invalid: " + "; ".join(baseline_result.failures))

    stats = IngestStats()
    replay_cache: dict[tuple[str, tuple[str, ...]], str | None] = {}
    unique: dict[tuple[str, tuple[str, ...]], None] = {}
    roots = list(args.inputs)
    for source in iter_csv_sources(
        args.baseline,
        roots,
        stats,
        max_member_bytes=args.max_member_mib * 1024 * 1024,
        max_zip_depth=args.max_zip_depth,
    ):
        candidates, _ = ingest_source(source, puzzle, tests, stats, replay_cache)
        for candidate in candidates:
            if (
                args.max_slack is not None
                and candidate.length > len(baseline[candidate.state_id]) + args.max_slack
            ):
                continue
            unique[(candidate.state_id, candidate.moves)] = None

    states, adjacency, indexes = build_graph(
        puzzle,
        tests,
        unique,
        induced_edges=args.induced_edges,
    )
    distance, parent = shortest_paths_to_goal(
        states, adjacency, bytes(puzzle.central_state), indexes
    )

    solutions: dict[str, tuple[str, ...]] = {}
    for state_id, initial in tests.items():
        start = indexes[bytes(initial)]
        if distance[start] < 0:
            raise AssertionError(f"test state {state_id} is disconnected from goal")
        solutions[state_id] = reconstruct(start, parent)

    ordered_ids = sorted(tests, key=int)
    atomic_write_text(
        args.output,
        csv_text(
            ("initial_state_id", "path"),
            ((state_id, ".".join(solutions[state_id])) for state_id in ordered_ids),
        ),
    )
    result = validate_submission(puzzle, tests, load_submission(args.output))
    if not result.valid:
        raise AssertionError("graph output failed replay: " + "; ".join(result.failures))

    edge_count = sum(len(items) for items in adjacency)
    improved = [
        state_id
        for state_id in ordered_ids
        if len(solutions[state_id]) < len(baseline[state_id])
    ]
    report = {
        "baseline": str(args.baseline),
        "baseline_score": baseline_result.score,
        "candidate_path_count": len(unique),
        "directed_edge_count": edge_count,
        "gain": baseline_result.score - result.score,
        "improved_id_count": len(improved),
        "improved_ids": improved,
        "induced_edges": args.induced_edges,
        "node_count": len(states),
        "output": str(args.output),
        "score": result.score,
    }
    atomic_write_text(args.report, json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        f"candidates={len(unique)} nodes={len(states)} edges={edge_count} "
        f"baseline={baseline_result.score} score={result.score} "
        f"gain={baseline_result.score - result.score} improved={len(improved)}"
    )


if __name__ == "__main__":
    main()
