#!/usr/bin/env python3
"""Compute-bounded local CayleyPy 4x4 scalar-agent beam search.

This is a small MPS/CPU adaptation of the public khoruzhii/cayleypy-cube
Pilgrim scalar network and beam-search algorithm, commit
f02604fa7b665b82e5fbe5692b336a4fe4a01bdc (MIT, copyright 2024
Khoruzhii Kirill), modified for this reproducibility bundle. Search is
heuristic, but every emitted path is replayed exactly with the competition
permutations. Nothing is uploaded by this program.
"""

from __future__ import annotations

import argparse
from collections import deque
import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from cube444 import Puzzle, load_submission, load_tests


class ResidualBlock(nn.Module):
    """Checkpoint-compatible public Pilgrim residual block."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(width, width)
        self.bn1 = nn.BatchNorm1d(width)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.0)
        self.fc2 = nn.Linear(width, width)
        self.bn2 = nn.BatchNorm1d(width)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.relu(self.bn1(self.fc1(inputs)))
        output = self.bn2(self.fc2(output))
        return self.relu(output + inputs)


class ScalarPilgrim(nn.Module):
    """Exact inference architecture used by the public Cube4 scalar agents."""

    def __init__(
        self,
        state_size: int,
        num_classes: int,
        hidden1: int,
        hidden2: int,
        residual_count: int,
    ) -> None:
        super().__init__()
        self.state_size = state_size
        self.num_classes = num_classes
        self.input_layer = nn.Linear(state_size * num_classes, hidden1)
        self.bn1 = nn.BatchNorm1d(hidden1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.0)
        self.hidden_layer = nn.Linear(hidden1, hidden2)
        self.bn2 = nn.BatchNorm1d(hidden2)
        self.residual_blocks = nn.ModuleList(
            ResidualBlock(hidden2) for _ in range(residual_count)
        )
        self.output_layer = nn.Linear(hidden2, 1)
        self.inference_dtype = torch.float32

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        encoded = F.one_hot(states.long(), num_classes=self.num_classes)
        values = encoded.reshape(states.shape[0], -1).to(self.inference_dtype)
        values = self.relu(self.bn1(self.input_layer(values)))
        values = self.relu(self.bn2(self.hidden_layer(values)))
        for block in self.residual_blocks:
            values = block(values)
        return self.output_layer(values).flatten()


@dataclass(frozen=True)
class ModelContract:
    state_size: int
    num_classes: int
    hidden1: int
    hidden2: int
    residual_count: int


@dataclass
class SearchResult:
    solution: tuple[int, ...] | None
    runtime_seconds: float
    depths_completed: int
    expanded_candidates: int
    scored_candidates: int
    peak_unique_candidates: int
    stop_reason: str


@dataclass(frozen=True)
class AttemptReport:
    puzzle_id: int
    incumbent_length: int
    prefix_length: int
    incumbent_suffix_length: int
    variant: str
    beam_width: int
    solution_length: int | None
    improved_by: int
    runtime_seconds: float
    depths_completed: int
    expanded_candidates: int
    scored_candidates: int
    peak_unique_candidates: int
    stop_reason: str


@dataclass(frozen=True)
class RotationView:
    """One target-preserving physical cube orientation."""

    index: int
    name: str
    position: np.ndarray
    color_map: np.ndarray
    move_to_rotated: np.ndarray
    move_from_rotated: np.ndarray

    def transform_state(self, state: Sequence[int] | np.ndarray) -> np.ndarray:
        values = np.asarray(state, dtype=np.uint8)
        if values.shape != (len(self.position),):
            raise ValueError(f"rotation expected one state, got shape {values.shape}")
        return self.color_map[values[self.position]]

    def transform_states(self, states: np.ndarray) -> np.ndarray:
        values = np.asarray(states, dtype=np.uint8)
        if values.ndim != 2 or values.shape[1] != len(self.position):
            raise ValueError(f"rotation expected a state batch, got shape {values.shape}")
        return self.color_map[values[:, self.position]]


def unwrap_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model", "net", "module"):
            nested = checkpoint.get(key)
            if isinstance(nested, dict):
                checkpoint = nested
                break
    if not isinstance(checkpoint, dict) or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in checkpoint.items()
    ):
        raise ValueError("checkpoint is not a tensor state_dict")
    return {
        (key[len("_orig_mod.") :] if key.startswith("_orig_mod.") else key): value
        for key, value in checkpoint.items()
    }


def infer_model_contract(state_dict: dict[str, torch.Tensor]) -> ModelContract:
    required = {
        "input_layer.weight",
        "hidden_layer.weight",
        "output_layer.weight",
        "bn1.running_mean",
        "bn2.running_mean",
    }
    missing = sorted(required - state_dict.keys())
    if missing:
        raise ValueError(f"unsupported scalar checkpoint; missing keys: {missing}")
    hidden1, flattened = state_dict["input_layer.weight"].shape
    hidden2, hidden_input = state_dict["hidden_layer.weight"].shape
    output_dim, output_input = state_dict["output_layer.weight"].shape
    if hidden_input != hidden1 or output_dim != 1 or output_input != hidden2:
        raise ValueError("inconsistent scalar checkpoint linear dimensions")
    residual_indices = sorted(
        {
            int(key.split(".")[1])
            for key in state_dict
            if key.startswith("residual_blocks.") and key.endswith(".fc1.weight")
        }
    )
    if residual_indices != list(range(len(residual_indices))):
        raise ValueError(f"non-contiguous residual blocks: {residual_indices}")
    # Cube4 checkpoints use 96 stickers and six central-state labels.  Infer
    # state_size after fixing the only contiguous alphabet compatible with 576.
    num_classes = 6
    if flattened % num_classes:
        raise ValueError(f"input width {flattened} is not divisible by six classes")
    return ModelContract(
        state_size=flattened // num_classes,
        num_classes=num_classes,
        hidden1=hidden1,
        hidden2=hidden2,
        residual_count=len(residual_indices),
    )


def load_scalar_model(
    checkpoint_path: Path, device: torch.device
) -> tuple[ScalarPilgrim, ModelContract]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = unwrap_state_dict(checkpoint)
    contract = infer_model_contract(state_dict)
    model = ScalarPilgrim(
        contract.state_size,
        contract.num_classes,
        contract.hidden1,
        contract.hidden2,
        contract.residual_count,
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    if device.type in {"mps", "cuda"}:
        model.half()
        model.inference_dtype = torch.float16
    model.to(device)
    return model, contract


def inverse_move_indices(permutations: np.ndarray) -> np.ndarray:
    lookup = {row.tobytes(): index for index, row in enumerate(permutations)}
    result = np.empty(len(permutations), dtype=np.int16)
    for index, permutation in enumerate(permutations):
        inverse = np.empty_like(permutation)
        inverse[permutation] = np.arange(len(permutation), dtype=permutation.dtype)
        try:
            result[index] = lookup[inverse.tobytes()]
        except KeyError as error:
            raise ValueError(f"generator index {index} has no inverse") from error
    return result


def invert_moves(moves: Sequence[int], inverse_indices: np.ndarray) -> tuple[int, ...]:
    return tuple(int(inverse_indices[move]) for move in reversed(moves))


def replay_indices(
    start: Sequence[int], moves: Sequence[int], permutations: np.ndarray
) -> np.ndarray:
    state = np.asarray(start, dtype=np.uint8)
    for move in moves:
        state = state[permutations[move]]
    return state


def compose_permutations(
    first: Sequence[int], second: Sequence[int]
) -> tuple[int, ...]:
    """Permutation for applying ``first`` and then ``second``."""
    return tuple(int(first[index]) for index in second)


def inverse_permutation(permutation: Sequence[int]) -> tuple[int, ...]:
    result = [0] * len(permutation)
    for destination, source in enumerate(permutation):
        result[int(source)] = destination
    return tuple(result)


def target_color_relabel(
    target: Sequence[int], position: Sequence[int]
) -> tuple[int, ...]:
    """Color relabeling that makes a rotated solved state exactly canonical."""
    rotated = tuple(int(target[index]) for index in position)
    mapping: dict[int, int] = {}
    for source, destination in zip(rotated, target, strict=True):
        previous = mapping.setdefault(source, int(destination))
        if previous != int(destination):
            raise ValueError("whole-cube rotation has inconsistent target colors")
    colors = sorted(set(map(int, target)))
    if sorted(mapping) != colors or sorted(mapping.values()) != colors:
        raise ValueError("rotation target color map is not bijective")
    return tuple(mapping[color] for color in colors)


def build_rotation_views(
    puzzle: Puzzle, move_names: Sequence[str]
) -> tuple[RotationView, ...]:
    """Derive and verify the 24 proper target-preserving cube rotations."""
    size = len(puzzle.central_state)
    identity = tuple(range(size))
    whole_paths = {
        "F": ("f0", "f1", "f2", "f3"),
        "R": ("r0", "r1", "r2", "r3"),
        "D": ("d0", "d1", "d2", "d3"),
    }
    whole: dict[str, tuple[int, ...]] = {}
    for axis, path in whole_paths.items():
        permutation = identity
        for move in path:
            permutation = compose_permutations(
                permutation, puzzle.generators[move]
            )
        whole[axis] = permutation

    words: dict[tuple[int, ...], tuple[str, ...]] = {identity: ()}
    queue = deque([identity])
    while queue:
        current = queue.popleft()
        for axis in ("F", "R", "D"):
            following = compose_permutations(current, whole[axis])
            if following in words:
                continue
            words[following] = words[current] + (axis,)
            queue.append(following)
    if len(words) != 24:
        raise ValueError(f"expected 24 cube rotations, derived {len(words)}")

    generator_lookup = {
        tuple(map(int, permutation)): name
        for name, permutation in puzzle.generators.items()
    }
    move_index = {name: index for index, name in enumerate(move_names)}
    views: list[RotationView] = []
    target = np.asarray(puzzle.central_state, dtype=np.uint8)
    for view_index, (position, word) in enumerate(words.items()):
        position_inverse = inverse_permutation(position)
        mapped_names: list[str] = []
        for move_name in move_names:
            move = puzzle.generators[move_name]
            conjugated = tuple(
                position_inverse[move[position[index]]] for index in range(size)
            )
            mapped = generator_lookup.get(conjugated)
            if mapped is None:
                raise ValueError(
                    f"rotation {view_index} does not normalize move {move_name}"
                )
            mapped_names.append(mapped)
        move_to_rotated = np.asarray(
            [move_index[name] for name in mapped_names], dtype=np.int16
        )
        move_from_rotated = np.empty(len(move_names), dtype=np.int16)
        move_from_rotated[move_to_rotated] = np.arange(
            len(move_names), dtype=np.int16
        )
        view = RotationView(
            index=view_index,
            name="I" if not word else "".join(word),
            position=np.asarray(position, dtype=np.int64),
            color_map=np.asarray(
                target_color_relabel(puzzle.central_state, position), dtype=np.uint8
            ),
            move_to_rotated=move_to_rotated,
            move_from_rotated=move_from_rotated,
        )
        if not np.array_equal(view.transform_state(target), target):
            raise RuntimeError(f"rotation {view.name} does not preserve the target")
        if not np.array_equal(
            view.move_from_rotated[view.move_to_rotated],
            np.arange(len(move_names), dtype=np.int16),
        ):
            raise RuntimeError(f"rotation {view.name} move maps are not inverse")
        views.append(view)
    if len(views) != 24 or sum(view.name == "I" for view in views) != 1:
        raise RuntimeError("rotation view construction failed")
    return tuple(views)


def exact_unique_indices(states: np.ndarray) -> np.ndarray:
    """Return one deterministic index for every exact uint8 state."""
    contiguous = np.ascontiguousarray(states)
    packed = contiguous.view(np.dtype((np.void, contiguous.shape[1]))).reshape(-1)
    _, first = np.unique(packed, return_index=True)
    return np.sort(first)


def build_touch_table(
    goal: np.ndarray,
    permutations: np.ndarray,
    inverse_indices: np.ndarray,
    radius: int,
) -> dict[bytes, tuple[int, ...]]:
    """Map exact radius-R goal neighbors to shortest suffixes back to goal."""
    root = np.asarray(goal, dtype=np.uint8)
    table: dict[bytes, tuple[int, ...]] = {root.tobytes(): ()}
    frontier = [root]
    for _ in range(radius):
        next_frontier: list[np.ndarray] = []
        for state in frontier:
            for move, permutation in enumerate(permutations):
                neighbor = state[permutation]
                key = neighbor.tobytes()
                if key in table:
                    continue
                table[key] = (int(inverse_indices[move]),) + table[state.tobytes()]
                next_frontier.append(neighbor)
        frontier = next_frontier
    # Fail closed if construction direction was accidentally reversed.
    for key, suffix in table.items():
        state = np.frombuffer(key, dtype=np.uint8)
        if not np.array_equal(replay_indices(state, suffix, permutations), root):
            raise RuntimeError("invalid touch-table suffix")
    return table


def trace_selected_state(
    state_index: int,
    parent_history: Sequence[np.ndarray],
    move_history: Sequence[np.ndarray],
) -> tuple[int, ...]:
    reversed_moves: list[int] = []
    index = int(state_index)
    for parents, moves in zip(reversed(parent_history), reversed(move_history)):
        reversed_moves.append(int(moves[index]))
        index = int(parents[index])
    reversed_moves.reverse()
    return tuple(reversed_moves)


class ScalarBeamSearcher:
    def __init__(
        self,
        model: ScalarPilgrim | Sequence[ScalarPilgrim],
        device: torch.device,
        permutations: np.ndarray,
        move_names: Sequence[str],
        goal: np.ndarray,
        *,
        inference_batch: int,
        touch_radius: int,
    ) -> None:
        if isinstance(model, ScalarPilgrim):
            self.models = (model,)
        else:
            self.models = tuple(model)
        if not self.models:
            raise ValueError("at least one scalar model is required")
        self.device = device
        self.permutations = np.asarray(permutations, dtype=np.int16)
        self.move_names = tuple(move_names)
        self.goal = np.asarray(goal, dtype=np.uint8)
        self.inference_batch = inference_batch
        self.inverse_indices = inverse_move_indices(self.permutations)
        self.touch_table = build_touch_table(
            self.goal, self.permutations, self.inverse_indices, touch_radius
        )
        self.move_axis = np.asarray(
            [name.lstrip("-")[0] for name in self.move_names], dtype="U1"
        )
        self.move_layer = np.asarray(
            [int(name.lstrip("-")[1:]) for name in self.move_names], dtype=np.int16
        )
        self.move_order = np.asarray(
            [
                self.move_layer[index] * 2 + int(name.startswith("-"))
                for index, name in enumerate(self.move_names)
            ],
            dtype=np.int16,
        )

    def score(self, states: np.ndarray) -> np.ndarray:
        """Return [state, agent] scalar predictions for quota selection."""
        scores = np.empty((len(states), len(self.models)), dtype=np.float32)
        with torch.inference_mode():
            for first in range(0, len(states), self.inference_batch):
                last = min(first + self.inference_batch, len(states))
                batch = torch.from_numpy(np.ascontiguousarray(states[first:last])).to(
                    self.device, non_blocking=False
                )
                for agent_index, model in enumerate(self.models):
                    prediction = model(batch).float().cpu().numpy()
                    scores[first:last, agent_index] = prediction
        return scores

    @staticmethod
    def select_agent_quotas(scores: np.ndarray, keep_count: int) -> np.ndarray:
        """Union equal per-agent top-score quotas, then fill by mean score."""
        if scores.ndim != 2 or keep_count <= 0 or keep_count > len(scores):
            raise ValueError("invalid ensemble selection inputs")
        agent_count = scores.shape[1]
        if agent_count == 1:
            values = scores[:, 0]
            if keep_count < len(values):
                selected = np.argpartition(values, keep_count - 1)[:keep_count]
            else:
                selected = np.arange(len(values))
            return selected[np.argsort(values[selected], kind="stable")]

        quotas = [keep_count // agent_count] * agent_count
        for index in range(keep_count % agent_count):
            quotas[index] += 1
        selected_mask = np.zeros(len(scores), dtype=bool)
        for agent_index, quota in enumerate(quotas):
            if quota == 0:
                continue
            values = scores[:, agent_index]
            if quota < len(values):
                chosen = np.argpartition(values, quota - 1)[:quota]
            else:
                chosen = np.arange(len(values))
            selected_mask[chosen] = True
        selected = np.flatnonzero(selected_mask)
        mean_score = scores.mean(axis=1)
        if len(selected) < keep_count:
            remaining = np.flatnonzero(~selected_mask)
            needed = keep_count - len(selected)
            if needed < len(remaining):
                fill = remaining[
                    np.argpartition(mean_score[remaining], needed - 1)[:needed]
                ]
            else:
                fill = remaining
            selected = np.concatenate((selected, fill))
        elif len(selected) > keep_count:
            selected = selected[
                np.argpartition(mean_score[selected], keep_count - 1)[:keep_count]
            ]
        return selected[np.argsort(mean_score[selected], kind="stable")]

    def _legal_mask(self, last_moves: np.ndarray) -> np.ndarray:
        parent_count = len(last_moves)
        move_count = len(self.permutations)
        previous = np.repeat(last_moves, move_count)
        current = np.tile(np.arange(move_count, dtype=np.int16), parent_count)
        legal = (previous < 0) | (current != self.inverse_indices[previous.clip(min=0)])

        # Parallel layers of one axis commute.  Keeping one canonical ordering
        # removes only duplicate paths, not states reachable at this depth.
        has_previous = previous >= 0
        previous_safe = previous.clip(min=0)
        distinct_parallel = (
            has_previous
            & (self.move_axis[previous_safe] == self.move_axis[current])
            & (self.move_layer[previous_safe] != self.move_layer[current])
        )
        legal &= ~distinct_parallel | (
            self.move_order[previous_safe] <= self.move_order[current]
        )
        return legal

    def search(
        self,
        start: np.ndarray,
        *,
        beam_width: int,
        max_total_length: int,
        time_limit: float,
        seed: int,
        score_noise: float,
    ) -> SearchResult:
        started = time.perf_counter()
        start = np.asarray(start, dtype=np.uint8)
        if np.array_equal(start, self.goal):
            return SearchResult((), 0.0, 0, 0, 0, 0, "already_solved")
        states = start.reshape(1, -1)
        last_moves = np.full(1, -1, dtype=np.int16)
        parent_history: list[np.ndarray] = []
        move_history: list[np.ndarray] = []
        best: tuple[int, ...] | None = None
        expanded = 0
        scored = 0
        peak_unique = 0
        depths_completed = 0
        rng = np.random.default_rng(seed)

        for depth in range(1, max_total_length + 1):
            if time.perf_counter() - started >= time_limit:
                stop_reason = "time_limit"
                break
            parent_count = len(states)
            move_count = len(self.permutations)
            neighbors = states[:, self.permutations].reshape(-1, states.shape[1])
            parent_indices = np.repeat(np.arange(parent_count, dtype=np.int32), move_count)
            move_indices = np.tile(np.arange(move_count, dtype=np.int16), parent_count)
            legal = self._legal_mask(last_moves)
            neighbors = neighbors[legal]
            parent_indices = parent_indices[legal]
            move_indices = move_indices[legal]
            expanded += len(neighbors)

            unique = exact_unique_indices(neighbors)
            neighbors = neighbors[unique]
            parent_indices = parent_indices[unique]
            move_indices = move_indices[unique]
            peak_unique = max(peak_unique, len(neighbors))

            # Exact touch-BFS join.  This loop is intentionally CPU-side: it is
            # the final correctness-sensitive join and candidate counts are
            # bounded by 24*beam_width.
            for candidate_index, candidate in enumerate(neighbors):
                suffix = self.touch_table.get(candidate.tobytes())
                if suffix is None:
                    continue
                total_length = depth + len(suffix)
                if total_length > max_total_length:
                    continue
                prefix = trace_selected_state(
                    int(parent_indices[candidate_index]), parent_history, move_history
                )
                candidate_path = prefix + (int(move_indices[candidate_index]),) + suffix
                if len(candidate_path) != total_length:
                    raise RuntimeError("beam path reconstruction length mismatch")
                if best is None or (len(candidate_path), candidate_path) < (len(best), best):
                    best = candidate_path

            depths_completed = depth
            if best is not None and depth + 1 >= len(best):
                stop_reason = "proved_no_shorter_within_beam"
                break
            if depth >= max_total_length:
                stop_reason = "depth_limit"
                break

            values = self.score(neighbors)
            scored += len(neighbors)
            if score_noise:
                values += rng.normal(0.0, score_noise, size=values.shape).astype(np.float32)
            if not np.isfinite(values).all():
                raise RuntimeError("model produced non-finite scores")
            keep_count = min(beam_width, len(neighbors))
            selected = self.select_agent_quotas(values, keep_count)
            states = np.ascontiguousarray(neighbors[selected])
            selected_parents = np.ascontiguousarray(parent_indices[selected])
            selected_moves = np.ascontiguousarray(move_indices[selected])
            parent_history.append(selected_parents)
            move_history.append(selected_moves)
            last_moves = selected_moves
        else:
            stop_reason = "depth_limit"

        runtime = time.perf_counter() - started
        if best is not None and not np.array_equal(
            replay_indices(start, best, self.permutations), self.goal
        ):
            raise RuntimeError("internal solution failed exact indexed replay")
        return SearchResult(
            best,
            runtime,
            depths_completed,
            expanded,
            scored,
            peak_unique,
            stop_reason,
        )


def incumbent_trajectory(
    start: np.ndarray,
    moves: Sequence[int],
    permutations: np.ndarray,
) -> np.ndarray:
    rows = [np.asarray(start, dtype=np.uint8)]
    for move in moves:
        rows.append(rows[-1][permutations[int(move)]])
    return np.stack(rows)


def rank_rotation_views(
    searcher: ScalarBeamSearcher,
    views: Sequence[RotationView],
    start: np.ndarray,
    incumbent_moves: Sequence[int],
) -> list[dict[str, object]]:
    """Rank views by scalar ranks of the exact incumbent next states."""
    moves = np.asarray(incumbent_moves, dtype=np.int16)
    trajectory = incumbent_trajectory(start, moves, searcher.permutations)
    if not np.array_equal(trajectory[-1], searcher.goal):
        raise ValueError("incumbent suffix does not terminate at the scalar target")

    neighbor_blocks: list[np.ndarray] = []
    rotated_move_blocks: list[np.ndarray] = []
    for view in views:
        viewed = view.transform_states(trajectory[:-1])
        neighbors = viewed[:, searcher.permutations].reshape(
            -1, searcher.permutations.shape[1]
        )
        neighbor_blocks.append(neighbors)
        rotated_move_blocks.append(view.move_to_rotated[moves])
    all_neighbors = np.ascontiguousarray(np.concatenate(neighbor_blocks))
    mean_values = searcher.score(all_neighbors).mean(axis=1)

    rows: list[dict[str, object]] = []
    block_size = len(moves) * len(searcher.permutations)
    for view_index, (view, rotated_moves) in enumerate(
        zip(views, rotated_move_blocks, strict=True)
    ):
        first = view_index * block_size
        values = mean_values[first : first + block_size].reshape(
            len(moves), len(searcher.permutations)
        )
        chosen = values[np.arange(len(moves)), rotated_moves]
        ranks = 1 + np.sum(values < chosen[:, None], axis=1)
        rows.append(
            {
                "view": view.name,
                "view_index": view.index,
                "identity": view.name == "I",
                "metric_mean_log2_rank": float(np.mean(np.log2(ranks))),
                "mean_rank": float(np.mean(ranks)),
                "top8": int(np.sum(ranks <= 8)),
                "steps": len(moves),
            }
        )
    rows.sort(
        key=lambda row: (
            float(row["metric_mean_log2_rank"]),
            float(row["mean_rank"]),
            -int(row["top8"]),
            int(row["view_index"]),
        )
    )
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    return rows


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.backends.mps.is_available():
            requested = "mps"
        elif torch.cuda.is_available():
            requested = "cuda"
        else:
            requested = "cpu"
    device = torch.device(requested)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def parse_int_list(value: str) -> list[int]:
    result = [int(item) for item in value.split(",") if item.strip()]
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("expected a comma-separated list of positive integers")
    return result


def parse_nonnegative_int_list(value: str) -> list[int]:
    try:
        result = [int(item) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not result or any(item < 0 for item in result):
        raise argparse.ArgumentTypeError("expected comma-separated nonnegative integers")
    return result


def cancel_adjacent_inverses(
    moves: Iterable[int], inverse_indices: np.ndarray
) -> tuple[int, ...]:
    stack: list[int] = []
    for raw_move in moves:
        move = int(raw_move)
        if stack and int(inverse_indices[move]) == stack[-1]:
            stack.pop()
        else:
            stack.append(move)
    return tuple(stack)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        action="append",
        required=True,
        help="repeat for equal-quota multi-agent beam selection",
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--puzzle-info", type=Path, default=Path("data/puzzle_info.json"))
    parser.add_argument("--test", type=Path, default=Path("data/test.csv"))
    parser.add_argument(
        "--assigned-states",
        type=Path,
        help="optional agent dataset tensor; restrict automatic ID selection to exact matching rows",
    )
    parser.add_argument("--ids", help="comma-separated explicit test IDs")
    parser.add_argument("--min-length", type=int, default=46)
    parser.add_argument("--top", type=int, default=1)
    parser.add_argument("--beams", type=parse_int_list, default=[512, 2048])
    parser.add_argument(
        "--prefix-cuts",
        type=parse_nonnegative_int_list,
        default=[0],
        help="incumbent prefix lengths to retain before optimizing the suffix",
    )
    parser.add_argument(
        "--variants",
        choices=["original", "rotations", "both"],
        default="both",
        help="search the original frame, verified non-identity physical rotations, or both",
    )
    parser.add_argument(
        "--top-rotations",
        type=int,
        default=3,
        help="number of scalar-ranked non-identity rotations per suffix anchor",
    )
    parser.add_argument(
        "--rotation-views",
        default="",
        help="comma-separated verified view names; overrides --top-rotations",
    )
    parser.add_argument(
        "--ranking-only",
        action="store_true",
        help="rank verified rotation views without launching beam searches",
    )
    parser.add_argument("--touch-radius", type=int, default=3)
    parser.add_argument("--inference-batch", type=int, default=8192)
    parser.add_argument("--time-limit", type=float, default=90.0, help="seconds per beam attempt")
    parser.add_argument("--score-noise", type=float, default=0.0)
    parser.add_argument("--min-gain", type=int, default=1)
    parser.add_argument("--seed", type=int, default=24)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if (
        args.top <= 0
        or args.min_length < 1
        or args.touch_radius < 0
        or args.min_gain < 1
        or args.top_rotations <= 0
    ):
        parser.error("--top and --min-length must be positive; --touch-radius must be nonnegative")
    if args.inference_batch <= 0 or not math.isfinite(args.time_limit) or args.time_limit <= 0:
        parser.error("--inference-batch and --time-limit must be positive")

    puzzle = Puzzle.load(args.puzzle_info)
    tests = load_tests(args.test)
    baseline = load_submission(args.baseline)
    validation_failures: list[str] = []
    incumbent_lengths: dict[str, int] = {}
    for state_id, initial in tests.items():
        moves = baseline.get(state_id)
        if moves is None:
            validation_failures.append(f"missing baseline id {state_id}")
            continue
        try:
            final = puzzle.replay(initial, moves)
        except ValueError as error:
            validation_failures.append(f"id {state_id}: {error}")
            continue
        if final != puzzle.central_state:
            validation_failures.append(f"id {state_id}: baseline does not solve")
        incumbent_lengths[state_id] = len(moves)
    if validation_failures:
        raise ValueError("invalid baseline: " + "; ".join(validation_failures[:10]))

    assigned_ids: list[str] | None = None
    if args.assigned_states is not None:
        assigned_tensor = torch.load(
            args.assigned_states, map_location="cpu", weights_only=True
        )
        if not isinstance(assigned_tensor, torch.Tensor) or assigned_tensor.ndim != 2:
            raise ValueError("--assigned-states must contain a rank-2 tensor")
        assigned_array = assigned_tensor.to(torch.uint8).numpy()
        test_by_state = {
            np.asarray(state, dtype=np.uint8).tobytes(): state_id
            for state_id, state in tests.items()
        }
        assigned_ids = []
        for row_number, state in enumerate(assigned_array):
            state_id = test_by_state.get(state.tobytes())
            if state_id is None:
                raise ValueError(
                    f"assigned state row {row_number} does not match an exact test state"
                )
            assigned_ids.append(state_id)
        if len(set(assigned_ids)) != len(assigned_ids):
            raise ValueError("--assigned-states contains duplicate test states")

    if args.ids:
        selected_ids = [str(int(value)) for value in args.ids.split(",") if value.strip()]
        unknown = [value for value in selected_ids if value not in tests]
        if unknown:
            raise ValueError(f"unknown test IDs: {unknown}")
    else:
        eligible_ids = set(tests) if assigned_ids is None else set(assigned_ids)
        selected_ids = [
            state_id
            for state_id, length in sorted(
                incumbent_lengths.items(), key=lambda item: (-item[1], int(item[0]))
            )
            if state_id in eligible_ids and length >= args.min_length
        ][: args.top]
    if not selected_ids:
        raise ValueError("no test rows selected")

    device = choose_device(args.device)
    models: list[ScalarPilgrim] = []
    contracts: list[ModelContract] = []
    for checkpoint_path in args.checkpoint:
        model, contract = load_scalar_model(checkpoint_path, device)
        models.append(model)
        contracts.append(contract)
    model_contract = contracts[0]
    if any(contract != model_contract for contract in contracts[1:]):
        raise ValueError(f"ensemble checkpoint contracts differ: {contracts}")
    if model_contract.state_size != len(puzzle.central_state):
        raise ValueError(
            f"checkpoint state_size={model_contract.state_size} but puzzle has {len(puzzle.central_state)}"
        )
    labels = set(puzzle.central_state)
    if labels != set(range(model_contract.num_classes)):
        raise ValueError(f"central-state labels {sorted(labels)} do not match checkpoint classes")
    move_names = tuple(puzzle.generators)
    permutations = np.asarray([puzzle.generators[name] for name in move_names], dtype=np.int16)
    goal = np.asarray(puzzle.central_state, dtype=np.uint8)
    searcher = ScalarBeamSearcher(
        models,
        device,
        permutations,
        move_names,
        goal,
        inference_batch=args.inference_batch,
        touch_radius=args.touch_radius,
    )
    rotation_views = build_rotation_views(puzzle, move_names)
    nonidentity_views = tuple(view for view in rotation_views if view.name != "I")
    requested_view_names = tuple(
        item.strip() for item in args.rotation_views.split(",") if item.strip()
    )
    if len(requested_view_names) != len(set(requested_view_names)):
        raise ValueError("--rotation-views contains duplicate names")
    view_by_name = {view.name: view for view in nonidentity_views}
    missing_views = [name for name in requested_view_names if name not in view_by_name]
    if missing_views:
        raise ValueError(f"unknown non-identity rotation views: {missing_views}")
    include_original = args.variants in {"original", "both"}
    include_rotations = args.variants in {"rotations", "both"}
    reports: list[AttemptReport] = []
    improvements: dict[str, tuple[str, ...]] = {}
    rotation_rankings: dict[str, list[dict[str, object]]] = {}

    print(
        json.dumps(
            {
                "device": str(device),
                "checkpoints": [str(path) for path in args.checkpoint],
                "agent_count": len(models),
                "model_contract": asdict(model_contract),
                "touch_states": len(searcher.touch_table),
                "selected_ids": selected_ids,
                "assigned_ids": assigned_ids,
                "beams": args.beams,
                "prefix_cuts": args.prefix_cuts,
                "variants": args.variants,
                "proper_rotation_views": len(rotation_views),
                "top_rotations": args.top_rotations,
                "requested_rotation_views": requested_view_names,
                "ranking_only": args.ranking_only,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    for state_id in selected_ids:
        initial = np.asarray(tests[state_id], dtype=np.uint8)
        incumbent_tokens = baseline[state_id]
        best_tokens = incumbent_tokens
        for beam_width in args.beams:
            for prefix_length in args.prefix_cuts:
                if prefix_length >= len(incumbent_tokens):
                    continue
                prefix_tokens = incumbent_tokens[:prefix_length]
                incumbent_suffix = incumbent_tokens[prefix_length:]
                prefix_state = np.asarray(
                    puzzle.replay(tests[state_id], prefix_tokens), dtype=np.uint8
                )
                # A retained prefix cannot improve the current best if there is
                # no positive suffix budget left.
                suffix_budget = len(best_tokens) - prefix_length - args.min_gain
                if suffix_budget < 0:
                    continue
                prefix_indices = tuple(move_names.index(token) for token in prefix_tokens)
                incumbent_suffix_indices = tuple(
                    move_names.index(token) for token in incumbent_suffix
                )
                attempt_views: list[tuple[str, RotationView | None]] = []
                if include_original:
                    attempt_views.append(("original", None))
                if include_rotations:
                    ranking_key = f"{state_id}:{prefix_length}"
                    ranking = rotation_rankings.get(ranking_key)
                    if ranking is None:
                        ranking = rank_rotation_views(
                            searcher,
                            rotation_views,
                            prefix_state,
                            incumbent_suffix_indices,
                        )
                        rotation_rankings[ranking_key] = ranking
                    if requested_view_names:
                        chosen_views = [view_by_name[name] for name in requested_view_names]
                    else:
                        chosen_views = [
                            view_by_name[str(row["view"])]
                            for row in ranking
                            if not bool(row["identity"])
                        ][: args.top_rotations]
                    attempt_views.extend(
                        (f"rotation:{view.name}", view) for view in chosen_views
                    )

                if args.ranking_only:
                    continue

                for variant, rotation_view in attempt_views:
                    search_start = (
                        prefix_state
                        if rotation_view is None
                        else rotation_view.transform_state(prefix_state)
                    )
                    result = searcher.search(
                        search_start,
                        beam_width=beam_width,
                        max_total_length=suffix_budget,
                        time_limit=args.time_limit,
                        seed=(
                            args.seed
                            + int(state_id) * 1009
                            + beam_width
                            + prefix_length * 65_537
                            + (
                                0
                                if rotation_view is None
                                else (rotation_view.index + 1) * 1_000_003
                            )
                        ),
                        score_noise=args.score_noise,
                    )
                    candidate_tokens: tuple[str, ...] | None = None
                    if result.solution is not None:
                        if rotation_view is None:
                            suffix_indices = result.solution
                        else:
                            rotated_solution = np.asarray(
                                result.solution, dtype=np.int16
                            )
                            if not np.array_equal(
                                replay_indices(
                                    search_start, result.solution, permutations
                                ),
                                goal,
                            ):
                                raise RuntimeError(
                                    f"rotation-space solution failed exact replay: {variant}"
                                )
                            suffix_indices = tuple(
                                int(value)
                                for value in rotation_view.move_from_rotated[
                                    rotated_solution
                                ]
                            )
                        full_indices = cancel_adjacent_inverses(
                            prefix_indices + suffix_indices, searcher.inverse_indices
                        )
                        candidate_tokens = tuple(
                            move_names[index] for index in full_indices
                        )
                        if puzzle.replay(tests[state_id], candidate_tokens) != puzzle.central_state:
                            raise RuntimeError(
                                f"candidate id {state_id} {variant} prefix {prefix_length} "
                                "failed exact competition replay"
                            )
                        if len(candidate_tokens) < len(best_tokens):
                            best_tokens = candidate_tokens
                            improvements[state_id] = candidate_tokens
                    report = AttemptReport(
                        puzzle_id=int(state_id),
                        incumbent_length=len(incumbent_tokens),
                        prefix_length=prefix_length,
                        incumbent_suffix_length=len(incumbent_suffix),
                        variant=variant,
                        beam_width=beam_width,
                        solution_length=None if candidate_tokens is None else len(candidate_tokens),
                        improved_by=(
                            0
                            if candidate_tokens is None
                            else len(incumbent_tokens) - len(candidate_tokens)
                        ),
                        runtime_seconds=round(result.runtime_seconds, 6),
                        depths_completed=result.depths_completed,
                        expanded_candidates=result.expanded_candidates,
                        scored_candidates=result.scored_candidates,
                        peak_unique_candidates=result.peak_unique_candidates,
                        stop_reason=result.stop_reason,
                    )
                    reports.append(report)
                    print(json.dumps(asdict(report), sort_keys=True), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("initial_state_id", "path"))
        for state_id in sorted(improvements, key=int):
            writer.writerow((state_id, ".".join(improvements[state_id])))
    report_path = args.report or args.output.with_suffix(".report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_payload = {
        "checkpoints": [str(path) for path in args.checkpoint],
        "agent_count": len(models),
        "baseline": str(args.baseline),
        "device": str(device),
        "model_contract": asdict(model_contract),
        "touch_radius": args.touch_radius,
        "touch_states": len(searcher.touch_table),
        "variants": args.variants,
        "proper_rotation_views": len(rotation_views),
        "top_rotations": args.top_rotations,
        "requested_rotation_views": list(requested_view_names),
        "rotation_rankings": rotation_rankings,
        "ranking_only": args.ranking_only,
        "selected_ids": [int(value) for value in selected_ids],
        "assigned_ids": None if assigned_ids is None else [int(value) for value in assigned_ids],
        "improvements": {
            state_id: {
                "old_length": incumbent_lengths[state_id],
                "new_length": len(tokens),
                "saved": incumbent_lengths[state_id] - len(tokens),
                "path": ".".join(tokens),
            }
            for state_id, tokens in sorted(improvements.items(), key=lambda item: int(item[0]))
        },
        "attempts": [asdict(report) for report in reports],
        "total_runtime_seconds": round(sum(report.runtime_seconds for report in reports), 6),
    }
    report_path.write_text(json.dumps(report_payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "report": str(report_path),
                "improved_rows": len(improvements),
                "saved_moves": sum(
                    incumbent_lengths[state_id] - len(tokens)
                    for state_id, tokens in improvements.items()
                ),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
