#!/usr/bin/env python3
"""Run the public cube444 Q heads as an exact, bounded local beam solver.

The public NPZ files store JAX-oriented linear weights (input, output).  This
module converts them to PyTorch modules, verifies the action ordering against
the competition generators, and accepts a replacement only after exact replay.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "work" / "models"))

from pilgrim.qsearcher import QSearcher  # noqa: E402
from unified_training.models import PieceTransformer  # noqa: E402

from cube444 import (  # noqa: E402
    Puzzle,
    load_submission,
    load_tests,
    validate_submission,
)


class FoldedPairQMLP(nn.Module):
    """PairQMLP inference with BatchNorm already folded to scale/shift."""

    dtype = torch.float32
    output_dim = 24

    def __init__(self, payload: dict[str, np.ndarray]) -> None:
        super().__init__()
        self.register_buffer("position_offsets", torch.from_numpy(payload["position_offsets"].astype(np.int64)))
        self.input_weight = nn.Parameter(torch.from_numpy(payload["input_w"]), requires_grad=False)
        self.input_bias = nn.Parameter(torch.from_numpy(payload["input_b"]), requires_grad=False)
        self.input_scale = nn.Parameter(torch.from_numpy(payload["input_bn/0"]), requires_grad=False)
        self.input_shift = nn.Parameter(torch.from_numpy(payload["input_bn/1"]), requires_grad=False)
        self.hidden_weight = nn.Parameter(torch.from_numpy(payload["hidden_w"]), requires_grad=False)
        self.hidden_bias = nn.Parameter(torch.from_numpy(payload["hidden_b"]), requires_grad=False)
        self.hidden_scale = nn.Parameter(torch.from_numpy(payload["hidden_bn/0"]), requires_grad=False)
        self.hidden_shift = nn.Parameter(torch.from_numpy(payload["hidden_bn/1"]), requires_grad=False)
        self.output_weight = nn.Parameter(torch.from_numpy(payload["out_w"]), requires_grad=False)
        self.output_bias = nn.Parameter(torch.from_numpy(payload["out_b"]), requires_grad=False)
        self.block_weights = nn.ParameterList()
        for block in range(4):
            for name in ("l1_w", "l1_b", "bn1/0", "bn1/1", "l2_w", "l2_b", "bn2/0", "bn2/1"):
                self.block_weights.append(
                    nn.Parameter(torch.from_numpy(payload[f"blocks/{block}/{name}"]), requires_grad=False)
                )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        token_ids = states.long() + self.position_offsets.unsqueeze(0)
        hidden = F.embedding_bag(token_ids, self.input_weight, mode="sum") + self.input_bias
        hidden = F.silu(hidden * self.input_scale + self.input_shift)
        hidden = hidden @ self.hidden_weight + self.hidden_bias
        hidden = F.silu(hidden * self.hidden_scale + self.hidden_shift)
        for block in range(4):
            first = block * 8
            residual = hidden
            value = hidden @ self.block_weights[first] + self.block_weights[first + 1]
            value = F.silu(value * self.block_weights[first + 2] + self.block_weights[first + 3])
            value = value @ self.block_weights[first + 4] + self.block_weights[first + 5]
            value = value * self.block_weights[first + 6] + self.block_weights[first + 7]
            hidden = F.silu(residual + value)
        return hidden @ self.output_weight + self.output_bias


class QBlend(nn.Module):
    dtype = torch.float32
    output_dim = 24

    def __init__(self, transformer: nn.Module, mlp: nn.Module, transformer_weight: float = 0.6) -> None:
        super().__init__()
        self.transformer = transformer
        self.mlp = mlp
        self.transformer_weight = float(transformer_weight)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        left = self.transformer(states)
        right = self.mlp(states)
        return self.transformer_weight * left + (1.0 - self.transformer_weight) * right


def _tensor(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(array))


def load_s3(path: Path) -> PieceTransformer:
    with np.load(path) as archive:
        payload = {name: archive[name] for name in archive.files}
    model = PieceTransformer(
        state_size=96,
        num_classes=6,
        output_dim=24,
        layout="cube4",
        d_model=256,
        nhead=8,
        num_layers=4,
        ff_dim=1024,
        dropout_rate=0.0,
        activation="relu",
        pooling="cls",
    )
    state = model.state_dict()
    state["cls_token"] = _tensor(payload["cls_token"])
    state["local_value_embedding.weight"] = _tensor(payload["local_value_embedding"])
    state["piece_projection.weight"] = _tensor(payload["piece_projection_w"].T)
    state["piece_projection.bias"] = _tensor(payload["piece_projection_b"])
    state["piece_position_embedding.weight"] = _tensor(payload["piece_position_embedding"])
    state["piece_type_embedding.weight"] = _tensor(payload["piece_type_embedding"])
    state["input_norm.weight"] = _tensor(payload["input_norm_w"])
    state["input_norm.bias"] = _tensor(payload["input_norm_b"])
    state["output_norm.weight"] = _tensor(payload["output_norm_w"])
    state["output_norm.bias"] = _tensor(payload["output_norm_b"])
    state["output_layer.weight"] = _tensor(payload["output_layer_w"].T)
    state["output_layer.bias"] = _tensor(payload["output_layer_b"])
    for block in range(4):
        source = f"blocks/{block}"
        target = f"blocks.{block}"
        state[f"{target}.norm1.weight"] = _tensor(payload[f"{source}/norm1_w"])
        state[f"{target}.norm1.bias"] = _tensor(payload[f"{source}/norm1_b"])
        state[f"{target}.attn.in_proj_weight"] = torch.cat(
            tuple(_tensor(payload[f"{source}/{name}"].T) for name in ("wq", "wk", "wv")), dim=0
        )
        state[f"{target}.attn.in_proj_bias"] = torch.cat(
            tuple(_tensor(payload[f"{source}/{name}"]) for name in ("bq", "bk", "bv")), dim=0
        )
        state[f"{target}.attn.out_proj.weight"] = _tensor(payload[f"{source}/out_w"].T)
        state[f"{target}.attn.out_proj.bias"] = _tensor(payload[f"{source}/out_b"])
        state[f"{target}.norm2.weight"] = _tensor(payload[f"{source}/norm2_w"])
        state[f"{target}.norm2.bias"] = _tensor(payload[f"{source}/norm2_b"])
        state[f"{target}.ff.0.weight"] = _tensor(payload[f"{source}/ff1_w"].T)
        state[f"{target}.ff.0.bias"] = _tensor(payload[f"{source}/ff1_b"])
        state[f"{target}.ff.3.weight"] = _tensor(payload[f"{source}/ff2_w"].T)
        state[f"{target}.ff.3.bias"] = _tensor(payload[f"{source}/ff2_b"])
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"S3 state mismatch: {result}")
    model.dtype = torch.float32
    model.output_dim = 24
    return model.eval()


def load_mlp(path: Path) -> FoldedPairQMLP:
    with np.load(path) as archive:
        payload = {name: archive[name] for name in archive.files}
    return FoldedPairQMLP(payload).eval()


def load_q_model(kind: str, assets: Path) -> nn.Module:
    if kind == "s3":
        return load_s3(assets / "s3.npz")
    if kind == "mlp":
        return load_mlp(assets / "mlp_x16.npz")
    if kind == "blend":
        return QBlend(load_s3(assets / "s3.npz"), load_mlp(assets / "mlp_x16.npz")).eval()
    raise ValueError(f"unknown model kind {kind!r}")


def inverse_indices(permutations: np.ndarray) -> np.ndarray:
    lookup = {tuple(row.tolist()): index for index, row in enumerate(permutations)}
    result = []
    for row in permutations:
        inverse = np.empty_like(row)
        inverse[row] = np.arange(len(row), dtype=row.dtype)
        result.append(lookup[tuple(inverse.tolist())])
    return np.asarray(result, dtype=np.int64)


def load_and_verify_actions(puzzle: Puzzle, public_spec: Path) -> tuple[list[str], np.ndarray]:
    names = list(puzzle.generators)
    permutations = np.asarray([puzzle.generators[name] for name in names], dtype=np.int64)
    public = json.loads(public_spec.read_text(encoding="utf-8"))
    public_moves = np.asarray(public.get("moves", public.get("actions")), dtype=np.int64)
    public_names = [str(name).replace("'", "-") for name in public.get("move_names", public.get("names"))]
    normalized_competition = [name if not name.startswith("-") else name[1:] + "-" for name in names]
    # Names differ by dash placement in some bundles, so permutations are authoritative.
    if public_moves.shape != permutations.shape or not np.array_equal(public_moves, permutations):
        raise ValueError("public Q action permutations do not match competition action order")
    if len(public_names) != len(normalized_competition):
        raise ValueError("public Q action-name count mismatch")
    return names, permutations


def replay_indices(state: Sequence[int], moves: Iterable[int], permutations: np.ndarray) -> np.ndarray:
    current = np.asarray(state, dtype=np.int64)
    for move in moves:
        current = current[permutations[int(move)]]
    return current


def parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def write_submission(path: Path, order: Sequence[str], paths: dict[str, tuple[str, ...]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["initial_state_id", "path"])
        writer.writeheader()
        for state_id in order:
            writer.writerow({"initial_state_id": state_id, "path": ".".join(paths[state_id])})


@dataclass
class Trial:
    state_id: str
    prefix: int
    beam: int
    limit: int
    found: int | None
    runtime: float
    accepted: bool


def make_searcher(
    model: nn.Module,
    device: torch.device,
    permutations: np.ndarray,
    names: Sequence[str],
    goal: Sequence[int],
    inference_batch: int,
    tail_depth: int,
) -> QSearcher:
    return QSearcher(
        model=model,
        all_moves=torch.as_tensor(permutations, dtype=torch.int64, device=device),
        V0=torch.as_tensor(goal, dtype=torch.int32, device=device),
        device=device,
        verbose=0,
        move_names=list(names),
        inverse_moves=inverse_indices(permutations),
        normalize_path=True,
        batch_size=inference_batch,
        state_dtype=torch.int32,
        tail_bfs_depth=tail_depth,
        move_order=4,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=Path("work/submissions/best_46302.csv"))
    parser.add_argument("--output", type=Path, default=Path("work/qbeam/best.csv"))
    parser.add_argument("--report", type=Path, default=Path("work/qbeam/report.json"))
    parser.add_argument("--puzzle-info", type=Path, default=Path("data/puzzle_info.json"))
    parser.add_argument("--test", type=Path, default=Path("data/test.csv"))
    parser.add_argument("--assets", type=Path, default=Path("work/new_public/cube444_q_beam_tpu_assets"))
    parser.add_argument("--model", choices=("s3", "mlp", "blend"), default="s3")
    parser.add_argument("--device", choices=("mps", "cpu", "cuda"), default="mps")
    parser.add_argument("--ids", type=parse_ints)
    parser.add_argument("--max-ids", type=int)
    parser.add_argument("--min-length", type=int, default=44)
    parser.add_argument("--beams", type=parse_ints, default=[4096])
    parser.add_argument(
        "--remaining",
        type=parse_ints,
        default=[24, 20, 16, 0],
        help="incumbent suffix lengths to re-solve; 0 means the full row",
    )
    parser.add_argument("--required-gain", type=int, default=2)
    parser.add_argument("--inference-batch", type=int, default=4096)
    parser.add_argument("--tail-depth", type=int, default=4)
    parser.add_argument("--stop-score", type=int, default=46298)
    parser.add_argument("--rank-only", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")
    puzzle = Puzzle.load(args.puzzle_info)
    tests = load_tests(args.test)
    current = load_submission(args.baseline)
    validation = validate_submission(puzzle, tests, current)
    if not validation.valid:
        raise ValueError(f"invalid baseline: {validation.failures[:3]}")
    names, permutations = load_and_verify_actions(puzzle, args.assets / "p002.json")
    move_to_index = {name: index for index, name in enumerate(names)}
    model = load_q_model(args.model, args.assets).to(device).eval()
    searcher = make_searcher(
        model, device, permutations, names, puzzle.central_state, args.inference_batch, args.tail_depth
    )

    if args.ids:
        selected_ids = [str(value) for value in args.ids]
    else:
        selected_ids = sorted(
            (state_id for state_id, path in current.items() if len(path) >= args.min_length),
            key=lambda state_id: (-len(current[state_id]), int(state_id)),
        )
    if args.max_ids is not None:
        selected_ids = selected_ids[: args.max_ids]

    if args.rank_only:
        rows = []
        with torch.inference_mode():
            for state_id in selected_ids:
                state = np.asarray(tests[state_id], dtype=np.int64)
                moves = [move_to_index[name] for name in current[state_id]]
                ranks = []
                chosen_values = []
                for move in moves:
                    values = model(torch.as_tensor(state[None], dtype=torch.int32, device=device))[0]
                    order = torch.argsort(values).detach().cpu().numpy()
                    ranks.append(int(np.flatnonzero(order == move)[0]) + 1)
                    chosen_values.append(float(values[move].item()))
                    state = state[permutations[move]]
                rows.append({
                    "id": state_id,
                    "length": len(moves),
                    "mean_rank": float(np.mean(ranks)),
                    "mean_log2_rank": float(np.mean(np.log2(ranks))),
                    "top1": int(np.sum(np.asarray(ranks) == 1)),
                    "top4": int(np.sum(np.asarray(ranks) <= 4)),
                    "ranks": ranks,
                    "chosen_values": chosen_values,
                })
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({"model": args.model, "rows": rows}, indent=2) + "\n")
        print(json.dumps({"ranked": len(rows), "report": str(args.report)}))
        return

    trials: list[Trial] = []
    order = list(tests)
    score = validation.score
    for state_id in selected_ids:
        incumbent = current[state_id]
        incumbent_indices = tuple(move_to_index[name] for name in incumbent)
        trajectory = [np.asarray(tests[state_id], dtype=np.int64)]
        for move in incumbent_indices:
            trajectory.append(trajectory[-1][permutations[move]])
        if not np.array_equal(trajectory[-1], np.asarray(puzzle.central_state)):
            raise RuntimeError(f"baseline trajectory failed for id {state_id}")

        for remaining in args.remaining:
            prefix = 0 if remaining == 0 else max(0, len(incumbent) - remaining)
            limit = len(incumbent) - prefix - args.required_gain
            if limit <= 0:
                continue
            for beam in args.beams:
                started = time.perf_counter()
                solution, _ = searcher.get_solution(
                    torch.as_tensor(trajectory[prefix], dtype=torch.int32),
                    B=beam,
                    num_steps=limit,
                    num_attempts=1,
                )
                runtime = time.perf_counter() - started
                found = None if solution is None else int(solution.numel())
                accepted = False
                if solution is not None:
                    suffix = tuple(int(value) for value in solution.tolist())
                    candidate_indices = incumbent_indices[:prefix] + suffix
                    if len(candidate_indices) <= len(incumbent) - args.required_gain:
                        final = replay_indices(tests[state_id], candidate_indices, permutations)
                        if np.array_equal(final, np.asarray(puzzle.central_state)):
                            candidate = tuple(names[index] for index in candidate_indices)
                            old_length = len(current[state_id])
                            current[state_id] = candidate
                            score -= old_length - len(candidate)
                            accepted = True
                            write_submission(args.output, order, current)
                            check = validate_submission(puzzle, tests, load_submission(args.output))
                            if not check.valid or check.score != score:
                                raise RuntimeError(f"accepted artifact failed replay: {check}")
                            print(
                                json.dumps({
                                    "event": "gain",
                                    "id": state_id,
                                    "old": old_length,
                                    "new": len(candidate),
                                    "score": score,
                                    "output": str(args.output),
                                }),
                                flush=True,
                            )
                trials.append(Trial(state_id, prefix, beam, limit, found, runtime, accepted))
                print(
                    json.dumps({
                        "event": "trial",
                        "id": state_id,
                        "prefix": prefix,
                        "beam": beam,
                        "limit": limit,
                        "found": found,
                        "runtime": round(runtime, 3),
                        "accepted": accepted,
                        "score": score,
                    }),
                    flush=True,
                )
                if accepted or score <= args.stop_score:
                    break
            if accepted or score <= args.stop_score:
                break
        if score <= args.stop_score:
            break

    write_submission(args.output, order, current)
    final = validate_submission(puzzle, tests, load_submission(args.output))
    if not final.valid or final.score != score:
        raise RuntimeError(f"final artifact failed replay: {final}")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(
            {
                "model": args.model,
                "baseline": str(args.baseline),
                "output": str(args.output),
                "score": score,
                "valid": final.valid,
                "trials": [trial.__dict__ for trial in trials],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"event": "complete", "score": score, "valid": final.valid, "output": str(args.output)}))


if __name__ == "__main__":
    main()
