# Modified/distributed for the CayleyPy 4x4x4 reproducibility bundle (2026).
# Derived from AnanasClassic/cayleypy-training-core under Apache-2.0.

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .config import resolve_path


@dataclass(frozen=True)
class Problem:
    name: str
    target: torch.Tensor
    actions: torch.Tensor
    move_names: list[str]
    inverse_actions: torch.Tensor
    face_ids: torch.Tensor
    num_classes: int
    metadata: dict[str, Any]

    @property
    def state_size(self) -> int:
        return int(self.target.numel())

    @property
    def num_actions(self) -> int:
        return int(self.actions.size(0))


def _parse_generator(data: dict[str, Any]) -> tuple[list[list[int]], list[str]]:
    actions = data.get("actions") or data.get("generators") or data.get("moves")
    names = data.get("names") or data.get("move_names")
    if not isinstance(actions, list) or not isinstance(names, list) or len(actions) != len(names):
        raise ValueError("malformed generator file")
    return actions, [str(name) for name in names]


def _inverse_indices(actions: torch.Tensor) -> torch.Tensor:
    actions = actions.cpu().to(torch.int64)
    identity = tuple(range(actions.size(1)))
    lookup = {tuple(row.tolist()): idx for idx, row in enumerate(actions)}
    result = []
    for action in actions:
        inverse = torch.empty_like(action)
        inverse[action] = torch.arange(action.numel())
        index = lookup.get(tuple(inverse.tolist()))
        if index is None:
            raise ValueError("generator is not closed under inverses")
        result.append(index)
    if any(tuple(actions[i].index_select(0, actions[j]).tolist()) != identity for i, j in enumerate(result)):
        raise ValueError("invalid inverse map")
    return torch.tensor(result, dtype=torch.int64)


def _face_name(name: str) -> str:
    value = str(name).replace("'", "").replace("-", "")
    value = re.sub(r"(?:_[^_]*)$", "", value)
    value = re.sub(r"\d+$", "", value)
    return value


def _face_ids(names: list[str]) -> torch.Tensor:
    ids: dict[str, int] = {}
    return torch.tensor([ids.setdefault(_face_name(name), len(ids)) for name in names], dtype=torch.int64)


def load_problem(spec: dict[str, Any], config_path: Path, device: torch.device) -> Problem:
    kind = spec["kind"]
    if kind == "generator_target":
        generator_path = resolve_path(spec["generator_file"], config_path)
        target_path = resolve_path(spec["target_file"], config_path)
        generator = json.loads(generator_path.read_text(encoding="utf-8"))
        action_rows, names = _parse_generator(generator)
        if target_path.suffix == ".json":
            payload = json.loads(target_path.read_text(encoding="utf-8"))
        else:
            payload = torch.load(target_path, map_location="cpu", weights_only=False)
        if isinstance(payload, dict):
            payload = payload.get("target", payload.get("state", payload))
        target = torch.as_tensor(payload, dtype=torch.int64).flatten()
        actions = torch.tensor(action_rows, dtype=torch.int64)
        inverses = _inverse_indices(actions)
        faces = _face_ids(names)
        metadata = {"generator_file": str(generator_path), "target_file": str(target_path)}
        num_classes = int(spec.get("num_classes", torch.unique(target).numel()))
    else:
        manifest_path = resolve_path(spec["manifest_file"], config_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        target = torch.tensor(manifest["target"], dtype=torch.int64)
        actions = torch.tensor(manifest["actions"], dtype=torch.int64)
        names = [str(name) for name in manifest["names"]]
        inverses = torch.tensor(manifest["inverse_actions"], dtype=torch.int64)
        faces = torch.tensor(manifest["face_ids"], dtype=torch.int64)
        num_classes = int(manifest["num_classes"])
        metadata = {"manifest_file": str(manifest_path), "manifest": manifest}
    if actions.ndim != 2 or target.ndim != 1 or actions.size(1) != target.numel():
        raise ValueError("action and target shapes disagree")
    return Problem(
        name=str(spec.get("name", "problem")), target=target.to(device), actions=actions.to(device),
        move_names=names, inverse_actions=inverses.to(device), face_ids=faces.to(device),
        num_classes=num_classes, metadata=metadata,
    )
