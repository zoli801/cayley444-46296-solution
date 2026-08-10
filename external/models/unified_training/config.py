# Modified for the CayleyPy 4x4x4 reproducibility bundle (2026).
# Derived from AnanasClassic/cayleypy-training-core under Apache-2.0.
# This copy contains bundle-specific configuration/path handling changes.

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parent


def resolve_path(value: str | Path, config_path: Path | None = None) -> Path:
    text = str(value)
    path = Path(text).expanduser()
    if path.is_absolute():
        return path
    base = config_path.parent if config_path is not None else PACKAGE_ROOT
    return (base / path).resolve()


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    config_path = Path(path).resolve()
    data = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(data)
    return data, config_path


def validate_config(data: dict[str, Any]) -> None:
    if data.get("schema_version") != 1:
        raise ValueError("config schema_version must be 1")
    for key in ("name", "problem", "sampler", "augmentation", "model", "objective", "training", "artifacts"):
        if key not in data:
            raise ValueError(f"config is missing {key!r}")
    if data["problem"].get("kind") not in {"generator_target", "problem_manifest"}:
        raise ValueError("unsupported problem.kind")
    if data["sampler"].get("kind") not in {"rw_middle_sparse", "rw_goal_set_sparse"}:
        raise ValueError("unsupported sampler.kind")
    if data["augmentation"].get("kind") not in {"identity", "move_symmetry"}:
        raise ValueError("unsupported augmentation.kind")
    if data["model"].get("provider") not in {"piece_transformer", "pair_qmlp"}:
        raise ValueError("unsupported model.provider")


def apply_overrides(data: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    result = copy.deepcopy(data)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override must be dotted.path=value, got {item!r}")
        dotted, raw = item.split("=", 1)
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        cursor: dict[str, Any] = result
        parts = dotted.split(".")
        for part in parts[:-1]:
            child = cursor.get(part)
            if not isinstance(child, dict):
                child = {}
                cursor[part] = child
            cursor = child
        cursor[parts[-1]] = value
    validate_config(result)
    return result
