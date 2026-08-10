#!/usr/bin/env python3
"""Generate a twsearch definition exactly matching competition move words.

Every sticker is distinguished.  This intentionally searches for identities of
the full 96-sticker permutation, so every rewrite is valid for every colored
cube state (the conservative notion required for path peephole shortening).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def format_numbers(values: list[int]) -> str:
    return " ".join(str(value) for value in values)


def generate(puzzle_info: Path) -> str:
    raw = json.loads(puzzle_info.read_text())
    generators: dict[str, list[int]] = raw["generators"]
    forward_names = [name for name in generators if not name.startswith("-")]
    width = len(raw["central_state"])
    identity = list(range(1, width + 1))
    zeros = [0] * width
    lines = [
        "Name CayleyPy444FullSticker",
        "",
        f"Set STICKER {width} 1",
        "",
        "Solved",
        "STICKER",
        format_numbers(identity),
        format_numbers(zeros),
        "End",
        "",
    ]
    for name in forward_names:
        permutation = generators[name]
        if len(permutation) != width or sorted(permutation) != list(range(width)):
            raise ValueError(f"invalid permutation for {name}")
        lines.extend(
            [
                f"Move {name}",
                "STICKER",
                format_numbers([index + 1 for index in permutation]),
                format_numbers(zeros),
                "End",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--puzzle-info", type=Path, default=Path("data/puzzle_info.json"))
    parser.add_argument("--output", type=Path, default=Path("work/cayleypy444.tws"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(generate(args.puzzle_info))
    print(args.output)


if __name__ == "__main__":
    main()
