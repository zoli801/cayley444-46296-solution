#!/usr/bin/env python3
"""Normalize versioned CayleyPy solution artifacts into one candidate CSV."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path


@dataclass(frozen=True, order=True)
class Candidate:
    initial_state_id: str
    path: str
    kernel_ref: str
    kernel_version: int
    artifact: str
    artifact_row: int
    variant: str
    source_sha256: str


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def extract(manifest_path: Path) -> tuple[list[Candidate], dict[str, object]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidates: list[Candidate] = []
    errors: list[str] = []
    source_files = 0
    for item in manifest["versions"]:
        if not item.get("candidate_rows") or not item.get("path"):
            continue
        source = Path(item["path"])
        source_files += 1
        digest = sha256(source.read_bytes()).hexdigest()
        if item.get("sha256") and digest != item["sha256"]:
            errors.append(f"{source}: digest does not match manifest")
            continue
        try:
            with source.open(newline="", encoding="utf-8-sig") as handle:
                reader = csv.DictReader(handle)
                for row_number, row in enumerate(reader, 2):
                    state_id = (
                        row.get("initial_state_id")
                        or row.get("puzzle_id")
                        or row.get("id")
                        or ""
                    ).strip()
                    # Reflected runs must be mapped back to the original puzzle
                    # orientation before they can be replayed against test.csv.
                    path = (
                        row.get("original_oriented_path")
                        or row.get("path")
                        or row.get("solution_path")
                        or ""
                    ).strip()
                    if not state_id or not path:
                        continue
                    candidates.append(
                        Candidate(
                            initial_state_id=state_id,
                            path=path,
                            kernel_ref=str(item["ref"]),
                            kernel_version=int(item["version"]),
                            artifact=str(item["artifact"]),
                            artifact_row=row_number,
                            variant=(row.get("variant") or "").strip(),
                            source_sha256=digest,
                        )
                    )
        except (csv.Error, OSError, UnicodeError) as error:
            errors.append(f"{source}: {type(error).__name__}: {error}")

    # The first deterministic provenance tuple owns duplicate (ID, path) pairs.
    unique: dict[tuple[str, str], Candidate] = {}
    for candidate in sorted(candidates):
        unique.setdefault((candidate.initial_state_id, candidate.path), candidate)
    ordered = sorted(unique.values(), key=lambda item: (int(item.initial_state_id), item))
    summary = {
        "manifest": str(manifest_path),
        "source_files": source_files,
        "raw_candidate_rows": len(candidates),
        "unique_id_path_pairs": len(ordered),
        "unique_path_strings": len({item.path for item in ordered}),
        "kernel_refs": len({item.kernel_ref for item in ordered}),
        "kernel_versions": len(
            {(item.kernel_ref, item.kernel_version) for item in ordered}
        ),
        "errors": errors,
    }
    return ordered, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("work/kernel_history_get_outputs/manifest.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("work/kernel_history_get_candidates.csv"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("work/kernel_history_get_candidates_report.json"),
    )
    args = parser.parse_args()
    candidates, summary = extract(args.manifest)
    buffer = StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(
        (
            "initial_state_id",
            "path",
            "kernel_ref",
            "kernel_version",
            "artifact",
            "artifact_row",
            "variant",
            "source_sha256",
        )
    )
    for item in candidates:
        writer.writerow(
            (
                item.initial_state_id,
                item.path,
                item.kernel_ref,
                item.kernel_version,
                item.artifact,
                item.artifact_row,
                item.variant,
                item.source_sha256,
            )
        )
    atomic_text(args.output, buffer.getvalue())
    atomic_text(args.report, json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(
        f"sources={summary['source_files']} raw={summary['raw_candidate_rows']} "
        f"unique={summary['unique_id_path_pairs']} refs={summary['kernel_refs']} "
        f"versions={summary['kernel_versions']} errors={len(summary['errors'])}"
    )


if __name__ == "__main__":
    main()
