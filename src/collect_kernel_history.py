#!/usr/bin/env python3
"""Read-only collector for immutable Kaggle kernel-version outputs.

The public Kaggle CLI only exposes the latest run, but its official SDK accepts
``versionNumber`` on both the kernel metadata and output-download GET methods.
This collector obtains ``currentVersionNumber`` for each public kernel and then
walks the contiguous version numbers without executing or modifying a kernel.

Only small solution-bearing CSVs are requested.  ``all_solutions.csv`` is the
preferred artifact because it is a superset of the selected solution file.  A
selected-solution file and finally the complete submission are used as fallbacks
for older/failed versions that did not emit the preferred artifact.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import random
import time
from typing import Iterable, Optional

import requests


PREFERRED_ARTIFACTS = (
    "cayleypy_public/output/solutions/all_solutions.csv",
    "cayleypy_public/output/solutions/solutions.csv",
    "cayleypy_public/output/submission.csv",
)

KAGGLE_API_ROOT = "https://www.kaggle.com/api/v1"


@dataclass(frozen=True)
class KernelInfo:
    ref: str
    current_version: int


@dataclass
class ArtifactResult:
    ref: str
    version: int
    artifact: str | None = None
    path: str | None = None
    size: int = 0
    sha256: str | None = None
    candidate_rows: int = 0
    status: str = "missing"
    error: str | None = None


def load_refs(manifest_path: Path) -> list[str]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("collector manifest must contain a JSON list")
    refs = {
        str(item["ref"])
        for item in payload
        if isinstance(item, dict) and item.get("ref")
    }
    for ref in refs:
        if ref.count("/") != 1:
            raise ValueError(f"invalid kernel ref: {ref!r}")
    return sorted(refs)


def get_kernel_info(ref: str) -> KernelInfo:
    owner, slug = ref.split("/", 1)
    response = requests.get(
        f"{KAGGLE_API_ROOT}/kernels/pull",
        params={"userName": owner, "kernelSlug": slug},
        timeout=60,
    )
    response.raise_for_status()
    if response.request.method != "GET":
        raise RuntimeError("refusing non-GET kernel metadata request")
    version = int(response.json()["metadata"]["currentVersionNumber"])
    if version < 1:
        raise ValueError(f"{ref}: invalid current version {version}")
    return KernelInfo(ref=ref, current_version=version)


def candidate_row_count(path: Path) -> int:
    """Count rows that appear to contain an ID and a solution path."""
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or ())
            has_id = bool(fields & {"initial_state_id", "puzzle_id", "id"})
            has_path = bool(
                fields & {"original_oriented_path", "path", "solution_path"}
            )
            if not (has_id and has_path):
                return 0
            count = 0
            for row in reader:
                state_id = (
                    row.get("initial_state_id")
                    or row.get("puzzle_id")
                    or row.get("id")
                    or ""
                ).strip()
                solution = (
                    row.get("original_oriented_path")
                    or row.get("path")
                    or row.get("solution_path")
                    or ""
                ).strip()
                if state_id and solution:
                    count += 1
            return count
    except (csv.Error, OSError, UnicodeError):
        return 0


def safe_artifact_name(artifact: str) -> str:
    name = artifact.rsplit("/", 1)[-1]
    if name not in {"all_solutions.csv", "solutions.csv", "submission.csv"}:
        raise ValueError(f"unexpected artifact path: {artifact!r}")
    return name


def download_artifact(
    ref: str,
    version: int,
    artifact: str,
    destination: Path,
    *,
    max_bytes: int,
    retries: int,
) -> tuple[str, int, str]:
    """Return status, byte count, and digest for one versioned GET."""
    if destination.is_file() and destination.stat().st_size:
        content = destination.read_bytes()
        return "cached", len(content), sha256(content).hexdigest()

    owner, slug = ref.split("/", 1)
    for attempt in range(retries + 1):
        try:
            response = requests.get(
                f"{KAGGLE_API_ROOT}/kernels/output/download/"
                f"{owner}/{slug}/{artifact}",
                params={"versionNumber": version},
                timeout=120,
            )
            response.raise_for_status()
            if any(item.request.method != "GET" for item in (*response.history, response)):
                raise RuntimeError("refusing non-GET kernel output request")
            content = response.content
            if len(content) > max_bytes:
                raise ValueError(
                    f"artifact is {len(content)} bytes, exceeds limit {max_bytes}"
                )
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(f".{destination.name}.tmp")
            temporary.write_bytes(content)
            temporary.replace(destination)
            return "downloaded", len(content), sha256(content).hexdigest()
        except requests.HTTPError as error:
            status = error.response.status_code if error.response is not None else None
            if status == 404:
                return "missing", 0, ""
            if status not in {429, 500, 502, 503, 504} or attempt >= retries:
                raise
            retry_after = error.response.headers.get("Retry-After")
            if status == 429 and retry_after:
                try:
                    delay = max(float(retry_after), 0.0) + 1.0 + random.random()
                except ValueError:
                    delay = 0.0
                if delay:
                    time.sleep(delay)
                    continue
        except (requests.ConnectionError, requests.Timeout):
            if attempt >= retries:
                raise
        time.sleep(min(8.0, 0.5 * (2**attempt)) + random.random() * 0.2)
    raise AssertionError("retry loop fell through")


def collect_version(
    output_dir: Path,
    ref: str,
    version: int,
    *,
    max_bytes: int,
    retries: int,
) -> ArtifactResult:
    owner, slug = ref.split("/", 1)
    version_dir = output_dir / f"{owner}__{slug}" / f"v{version:04d}"
    missing: list[str] = []
    for artifact in PREFERRED_ARTIFACTS:
        destination = version_dir / safe_artifact_name(artifact)
        try:
            status, size, digest = download_artifact(
                ref,
                version,
                artifact,
                destination,
                max_bytes=max_bytes,
                retries=retries,
            )
        except Exception as error:  # Preserve other versions if one vanished.
            return ArtifactResult(
                ref=ref,
                version=version,
                artifact=artifact,
                status="error",
                error=f"{type(error).__name__}: {error}",
            )
        if status == "missing":
            missing.append(artifact)
            continue
        rows = candidate_row_count(destination)
        if rows:
            return ArtifactResult(
                ref=ref,
                version=version,
                artifact=artifact,
                path=str(destination),
                size=size,
                sha256=digest,
                candidate_rows=rows,
                status=status,
            )
        # A header-only or non-solution artifact is kept for audit, but try the
        # next fallback in case the version still emitted a useful submission.
    return ArtifactResult(
        ref=ref,
        version=version,
        status="no_candidates",
        error="no candidate rows; attempted " + ", ".join(PREFERRED_ARTIFACTS),
    )


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def collect(
    refs: Iterable[str],
    output_dir: Path,
    *,
    workers: int,
    max_bytes: int,
    retries: int,
    version_catalog: Optional[Path] = None,
) -> dict[str, object]:
    refs = list(refs)
    infos: list[KernelInfo] = []
    metadata_errors: list[dict[str, str]] = []
    if version_catalog is not None:
        catalog = json.loads(version_catalog.read_text(encoding="utf-8"))
        known = {
            str(item["ref"]): KernelInfo(
                ref=str(item["ref"]),
                current_version=int(item["current_version"]),
            )
            for item in catalog["kernels"]
        }
        missing = sorted(set(refs) - known.keys())
        if missing:
            raise ValueError(
                "version catalog lacks kernel refs: " + ", ".join(missing)
            )
        infos = [known[ref] for ref in refs]
    else:
        for ref in refs:
            try:
                infos.append(get_kernel_info(ref))
            except Exception as error:
                metadata_errors.append(
                    {"ref": ref, "error": f"{type(error).__name__}: {error}"}
                )
    if metadata_errors:
        raise RuntimeError(
            "kernel metadata enumeration incomplete; preserving the existing "
            f"manifest ({len(metadata_errors)} errors)"
        )

    versions = [
        (info.ref, version)
        for info in infos
        for version in range(1, info.current_version + 1)
    ]
    prior: dict[tuple[str, int], ArtifactResult] = {}
    manifest_path = output_dir / "manifest.json"
    if manifest_path.is_file():
        try:
            old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for item in old_manifest.get("versions", []):
                restored = ArtifactResult(**item)
                key = (restored.ref, restored.version)
                if restored.status == "no_candidates":
                    prior[key] = restored
                elif restored.candidate_rows and restored.path:
                    source = Path(restored.path)
                    if source.is_file() and sha256(source.read_bytes()).hexdigest() == (
                        restored.sha256
                    ):
                        prior[key] = restored
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
            prior = {}

    results: list[ArtifactResult] = list(prior.values())
    pending = [item for item in versions if item not in prior]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                collect_version,
                output_dir,
                ref,
                version,
                max_bytes=max_bytes,
                retries=retries,
            ): (ref, version)
            for ref, version in pending
        }
        for future in as_completed(futures):
            ref, version = futures[future]
            try:
                results.append(future.result())
            except Exception as error:
                results.append(
                    ArtifactResult(
                        ref=ref,
                        version=version,
                        status="error",
                        error=f"{type(error).__name__}: {error}",
                    )
                )

    results.sort(key=lambda item: (item.ref, item.version))
    successful = [item for item in results if item.candidate_rows]
    manifest = {
        "read_only_methods": [
            "GET https://www.kaggle.com/api/v1/kernels/pull",
            "GET https://www.kaggle.com/api/v1/kernels/output/download/{owner}/{slug}/{artifact}",
            "GET signed artifact URL",
        ],
        "kernel_count": len(infos),
        "version_count": len(versions),
        "candidate_version_count": len(successful),
        "candidate_rows": sum(item.candidate_rows for item in successful),
        "unique_artifact_payloads": len(
            {item.sha256 for item in successful if item.sha256}
        ),
        "metadata_errors": metadata_errors,
        "kernels": [asdict(info) for info in infos],
        "versions": [asdict(item) for item in results],
    }
    atomic_json(output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--latest-manifest",
        type=Path,
        default=Path("work/public_outputs/manifest.json"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("work/kernel_history_get_outputs")
    )
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--max-mib", type=int, default=20)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--version-catalog",
        type=Path,
        help="reuse a prior manifest's immutable current-version catalog",
    )
    args = parser.parse_args()
    if not 1 <= args.workers <= 32:
        parser.error("--workers must be between 1 and 32")
    refs = load_refs(args.latest_manifest)
    result = collect(
        refs,
        args.output_dir,
        workers=args.workers,
        max_bytes=args.max_mib * 1024 * 1024,
        retries=args.retries,
        version_catalog=args.version_catalog,
    )
    print(
        f"kernels={result['kernel_count']} versions={result['version_count']} "
        f"candidate_versions={result['candidate_version_count']} "
        f"candidate_rows={result['candidate_rows']} "
        f"unique_payloads={result['unique_artifact_payloads']}"
    )


if __name__ == "__main__":
    main()
