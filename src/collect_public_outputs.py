#!/usr/bin/env python3
"""Download small, static outputs from public competition notebooks.

The collector is deliberately read-only with respect to Kaggle and never executes
downloaded code.  It is used to build a legal, reproducible solution portfolio
from artifacts that Kaggle exposes publicly to every participant.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path, PurePosixPath
import re

import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import (
    ApiListKernelSessionOutputRequest,
)


DEFAULT_PATTERN = re.compile(r"(?i)\.(?:csv|zip|json|txt)$")


def safe_relative_path(name: str) -> Path:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe output path {name!r}")
    return Path(*path.parts)


def parse_time(value: object) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    text = str(value).rstrip("Z")
    try:
        return datetime.fromisoformat(text).replace(tzinfo=None)
    except ValueError:
        return None


def collect(
    competition: str,
    output_dir: Path,
    after: datetime | None,
    max_bytes: int,
) -> list[dict[str, object]]:
    api = KaggleApi()
    api.authenticate()
    kernels = api.kernels_list(
        page_size=100,
        competition=competition,
        sort_by="dateRun",
        output_type="all",
    ) or []

    manifest: list[dict[str, object]] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    with api.build_kaggle_client() as client:
        for kernel in kernels:
            if kernel is None:
                continue
            ref = str(kernel.ref)
            last_run = parse_time(getattr(kernel, "last_run_time", None))
            if after is not None and last_run is not None and last_run < after:
                continue
            owner, slug = ref.split("/", 1)
            request = ApiListKernelSessionOutputRequest()
            request.user_name = owner
            request.kernel_slug = slug
            request.page_size = 100
            try:
                response = client.kernels.kernels_api_client.list_kernel_session_output(
                    request
                )
            except Exception as error:  # Keep collecting if one public run vanished.
                manifest.append({"ref": ref, "error": f"{type(error).__name__}: {error}"})
                continue

            for item in response.files or []:
                name = str(item.file_name)
                size = int(getattr(item, "file_size", 0) or 0)
                entry: dict[str, object] = {
                    "ref": ref,
                    "last_run_time": last_run.isoformat() if last_run else None,
                    "name": name,
                    "size": size,
                }
                if not DEFAULT_PATTERN.search(name) or size > max_bytes:
                    entry["downloaded"] = False
                    manifest.append(entry)
                    continue
                destination = output_dir / f"{owner}__{slug}" / safe_relative_path(name)
                destination.parent.mkdir(parents=True, exist_ok=True)
                response_file = requests.get(item.url, timeout=120)
                response_file.raise_for_status()
                destination.write_bytes(response_file.content)
                entry["downloaded"] = True
                entry["path"] = str(destination)
                entry["downloaded_size"] = len(response_file.content)
                manifest.append(entry)

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--competition", default="cayley-py-444-cube")
    parser.add_argument("--output-dir", type=Path, default=Path("work/public_outputs"))
    parser.add_argument("--after", type=datetime.fromisoformat)
    parser.add_argument("--max-mib", type=int, default=20)
    args = parser.parse_args()
    manifest = collect(
        args.competition,
        args.output_dir,
        args.after,
        args.max_mib * 1024 * 1024,
    )
    downloaded = [entry for entry in manifest if entry.get("downloaded")]
    errors = [entry for entry in manifest if "error" in entry]
    print(
        f"kernels={len({entry['ref'] for entry in manifest})} "
        f"files={len(manifest)} downloaded={len(downloaded)} errors={len(errors)}"
    )


if __name__ == "__main__":
    main()
