#!/usr/bin/env python3
"""Small read-only Kaggle helper used to recover this account's submissions.

Authentication is read from KAGGLE_API_TOKEN by the official Kaggle client.
This module intentionally contains no upload or submission functionality.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.competitions.types.competition_api_service import (
    ApiDownloadSubmissionRequest,
)


def download_submission(submission_id: int, output: Path) -> None:
    """Download a submission that the authenticated account may access."""
    api = KaggleApi()
    api.authenticate()
    request = ApiDownloadSubmissionRequest()
    request.submission_id = submission_id
    with api.build_kaggle_client() as client:
        redirect = client.competitions.competition_api_client.download_submission(
            request
        )
    response = requests.get(redirect.url, timeout=60)
    response.raise_for_status()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(response.content)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission_id", type=int)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    download_submission(args.submission_id, args.output)
    print(f"downloaded submission {args.submission_id} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
