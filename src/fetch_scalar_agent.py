#!/usr/bin/env python3
"""Fetch one raw-deflate checkpoint member from the public CayleyPy archive.

The full Zenodo ZIP is several GiB.  This downloader deliberately issues one
validated HTTP Range GET and never uploads or posts anything.
"""

from __future__ import annotations

import argparse
import binascii
import hashlib
from pathlib import Path
import ssl
import urllib.error
import urllib.request
import zlib


ZENODO_URL = "https://zenodo.org/api/records/14886876/files/weights_and_datasets.zip/content"


def fetch_range(url: str, first: int, last: int, timeout: float) -> tuple[bytes, str]:
    request = urllib.request.Request(
        url,
        headers={
            "Range": f"bytes={first}-{last}",
            "Accept-Encoding": "identity",
            "User-Agent": "cayleypy-local-range-fetch/1",
        },
        method="GET",
    )
    # Some framework Python installations do not inherit macOS Keychain roots.
    # Prefer certifi's Mozilla bundle when available, while retaining normal TLS
    # certificate and hostname verification in all cases.
    try:
        import certifi
    except ImportError:
        context = ssl.create_default_context()
    else:
        context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
        if response.status != 206:
            raise RuntimeError(f"range request must return HTTP 206, got {response.status}")
        expected_content_range = f"bytes {first}-{last}/"
        content_range = response.headers.get("Content-Range", "")
        if not content_range.startswith(expected_content_range):
            raise RuntimeError(
                f"unexpected Content-Range: {content_range!r}; expected prefix {expected_content_range!r}"
            )
        encoding = response.headers.get("Content-Encoding", "identity")
        if encoding not in ("", "identity"):
            raise RuntimeError(f"unexpected HTTP Content-Encoding: {encoding!r}")
        return response.read(), content_range


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--url", default=ZENODO_URL)
    parser.add_argument("--first", type=int, default=4_091_134_676)
    parser.add_argument("--last", type=int, default=4_106_126_543)
    parser.add_argument("--uncompressed-size", type=int, default=16_089_394)
    parser.add_argument("--crc32", type=lambda value: int(value, 0), default=0xAF81FD94)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    if args.first < 0 or args.last < args.first:
        raise ValueError("invalid byte range")
    compressed, content_range = fetch_range(args.url, args.first, args.last, args.timeout)
    expected_compressed_size = args.last - args.first + 1
    if len(compressed) != expected_compressed_size:
        raise RuntimeError(
            f"compressed member size mismatch: expected={expected_compressed_size} got={len(compressed)}"
        )
    try:
        checkpoint = zlib.decompress(compressed, wbits=-15)
    except zlib.error as error:
        raise RuntimeError(f"raw deflate decompression failed: {error}") from error
    if len(checkpoint) != args.uncompressed_size:
        raise RuntimeError(
            f"uncompressed size mismatch: expected={args.uncompressed_size} got={len(checkpoint)}"
        )
    observed_crc = binascii.crc32(checkpoint) & 0xFFFFFFFF
    if observed_crc != args.crc32:
        raise RuntimeError(f"CRC32 mismatch: expected={args.crc32:#010x} got={observed_crc:#010x}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.part")
    temporary.write_bytes(checkpoint)
    temporary.replace(args.output)
    print(
        f"fetched={args.output} content_range={content_range} compressed={len(compressed)} "
        f"uncompressed={len(checkpoint)} crc32={observed_crc:#010x} "
        f"sha256={hashlib.sha256(checkpoint).hexdigest()}"
    )


if __name__ == "__main__":
    try:
        main()
    except urllib.error.HTTPError as error:
        raise SystemExit(f"HTTP error {error.code}: {error.reason}") from error
