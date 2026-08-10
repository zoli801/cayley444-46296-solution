#!/usr/bin/env python3
"""Validate and merge local CayleyPy solution portfolios.

The merger treats all candidate artifacts as untrusted.  It recursively reads
CSV files (including CSV members of ZIP files), replays every recognized path,
and keeps the shortest solved path for each test ID.  Ties are deterministic
and prefer the baseline, which avoids changing rows without a score gain.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from io import BytesIO, StringIO
import json
from pathlib import Path
import re
from typing import Iterable, Iterator, Sequence
import zipfile

from cube444 import Puzzle, State, load_tests, split_path


ID_COLUMNS = (
    "initial_state_id",
    "puzzle_id",
    "state_id",
    "id",
)
PATH_COLUMNS = (
    "path",
    "solution_path",
    "move_sequence",
    "moves",
    "solution",
)
INTEGER_LIKE = re.compile(r"[+]?(?:0|[1-9][0-9]*)(?:[.]0+)?\Z")


@dataclass(frozen=True)
class CsvSource:
    name: str
    data: bytes
    baseline: bool = False


@dataclass(frozen=True)
class Candidate:
    state_id: str
    path: str
    moves: tuple[str, ...]
    source: str
    row: int
    baseline: bool

    @property
    def length(self) -> int:
        return len(self.moves)

    @property
    def selection_key(self) -> tuple[object, ...]:
        # Retain an equally short baseline row.  Otherwise choose a stable path
        # and provenance independent of filesystem traversal order.
        return (
            self.length,
            0 if self.baseline else 1,
            self.path,
            self.source,
            self.row,
        )


@dataclass(frozen=True)
class Rejection:
    source: str
    row: int | None
    state_id: str
    path: str
    reason: str


@dataclass
class IngestStats:
    filesystem_files: int = 0
    zip_files: int = 0
    csv_sources: int = 0
    candidate_csv_sources: int = 0
    skipped_csv_sources: int = 0
    rows_seen: int = 0
    candidate_rows: int = 0
    valid_rows: int = 0
    invalid_rows: int = 0
    unique_replays: int = 0
    replay_cache_hits: int = 0
    source_errors: list[str] = field(default_factory=list)


def display_path(path: Path) -> str:
    """Return stable, readable provenance relative to the current directory."""

    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def normalized_header(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().casefold()).strip("_")


def choose_column(headers: Sequence[str], aliases: Sequence[str]) -> int | None:
    positions = {name: index for index, name in enumerate(headers)}
    return next((positions[name] for name in aliases if name in positions), None)


def decode_csv(data: bytes) -> str:
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16")
    return data.decode("utf-8-sig")


def normalize_state_id(raw: str, tests: dict[str, State]) -> str | None:
    value = raw.strip()
    if value in tests:
        return value
    if not INTEGER_LIKE.fullmatch(value):
        return None
    canonical = str(int(value.split(".", 1)[0]))
    return canonical if canonical in tests else None


def iter_zip_csv_sources(
    data: bytes,
    source_name: str,
    stats: IngestStats,
    *,
    max_member_bytes: int,
    max_zip_depth: int,
    depth: int = 1,
) -> Iterator[CsvSource]:
    """Yield CSVs from a ZIP without extracting paths to the filesystem."""

    stats.zip_files += 1
    if depth > max_zip_depth:
        stats.source_errors.append(
            f"{source_name}: nested ZIP depth exceeds {max_zip_depth}"
        )
        return
    try:
        archive = zipfile.ZipFile(BytesIO(data))
    except (OSError, zipfile.BadZipFile) as error:
        stats.source_errors.append(f"{source_name}: {type(error).__name__}: {error}")
        return

    with archive:
        for info in sorted(archive.infolist(), key=lambda item: item.filename):
            if info.is_dir():
                continue
            member_name = info.filename.replace("\\", "/")
            label = f"{source_name}!{member_name}"
            suffix = Path(member_name).suffix.casefold()
            if suffix not in {".csv", ".zip"}:
                continue
            if info.file_size > max_member_bytes:
                stats.source_errors.append(
                    f"{label}: uncompressed size {info.file_size} exceeds "
                    f"{max_member_bytes} bytes"
                )
                continue
            try:
                member_data = archive.read(info)
            except (OSError, RuntimeError, zipfile.BadZipFile) as error:
                stats.source_errors.append(f"{label}: {type(error).__name__}: {error}")
                continue
            if suffix == ".csv":
                yield CsvSource(label, member_data)
            else:
                yield from iter_zip_csv_sources(
                    member_data,
                    label,
                    stats,
                    max_member_bytes=max_member_bytes,
                    max_zip_depth=max_zip_depth,
                    depth=depth + 1,
                )


def iter_csv_sources(
    baseline: Path,
    roots: Sequence[Path],
    stats: IngestStats,
    *,
    max_member_bytes: int,
    max_zip_depth: int,
) -> Iterator[CsvSource]:
    """Walk inputs deterministically, de-duplicating overlapping roots."""

    baseline_resolved = baseline.resolve()
    seen: set[Path] = set()

    def visit(path: Path, *, is_baseline: bool = False) -> Iterator[CsvSource]:
        resolved = path.resolve()
        if resolved in seen:
            return
        seen.add(resolved)
        stats.filesystem_files += 1
        label = display_path(path)
        try:
            size = path.stat().st_size
            if size > max_member_bytes:
                stats.source_errors.append(
                    f"{label}: size {size} exceeds {max_member_bytes} bytes"
                )
                return
            data = path.read_bytes()
        except OSError as error:
            stats.source_errors.append(f"{label}: {type(error).__name__}: {error}")
            return
        suffix = path.suffix.casefold()
        if suffix == ".csv":
            yield CsvSource(label, data, baseline=is_baseline)
        elif suffix == ".zip":
            yield from iter_zip_csv_sources(
                data,
                label,
                stats,
                max_member_bytes=max_member_bytes,
                max_zip_depth=max_zip_depth,
            )

    yield from visit(baseline, is_baseline=True)
    for root in sorted(roots, key=lambda item: display_path(item)):
        if not root.exists():
            stats.source_errors.append(f"{display_path(root)}: input does not exist")
            continue
        paths: Iterable[Path]
        if root.is_file():
            paths = (root,)
        else:
            paths = sorted(
                (
                    path
                    for path in root.rglob("*")
                    if path.is_file() and path.suffix.casefold() in {".csv", ".zip"}
                ),
                key=lambda item: display_path(item),
            )
        for path in paths:
            yield from visit(path, is_baseline=path.resolve() == baseline_resolved)


def validate_candidate(
    puzzle: Puzzle,
    initial: State,
    moves: tuple[str, ...],
) -> str | None:
    try:
        final = puzzle.replay(initial, moves)
    except (IndexError, ValueError) as error:
        return str(error)
    if final != puzzle.central_state:
        mismatch = sum(a != b for a, b in zip(final, puzzle.central_state))
        if len(final) != len(puzzle.central_state):
            mismatch += abs(len(final) - len(puzzle.central_state))
        return f"unsolved ({mismatch} mismatched stickers)"
    return None


def ingest_source(
    source: CsvSource,
    puzzle: Puzzle,
    tests: dict[str, State],
    stats: IngestStats,
    replay_cache: dict[tuple[str, tuple[str, ...]], str | None],
) -> tuple[list[Candidate], list[Rejection]]:
    stats.csv_sources += 1
    candidates: list[Candidate] = []
    rejections: list[Rejection] = []
    try:
        handle = StringIO(decode_csv(source.data), newline="")
        reader = csv.reader(handle)
        raw_headers = next(reader, None)
    except (UnicodeError, csv.Error) as error:
        stats.skipped_csv_sources += 1
        stats.source_errors.append(
            f"{source.name}: cannot parse CSV header: {type(error).__name__}: {error}"
        )
        return candidates, rejections
    if raw_headers is None:
        stats.skipped_csv_sources += 1
        return candidates, rejections

    headers = tuple(normalized_header(value) for value in raw_headers)
    if len(set(headers)) != len(headers):
        stats.skipped_csv_sources += 1
        stats.source_errors.append(f"{source.name}: duplicate normalized CSV columns")
        return candidates, rejections
    id_index = choose_column(headers, ID_COLUMNS)
    path_index = choose_column(headers, PATH_COLUMNS)
    if id_index is None or path_index is None:
        stats.skipped_csv_sources += 1
        return candidates, rejections

    stats.candidate_csv_sources += 1
    needed_index = max(id_index, path_index)
    try:
        for row_number, row in enumerate(reader, start=2):
            stats.rows_seen += 1
            stats.candidate_rows += 1
            raw_id = row[id_index].strip() if id_index < len(row) else ""
            raw_path = row[path_index].strip() if path_index < len(row) else ""
            if len(row) <= needed_index:
                rejections.append(
                    Rejection(source.name, row_number, raw_id, raw_path, "short CSV row")
                )
                continue
            state_id = normalize_state_id(raw_id, tests)
            if state_id is None:
                rejections.append(
                    Rejection(source.name, row_number, raw_id, raw_path, "unknown test ID")
                )
                continue
            try:
                moves = split_path(raw_path)
            except ValueError as error:
                rejections.append(
                    Rejection(source.name, row_number, state_id, raw_path, str(error))
                )
                continue
            canonical_path = ".".join(moves)
            cache_key = (state_id, moves)
            if cache_key in replay_cache:
                stats.replay_cache_hits += 1
                reason = replay_cache[cache_key]
            else:
                stats.unique_replays += 1
                reason = validate_candidate(puzzle, tests[state_id], moves)
                replay_cache[cache_key] = reason
            if reason is not None:
                rejections.append(
                    Rejection(source.name, row_number, state_id, canonical_path, reason)
                )
                continue
            candidates.append(
                Candidate(
                    state_id=state_id,
                    path=canonical_path,
                    moves=moves,
                    source=source.name,
                    row=row_number,
                    baseline=source.baseline,
                )
            )
    except csv.Error as error:
        stats.source_errors.append(
            f"{source.name}: CSV error after row {reader.line_num}: {error}"
        )
    stats.valid_rows += len(candidates)
    stats.invalid_rows += len(rejections)
    return candidates, rejections


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def csv_text(headers: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    handle = StringIO(newline="")
    writer = csv.writer(handle, lineterminator="\n")
    writer.writerow(headers)
    writer.writerows(rows)
    return handle.getvalue()


def merge(
    *,
    baseline_path: Path,
    input_roots: Sequence[Path],
    puzzle_info_path: Path,
    test_path: Path,
    output_path: Path,
    provenance_path: Path,
    report_path: Path,
    rejections_path: Path,
    max_member_bytes: int,
    max_zip_depth: int,
) -> dict[str, object]:
    puzzle = Puzzle.load(puzzle_info_path)
    tests = load_tests(test_path)
    width = len(puzzle.central_state)
    wrong_width = [state_id for state_id, state in tests.items() if len(state) != width]
    if wrong_width:
        raise ValueError(f"test states have wrong width: {','.join(wrong_width)}")

    stats = IngestStats()
    replay_cache: dict[tuple[str, tuple[str, ...]], str | None] = {}
    best: dict[str, Candidate] = {}
    baseline: dict[str, Candidate] = {}
    rejections: list[Rejection] = []
    valid_candidate_count = 0
    for source in iter_csv_sources(
        baseline_path,
        input_roots,
        stats,
        max_member_bytes=max_member_bytes,
        max_zip_depth=max_zip_depth,
    ):
        source_candidates, source_rejections = ingest_source(
            source, puzzle, tests, stats, replay_cache
        )
        valid_candidate_count += len(source_candidates)
        rejections.extend(source_rejections)
        for candidate in source_candidates:
            if candidate.baseline:
                previous_baseline = baseline.get(candidate.state_id)
                if previous_baseline is not None:
                    raise ValueError(
                        f"duplicate baseline ID {candidate.state_id} at "
                        f"{previous_baseline.source}:{previous_baseline.row} and "
                        f"{candidate.source}:{candidate.row}"
                    )
                baseline[candidate.state_id] = candidate
            previous = best.get(candidate.state_id)
            if previous is None or candidate.selection_key < previous.selection_key:
                best[candidate.state_id] = candidate

    missing_baseline = tests.keys() - baseline.keys()
    missing_best = tests.keys() - best.keys()
    if missing_baseline:
        raise ValueError(
            "baseline lacks valid solutions for IDs: "
            + ",".join(sorted(missing_baseline, key=int))
        )
    if missing_best:
        raise ValueError(
            "portfolio lacks valid solutions for IDs: "
            + ",".join(sorted(missing_best, key=int))
        )

    ordered_ids = sorted(tests, key=int)
    baseline_score = sum(baseline[state_id].length for state_id in ordered_ids)
    merged_score = sum(best[state_id].length for state_id in ordered_ids)
    improvements = [
        state_id
        for state_id in ordered_ids
        if best[state_id].length < baseline[state_id].length
    ]

    # Count all distinct solved (ID, path) pairs, not repeated copies in artifacts.
    unique_valid = {
        cache_key for cache_key, reason in replay_cache.items() if reason is None
    }
    source_contributions: dict[str, dict[str, int]] = {}
    for state_id in improvements:
        selected = best[state_id]
        item = source_contributions.setdefault(
            selected.source, {"improved_ids": 0, "gain": 0}
        )
        item["improved_ids"] += 1
        item["gain"] += baseline[state_id].length - selected.length

    atomic_write_text(
        output_path,
        csv_text(
            ("initial_state_id", "path"),
            ((state_id, best[state_id].path) for state_id in ordered_ids),
        ),
    )
    atomic_write_text(
        provenance_path,
        csv_text(
            (
                "initial_state_id",
                "baseline_length",
                "selected_length",
                "gain",
                "selected_source",
                "selected_source_row",
                "selected_path_sha256",
            ),
            (
                (
                    state_id,
                    baseline[state_id].length,
                    best[state_id].length,
                    baseline[state_id].length - best[state_id].length,
                    best[state_id].source,
                    best[state_id].row,
                    sha256(best[state_id].path.encode()).hexdigest(),
                )
                for state_id in ordered_ids
            ),
        ),
    )
    atomic_write_text(
        rejections_path,
        csv_text(
            ("source", "row", "initial_state_id", "path", "reason"),
            (
                (item.source, item.row or "", item.state_id, item.path, item.reason)
                for item in rejections
            ),
        ),
    )

    report: dict[str, object] = {
        "baseline": display_path(baseline_path),
        "baseline_score": baseline_score,
        "merged_score": merged_score,
        "gain": baseline_score - merged_score,
        "puzzle_count": len(tests),
        "improved_id_count": len(improvements),
        "improved_ids": [
            {
                "initial_state_id": state_id,
                "baseline_length": baseline[state_id].length,
                "selected_length": best[state_id].length,
                "gain": baseline[state_id].length - best[state_id].length,
                "source": best[state_id].source,
                "source_row": best[state_id].row,
            }
            for state_id in improvements
        ],
        "source_contributions": dict(sorted(source_contributions.items())),
        "output": display_path(output_path),
        "provenance": display_path(provenance_path),
        "rejections": display_path(rejections_path),
        "unique_valid_paths": len(unique_valid),
        "valid_candidate_rows": valid_candidate_count,
        "ingest": asdict(stats),
    }
    atomic_write_text(report_path, json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        default=[Path("work/public_outputs")],
        help="candidate files or directories (default: work/public_outputs)",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("work/submissions/baseline_46718.csv"),
    )
    parser.add_argument(
        "--puzzle-info", type=Path, default=Path("data/puzzle_info.json")
    )
    parser.add_argument("--test", type=Path, default=Path("data/test.csv"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("work/submissions/merged_public.csv"),
    )
    parser.add_argument(
        "--provenance",
        type=Path,
        default=Path("work/submissions/merged_public_provenance.csv"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("work/submissions/merged_public_report.json"),
    )
    parser.add_argument(
        "--rejections",
        type=Path,
        default=Path("work/submissions/merged_public_rejections.csv"),
    )
    parser.add_argument("--max-member-mib", type=int, default=64)
    parser.add_argument("--max-zip-depth", type=int, default=3)
    args = parser.parse_args()
    if args.max_member_mib <= 0:
        parser.error("--max-member-mib must be positive")
    if args.max_zip_depth <= 0:
        parser.error("--max-zip-depth must be positive")

    report = merge(
        baseline_path=args.baseline,
        input_roots=args.inputs,
        puzzle_info_path=args.puzzle_info,
        test_path=args.test,
        output_path=args.output,
        provenance_path=args.provenance,
        report_path=args.report,
        rejections_path=args.rejections,
        max_member_bytes=args.max_member_mib * 1024 * 1024,
        max_zip_depth=args.max_zip_depth,
    )
    print(
        f"puzzles={report['puzzle_count']} baseline={report['baseline_score']} "
        f"merged={report['merged_score']} gain={report['gain']} "
        f"improved_ids={report['improved_id_count']}"
    )
    ingest = report["ingest"]
    assert isinstance(ingest, dict)
    print(
        f"candidate_rows={ingest['candidate_rows']} valid={ingest['valid_rows']} "
        f"invalid={ingest['invalid_rows']} unique_valid={report['unique_valid_paths']}"
    )


if __name__ == "__main__":
    main()
