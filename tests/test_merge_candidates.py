from __future__ import annotations

import csv
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from merge_candidates import merge


def csv_bytes(headers: tuple[str, ...], rows: tuple[tuple[str, ...], ...]) -> bytes:
    handle = StringIO(newline="")
    writer = csv.writer(handle, lineterminator="\n")
    writer.writerow(headers)
    writer.writerows(rows)
    return handle.getvalue().encode()


class MergeCandidatesTest(unittest.TestCase):
    def test_validates_regular_and_zip_candidates_and_uses_deterministic_tie(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            puzzle_info = directory / "puzzle_info.json"
            puzzle_info.write_text(
                json.dumps(
                    {
                        "name": "tiny",
                        "central_state": [0, 1, 2],
                        "generators": {"s": [1, 0, 2]},
                    }
                )
            )
            tests = directory / "test.csv"
            tests.write_bytes(
                csv_bytes(
                    ("initial_state_id", "initial_state"),
                    (("0", "1,0,2"), ("1", "0,1,2")),
                )
            )
            baseline = directory / "baseline.csv"
            baseline.write_bytes(
                csv_bytes(
                    ("initial_state_id", "path"),
                    (("0", "s.s.s"), ("1", "")),
                )
            )
            candidates = directory / "candidates"
            candidates.mkdir()
            regular = candidates / "a.csv"
            regular.write_bytes(
                csv_bytes(
                    ("puzzle_id", "path"),
                    (("0.0", "s"), ("0", "x"), ("0", "")),
                )
            )
            archive_path = candidates / "z.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr(
                    "nested/solutions.csv",
                    csv_bytes(("initial_state_id", "moves"), (("0", "s"),)),
                )

            output = directory / "merged.csv"
            provenance = directory / "provenance.csv"
            report_path = directory / "report.json"
            rejections = directory / "rejections.csv"
            report = merge(
                baseline_path=baseline,
                input_roots=(candidates,),
                puzzle_info_path=puzzle_info,
                test_path=tests,
                output_path=output,
                provenance_path=provenance,
                report_path=report_path,
                rejections_path=rejections,
                max_member_bytes=1024 * 1024,
                max_zip_depth=3,
            )

            self.assertEqual(report["baseline_score"], 3)
            self.assertEqual(report["merged_score"], 1)
            self.assertEqual(report["gain"], 2)
            self.assertEqual(report["improved_id_count"], 1)
            self.assertEqual(report["ingest"]["zip_files"], 1)
            self.assertEqual(report["ingest"]["invalid_rows"], 2)
            self.assertTrue(report["improved_ids"][0]["source"].endswith("/a.csv"))

            with output.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(
                rows,
                [
                    {"initial_state_id": "0", "path": "s"},
                    {"initial_state_id": "1", "path": ""},
                ],
            )


if __name__ == "__main__":
    unittest.main()
