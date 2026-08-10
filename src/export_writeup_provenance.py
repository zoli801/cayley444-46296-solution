#!/usr/bin/env python3
"""Export the transitive, human-readable provenance of the 46,296 portfolio."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from cube444 import Puzzle, load_submission, load_tests, validate_submission


MILESTONES = [
    ("public_and_local_merge", Path("work/submissions/merged_46378_all_report.json")),
    ("cross_workspace_merge", Path("work/submissions/merged_cross_workspace_report.json")),
    ("twsearch_d8_high48", Path("work/submissions/best_46328_report.json")),
    ("kernel_history_and_induced", Path("work/submissions/best_46322_report.json")),
    ("twsearch_d8_history", Path("work/submissions/best_46318_report.json")),
    ("colored_mitm_r8_id575", Path("work/submissions/slot1_colored_live_report.json")),
    ("twsearch_d10_id850", Path("work/submissions/merged_tw_d10_into_46316.json")),
    ("colored_mitm_r8_id686", Path("work/reports/best_46312_broad_even_r8.json")),
    ("sibling_portfolio_merge", Path("work/submissions/merged_sibling46304.json")),
    ("final_user_portfolio_merge", Path("work/user_46372/direct_report.json")),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/cayley444_46296_writeup_assets"))
    parser.add_argument("--final", type=Path, default=Path("outputs/cube4_submission_46296.csv"))
    parser.add_argument("--puzzle-info", type=Path, default=Path("data/puzzle_info.json"))
    parser.add_argument("--test", type=Path, default=Path("data/test.csv"))
    args = parser.parse_args()

    final_paths = load_submission(args.final)
    result = validate_submission(Puzzle.load(args.puzzle_info), load_tests(args.test), final_paths)
    if not result.valid or result.score != 46296:
        raise SystemExit(f"unexpected final artifact: score={result.score} failures={result.failures[:3]}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    chronology_rows = []
    replacement_rows = []
    current_score = 46378
    seen_ids: set[str] = set()
    for order, (label, report_path) in enumerate(MILESTONES, 1):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        before = int(report["baseline_score"])
        after = int(report["merged_score"])
        if before != current_score:
            raise ValueError(f"non-contiguous milestone {label}: expected {current_score}, got {before}")
        output_path = Path(report["output"])
        selected = load_submission(output_path)
        improvements = report["improved_ids"]
        chronology_rows.append({
            "step": order,
            "label": label,
            "score_before": before,
            "score_after": after,
            "gain": before - after,
            "improved_ids": ",".join(str(row["initial_state_id"]) for row in improvements),
            "report": str(report_path),
            "output": str(output_path),
        })
        for row in improvements:
            state_id = str(row["initial_state_id"])
            if state_id in seen_ids:
                raise ValueError(f"ID {state_id} was replaced in more than one milestone")
            seen_ids.add(state_id)
            path = selected[state_id]
            path_text = ".".join(path)
            replacement_rows.append({
                "step": order,
                "label": label,
                "initial_state_id": state_id,
                "score_before": before,
                "score_after": after,
                "old_length": int(row["baseline_length"]),
                "new_length": int(row["selected_length"]),
                "gain": int(row["gain"]),
                "source": str(row["source"]),
                "source_row": row.get("source_row", ""),
                "final_path_sha256": hashlib.sha256(path_text.encode()).hexdigest(),
                "final_path": path_text,
            })
        current_score = after

    if current_score != 46296 or len(seen_ids) != 39:
        raise ValueError(f"unexpected provenance closure: score={current_score}, ids={len(seen_ids)}")
    if any(".".join(final_paths[row["initial_state_id"]]) != row["final_path"] for row in replacement_rows):
        raise ValueError("a milestone replacement does not match the final artifact")

    with (args.output_dir / "score_chronology.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(chronology_rows[0]))
        writer.writeheader()
        writer.writerows(chronology_rows)
    with (args.output_dir / "final_replacements.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(replacement_rows[0]))
        writer.writeheader()
        writer.writerows(replacement_rows)

    source_counts = Counter(row["source"] for row in replacement_rows)
    summary = {
        "final": str(args.final),
        "score": result.score,
        "puzzles": result.puzzle_count,
        "valid": result.valid,
        "final_sha256": hashlib.sha256(args.final.read_bytes()).hexdigest(),
        "starting_score": 46378,
        "total_gain": 82,
        "replaced_ids": len(seen_ids),
        "rows_retained_from_starting_submission": result.puzzle_count - len(seen_ids),
        "source_row_counts": dict(sorted(source_counts.items())),
        "milestones": chronology_rows,
    }
    (args.output_dir / "provenance_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
