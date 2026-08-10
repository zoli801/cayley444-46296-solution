#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
REBUILD_DIR="${1:-rebuild}"

if [[ -e "$REBUILD_DIR" ]]; then
  echo "Refusing to overwrite existing path: $REBUILD_DIR" >&2
  exit 2
fi
mkdir -p "$REBUILD_DIR"

merge_step() {
  local name="$1"
  local baseline="$2"
  shift 2
  "$PYTHON_BIN" src/merge_candidates.py \
    --baseline "$baseline" \
    --output "$REBUILD_DIR/$name.csv" \
    --provenance "$REBUILD_DIR/$name.provenance.csv" \
    --report "$REBUILD_DIR/$name.report.json" \
    --rejections "$REBUILD_DIR/$name.rejections.csv" \
    "$@"
}

merge_step 01_46332 inputs/cube4_submission_46378.csv \
  work/service_investigation/official_cube4_results.csv \
  work/submissions/best_46452_twshort_d9.csv \
  work/submissions/merged_public_twmerge.csv \
  work/public_outputs/dingshijiana__minimal-cayleypy-notebook-for-cube4/cayleypy_public/output/solutions/all_solutions.csv

merge_step 02_46330 "$REBUILD_DIR/01_46332.csv" \
  external/cayley6/outputs/submission_46330_valid.csv

merge_step 03_46328 "$REBUILD_DIR/02_46330.csv" \
  work/candidates_tw_d8_high48.csv

merge_step 04_46322 "$REBUILD_DIR/03_46328.csv" \
  work/submissions/merged_cross_workspace_history_partial_induced.csv

merge_step 05_46318 "$REBUILD_DIR/04_46322.csv" \
  work/candidates_tw_d8_history_partial.csv

merge_step 06_46316 "$REBUILD_DIR/05_46318.csv" \
  work/harvest/slot1_colored8/batch_000.csv

merge_step 07_46314 "$REBUILD_DIR/06_46316.csv" \
  work/candidates_tw_d10_history_full.csv

merge_step 08_46312 "$REBUILD_DIR/07_46314.csv" \
  work/broad_slot0_even_r8/batch_0016.csv

merge_step 09_46302 "$REBUILD_DIR/08_46312.csv" \
  external/cayley6/outputs/submission_46304_valid.csv

merge_step 10_46296 "$REBUILD_DIR/09_46302.csv" \
  inputs/cube444_merged_46372.csv

"$PYTHON_BIN" src/cube444.py "$REBUILD_DIR/10_46296.csv" \
  --puzzle-info data/puzzle_info.json --test data/test.csv

if ! cmp -s "$REBUILD_DIR/10_46296.csv" outputs/cube4_submission_46296.csv; then
  echo "Replay score is correct, but rebuilt bytes differ from the archived final CSV." >&2
  exit 3
fi

echo "Rebuilt artifact is byte-identical to outputs/cube4_submission_46296.csv"

