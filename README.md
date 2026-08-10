# CayleyPy 4×4×4 — verified score 46,296

Reproducible portfolio solution for the Kaggle competition
[CayleyPy 444 Cube Solve Optimally](https://www.kaggle.com/competitions/cayley-py-444-cube).

The final submission solves all **1,043** instances and has an exactly replayed
score of **46,296** (44.387344 moves per instance). It improved the supplied
46,378 portfolio by **82 moves across 39 rows**.

```text
puzzles=1043 score=46296 valid=True
SHA-256: 8a10867da63d388de0b0131be38badddd9b4c0fa7edfe0906f188a6b9cbd21f7
```

The result is a bounded, reproducible improvement of known paths; it is not a
claim of a globally optimal 4×4×4 solver.

## Documentation

- [Complete Russian write-up](README_RU.md)
- [Exact score chronology](provenance/score_chronology.csv)
- [All 39 final row replacements](provenance/final_replacements.csv)
- [Machine-readable provenance summary](provenance/provenance_summary.json)
- [Pinned external versions](VENDOR_VERSIONS.md)
- [Third-party notices](THIRD_PARTY_NOTICES.md)

## Quick validation

```bash
python3 src/cube444.py outputs/cube4_submission_46296.csv \
  --puzzle-info data/puzzle_info.json \
  --test data/test.csv
```

Expected output:

```text
puzzles=1043 score=46296 valid=True
```

Regression test:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  python3 -m unittest discover -s tests -v
```

## Deterministic rebuild

The rebuild replays the ten recorded merge milestones and requires byte-for-byte
identity with the checked-in final CSV.

```bash
./REBUILD_FINAL.sh /tmp/cayley444-rebuild
```

## What is included

- `src/` — validator, safe portfolio merger, trajectory graph search, exact
  state splicing, bounded coloured MITM tooling, `twsearch` adapters and model
  search infrastructure;
- `inputs/` — the 46,378 starting submission and the additional user portfolio;
- `work/` — exact intermediate candidates, reports and milestone submissions;
- `external/` — pinned third-party/source snapshots required for provenance;
- `outputs/` — the final 46,296 submission;
- `tests/` — merger regression test.

The final paths were selected only after full 96-sticker replay. Neural models
were used for ranking/search diversification, never as a correctness oracle.
No Kaggle submission or external result publisher is invoked by the rebuild.

The exact final rebuild is self-contained. Optional neural-search experiments
reference public checkpoints/NPZ assets that are intentionally not bundled;
their origins and pinned versions are documented in the Russian write-up and
`VENDOR_VERSIONS.md`.

## License

Original code and documentation are available under the MIT License. Vendored
code and third-party artifacts retain their original licenses; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and the license files within
their directories.
