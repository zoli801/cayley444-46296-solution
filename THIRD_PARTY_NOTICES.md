# Third-party notices

The root MIT license applies only to original project code and documentation.
It does not relicense `external/`, competition data, inputs, outputs, work
artifacts, or material identified below. The following bundled or derived
material retains its upstream license and attribution.

## `external/twsearch/`

Pinned [`cubing/twips`](https://github.com/cubing/twips) (`twsearch`) source
snapshot at commit
`f90bbc843a30a9fc22d7dd3ca3c441c5a77c7270`.

It is distributed under the licenses included as:

- `external/twsearch/LICENSE-MPL.md`
- `external/twsearch/LICENSE-GPL.md`

## `external/models/pilgrim/`

Source-derived modules from
[`khoruzhii/cayleypy-cube`](https://github.com/khoruzhii/cayleypy-cube), pinned
at commit `f02604fa7b665b82e5fbe5692b336a4fe4a01bdc`.
Upstream license: MIT, copyright 2024 Khoruzhii Kirill. A copy is included at
`external/models/LICENSE-MIT-khoruzhii.txt`.

`src/scalar_beam444.py` is also derived from this upstream project. The
additional model files used during the experiment came from the public Kaggle
dataset `trydotatwo/cube4-full-transformer-inference` v1, whose API metadata
declared CC0; that declaration does not supersede the embedded MIT notice.

## `external/models/unified_training/`

Source-derived modules from
[`AnanasClassic/cayleypy-training-core`](https://github.com/AnanasClassic/cayleypy-training-core),
pinned at commit `5a721748535f3b889295f7ccbf6e6e9c0cef0695`.
Upstream license: Apache License 2.0. The required license and notice are
included as:

- `external/models/unified_training/LICENSE-APACHE-2.0.txt`
- `external/models/unified_training/NOTICE`

The vendored package differs from that upstream snapshot by Cube4-specific
additions and changes. Prominent modification notices are included in all five
vendored Python files.

## Public solution artifacts

Candidate paths and manifests under `work/` were collected from public Kaggle
notebook histories, the public CayleyPy submission dataset, and
[`TryDotAtwo/cayleypy-beam-results`](https://github.com/TryDotAtwo/cayleypy-beam-results).
The exact row-level origins used in the final result are recorded in
`provenance/`, `work/submissions/*_provenance.csv`, and the Russian write-up.
Those artifacts are provided for research reproducibility and attribution;
they are not relicensed by the root MIT license. The audited beam-results
repository did not declare a repository-wide license, so users who redistribute
its raw records should verify permission with the upstream owner.

## Competition data

`data/puzzle_info.json` and `data/test.csv` come from the public Kaggle
competition bundle; Kaggle API metadata inspected on 2026-08-10 identified the
data license as CC0 1.0.

## References not vendored

The experiments also reference `TryDotAtwo/MultiGPUBeamSearch`, public Kaggle
notebooks and Zenodo DOI `10.5281/zenodo.14886876`. Their full upstream assets
are not included unless explicitly present in this repository.
