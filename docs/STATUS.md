# Status

## 2026-07-21 - PAC-Test dataset README

- Phase completed: created a root-level README documenting PAC-Test v5 and PAC-D v2.1 data layout, construction variables, metric definitions, and version boundaries.
- Commands run: inspected `PAC/README.md`, `PAC/manifest.json`, dataset JSONL condition counts, and the PAC v5/v2.1 generation and queue scripts.
- Files changed: `README.md` and `docs/STATUS.md`.
- Verification: confirmed 291 v5 main samples (75/72/72/72 across PAC-A/B/C/D) and 12 PAC-D v2.1 hard samples.
- Blockers: none.
- Next step: use the root README as the dataset-level entry point; retain `PAC/README.md` as the runner-oriented reference.

## 2026-07-21 - English public benchmark data README

- Phase completed: documented the NIAH, RULER, and LongBench data-generation and preparation methods in English.
- Commands run: inspected the official NIAH/RULER documentation, LongBench/RULER batch documentation, and the existing processed-data configuration.
- Files changed: `docs/PUBLIC_BENCHMARK_DATA_README.md` and this status log.
- Verification: checked that the README distinguishes generator-based NIAH/RULER data from static LongBench test records and separates official RULER rows from legacy fallback rows.
- Blockers: none.
- Next step: cite this README when describing public benchmark provenance in the thesis.

## 2026-07-21 - Repository-level English README

- Phase completed: rewrote the root `README.md` as an English overview of the complete long-context attribution project, rather than a PAC-only dataset guide.
- Commands run: inspected root structure, LMAF experiment/runtime configuration, result-summary documentation, and public-data documentation.
- Files changed: `README.md` and this status log.
- Verification: confirmed that the README contains no Chinese characters and links to existing PAC, LMAF, public-data, and status documentation paths.
- Blockers: none.
- Next step: add a repository-wide license and formal citation metadata before public release.
