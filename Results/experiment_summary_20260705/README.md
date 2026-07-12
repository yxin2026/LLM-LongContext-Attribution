# Existing Experiment Results Summary

This folder was generated from the current `lmaf_experiments/results/raw` files. Raw files were not modified.

## Completion Overview

| Dataset | Complete Models | Incomplete Models | Missing Models | Retryable Errors | Missing Samples Estimate |
| --- | ---: | ---: | ---: | ---: | ---: |
| NIAH framework_v2_without_fast16k | 0 | 9 | 0 | 4361 | 0 |
| LongBench framework_v2 | 0 | 9 | 0 | 2032 | 0 |
| RULER framework_v2 | 0 | 9 | 0 | 7696 | 0 |
| PAC A position | 1 | 8 | 0 | 1325 | 0 |
| PAC B interference | 0 | 9 | 0 | 10403 | 1233 |
| PAC C overlap | 0 | 0 | 9 | 0 | 5400 |
| PAC D multihop | 0 | 0 | 9 | 0 | 10440 |

## Generated Tables

- `tables/completion_status.csv`: per dataset/model completion status.
- `tables/clean_manifest.csv`: mapping from raw JSONL files to cleaned JSONL files.
- `tables/model_overview.csv`: mean scores by model and aggregate.
- `tables/*_framework*.csv` and `tables/pac_*.csv`: cleaned aggregate tables.
- `tables/confirmed_complete_aggregate_rows.csv`: aggregate rows for only confirmed-complete dataset/model pairs.
- `tables/pac_A_position_complete_only.csv`: complete-only PAC A rows for `qwen35_9b`.

## Generated Figures

- `figures/longbench_score_bar.png`
- `figures/niah_position_curve.png`
- `figures/overview_mean_accuracy.png`
- `figures/pac_A_position_curve.png`
- `figures/pac_B_density_curve.png`
- `figures/ruler_effective_context.png`
- `figures/confirmed_complete_pac_A_position_curve.png`

## Confirmed Complete Data

- PAC A position: 1 model(s) complete.

Strict definition: unique final rows >= expected samples and no retryable API errors remain. Terminal overlength skips count as completed bookkeeping but are excluded by default in aggregate tables.
