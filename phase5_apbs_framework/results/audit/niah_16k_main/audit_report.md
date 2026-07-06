# NIAH Dataset Audit

dataset: `data\niah_16k_main.jsonl`
rows: 150
issues: 0

## Cell Summary

| length | position | n | words_p50 | decoys_p50 | answer_count_bad | pos_error_p50 | pos_error_max |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16384 | 10 | 50 | 12167 | 42 | 0 | 0.96 | 1.09 |
| 16384 | 50 | 50 | 12167 | 42 | 0 | 0.14 | 0.27 |
| 16384 | 90 | 50 | 12167 | 42 | 0 | 0.76 | 0.90 |

## Issues

No structural issues found.

## Pass Criteria

- `answer_count_bad` should be 0.
- median decoys should be comfortably above 30 for the 16K hard set.
- median target position error should be small, preferably under 3 percentage points.
- position 50 should not be trivially easier than 10/90 by structure.