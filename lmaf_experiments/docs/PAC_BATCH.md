# PAC-Test Automatic Batch Runs

Script:

```text
scripts/run_pac_batch.py
```

It runs the standalone PAC-Test-Dataset v3 files through the same SiliconFlow/local/custom OpenAI-compatible interface used by NIAH, LongBench, and RULER.

## Expected Source Data

Use a directory containing:

```text
subset_A.jsonl
subset_B.jsonl
subset_C.jsonl
subset_D.jsonl
PAC-Test_complete.jsonl
```

The script reads `subset_A/B/C/D.jsonl` by default. You can also pass `PAC-Test_complete.jsonl`; inner `run_pac.py` filters by subset.

External PAC-Test-Dataset rows are adapted automatically:

- `context + question` becomes `prompt`
- `subset=A/B/C/D` becomes `subtask=A_position/B_interference/C_overlap/D_multihop`
- `total_length` becomes `length_tokens_target`
- `position_ratio` becomes `position_percent`
- `noise_density` becomes percentage `density`
- `dilution_type` becomes `interference_type`
- `num_hops` and `distance_level` become PAC-D grouping fields

## API

For SiliconFlow:

```cmd
set "SILICONFLOW_API_KEY=sk-your-key"
```

If your PAC data path contains non-ASCII characters, set it once and then use `%PAC_TEST_DATA_DIR%` in commands:

```cmd
set "PAC_TEST_DATA_DIR=D:\path\to\PAC-Test-Dataset\data"
```

## Recommended Commands

Preview commands without calling the API:

```cmd
python scripts\run_pac_batch.py --source-data "%PAC_TEST_DATA_DIR%" --profile minimal --dry-run
```

Smoke run, one sample per subset per model:

```cmd
python scripts\run_pac_batch.py --source-data "%PAC_TEST_DATA_DIR%" --profile minimal --sample-limit 1 --run-id pac_smoke
```

Run all PAC A/B/C/D experiments for all retained Framework V2 models:

```cmd
python scripts\run_pac_batch.py --source-data "%PAC_TEST_DATA_DIR%" --profile all_framework --run-id pac_main
```

If interrupted, rerun the exact same command with the same `--run-id`; inner calls use `--resume`.

Run only selected subsets:

```cmd
python scripts\run_pac_batch.py --source-data "%PAC_TEST_DATA_DIR%" --subsets A,B --profile all_framework --run-id pac_ab_main
```

## Outputs

Raw results:

```text
results/raw/pac_batch/{run_id}/A/{model_alias}.jsonl
results/raw/pac_batch/{run_id}/B/{model_alias}.jsonl
results/raw/pac_batch/{run_id}/C/{model_alias}.jsonl
results/raw/pac_batch/{run_id}/D/{model_alias}.jsonl
```

Aggregate CSV:

```text
results/aggregate/pac_batch/{run_id}/pac.csv
```

Figures:

```text
results/figures/pac_batch/{run_id}/pac_A_position_curve.png
results/figures/pac_batch/{run_id}/pac_B_density_curve.png
results/figures/pac_batch/{run_id}/pac_C_confusion_matrix.png
results/figures/pac_batch/{run_id}/pac_D_multihop_decay.png
```

Metadata:

```text
results/raw/pac_batch/{run_id}/pac_batch_metadata.json
```

## Notes

- Models with a smaller context window automatically write `skipped_overlength` for samples longer than their configured `max_model_len`.
- `skipped_overlength` rows are treated as completed for resume and are filtered out by default during aggregation.
- Removed models are not part of default profiles and are filtered out by default in aggregate/plot scripts.
