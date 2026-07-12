# NIAH Automatic Batch Runs

Script:

```text
scripts/run_niah_batch.py
```

It generates the Framework V2.0 NIAH suite and then runs the selected model profile through the configured provider, usually SiliconFlow.

## Suites

`--suite smoke`

- Single-NIAH
- 4K
- 50% position
- 2 samples by default
- Best first check for API, model names, and scoring.

`--suite fast16k`

- Single-NIAH
- 16K
- 10% / 50% / 90%
- 50 samples per position by default
- Fast Lost in the Middle screening.

`--suite framework_v2`

- Single-NIAH: 4K / 16K / 32K / 64K, positions 10% / 50% / 90%
- Multi-NIAH: 16K / 32K, uniform / clustered
- Sequential-NIAH: 16K / 32K, chain distribution

`--suite framework_v2_without_fast16k`

- Single-NIAH: 4K / 32K / 64K, positions 10% / 50% / 90%
- Multi-NIAH: 16K / 32K, uniform / clustered
- Sequential-NIAH: 16K / 32K, chain distribution
- Use this after `--suite fast16k` has already completed, so Single-NIAH 16K is not rerun.

## Model Profiles

Removed from all default profiles and summaries:

- `deepseek_r1_distill_qwen_14b`
- `gemma4_26b_a4b`
- `gemma4_31b`

`--profile minimal`

- `qwen35_9b`
- `qwen3_8b`

`--profile single_card`

- `qwen35_9b`
- `qwen3_8b`
- `qwen35_27b`
- `qwen35_35b_a3b`
- `qwen35_122b_a10b`

`--profile all_framework`

All models currently retained in `run_niah_batch.py`, including Qwen, Hunyuan, Seed-OSS, and Qwen3-14B thinking/no-thinking.

## Recommended Order

Set the SiliconFlow API key:

```cmd
set "SILICONFLOW_API_KEY=sk-your-key"
```

Preview commands without spending API credits:

```cmd
python scripts\run_niah_batch.py --suite smoke --profile minimal --dry-run
```

Run smoke:

```cmd
python scripts\run_niah_batch.py --suite smoke --profile minimal
```

Run 16K screening:

```cmd
python scripts\run_niah_batch.py --suite fast16k --profile all_framework --run-id fast16k_main
```

Run the full Framework V2 NIAH suite:

```cmd
python scripts\run_niah_batch.py --suite framework_v2 --profile all_framework --run-id framework_v2_main
```

If `fast16k` is already done and you want to avoid rerunning Single-NIAH 16K:

```cmd
python scripts\run_niah_batch.py --suite framework_v2_without_fast16k --profile all_framework --run-id framework_v2_extra
```

Use a fixed `--run-id` for long runs. If the process is interrupted, rerun the exact same command with the same `--run-id`; inner `run_niah.py` calls use `--resume` and will skip completed samples.

## Outputs

Generated samples:

```text
data/generated/niah_batch/{suite}/
```

Raw results:

```text
results/raw/niah_batch/{suite}/{run_id}/{model_alias}.jsonl
```

Metadata:

```text
results/raw/niah_batch/{suite}/{run_id}/niah_batch_metadata.json
```

## Aggregation And Plots

The removed models are filtered out by default when aggregating and plotting, even if old raw files still contain them:

```cmd
python scripts\aggregate_results.py --input results\raw\niah_batch\fast16k\RUN_ID --experiment niah --output results\aggregate\niah_fast16k_results.csv
python scripts\plot_results.py --input results\aggregate\niah_fast16k_results.csv --plot niah_position_curve --output results\figures\niah_fast16k_position_curve.png
```

Samples marked `skipped_by_model_length` are also excluded from default aggregate CSVs and plots, so unsupported lengths do not appear as false zero-accuracy points. To inspect skipped rows, pass:

```cmd
--include-skipped
```

If you ever need to inspect old excluded-model rows, pass:

```cmd
--include-excluded-models
```
