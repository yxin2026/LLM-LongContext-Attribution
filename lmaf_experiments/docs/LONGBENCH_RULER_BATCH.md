# LongBench + RULER Automatic Batch Runs

Script:

```text
scripts/run_longbench_ruler_batch.py
```

It prepares LongBench, generates the RULER fallback suite, runs the selected model profile through SiliconFlow or another OpenAI-compatible provider, then writes aggregate CSVs and plots.

The script runs experiments sequentially by default. This is intentional: it is easier to resume and less likely to hit API rate limits.

## API

LongBench and RULER use the same model API interface as NIAH. For SiliconFlow, set:

```cmd
set "SILICONFLOW_API_KEY=sk-your-key"
```

You do not need separate LongBench/RULER API keys.

## Suites

`--suite smoke`

- LongBench: `narrativeqa`, `hotpotqa`, `gov_report`, 2 samples per task
- RULER: `niah`, `variable_tracking`, 4K, 1 sample per cell
- Recommended before spending time or API credits on full runs.

`--suite framework_v2`

- LongBench: `narrativeqa`, `qasper`, `multifieldqa_en`, `hotpotqa`, `2wikimqa`, `musique`, `gov_report`, `qmsum`, `multi_news`
- LongBench default sample cap: 200 rows per task
- RULER: `niah`, `variable_tracking`, `common_words_extraction`, `freq_words_extraction`, `qa_squad`, `qa_hotpotqa`
- RULER lengths: 4K / 16K / 32K
- RULER default samples per task-length cell: 50

Use `--longbench-full` if you want every available LongBench test row instead of the 200-row cap.

## Model Profiles

Removed from default profiles:

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

All retained Framework V2 models, including Qwen, Hunyuan, Seed-OSS, and Qwen3-14B thinking/no-thinking.

## Recommended Commands

Preview without calling the API:

```cmd
python scripts\run_longbench_ruler_batch.py --suite smoke --profile minimal --dry-run
```

Run smoke:

```cmd
python scripts\run_longbench_ruler_batch.py --suite smoke --profile minimal --run-id longbench_ruler_smoke
```

Run Framework V2 LongBench + RULER:

```cmd
python scripts\run_longbench_ruler_batch.py --suite framework_v2 --profile all_framework --run-id longbench_ruler_main
```

If interrupted, rerun the exact same command with the same `--run-id`. Inner experiment scripts use `--resume` and skip completed samples.

Only prepare/generate data:

```cmd
python scripts\run_longbench_ruler_batch.py --suite framework_v2 --profile all_framework --generate-only
```

Only run models against already prepared/generated data:

```cmd
python scripts\run_longbench_ruler_batch.py --suite framework_v2 --profile all_framework --run-id longbench_ruler_main --run-only
```

Run only one experiment:

```cmd
python scripts\run_longbench_ruler_batch.py --experiments longbench --suite framework_v2 --profile all_framework --run-id longbench_main
python scripts\run_longbench_ruler_batch.py --experiments ruler --suite framework_v2 --profile all_framework --run-id ruler_main
```

## Outputs

Prepared/generated samples:

```text
data/processed/longbench_ruler_batch/{suite}/longbench/
data/processed/longbench_ruler_batch/{suite}/ruler/
```

Raw results:

```text
results/raw/longbench_ruler_batch/{suite}/{run_id}/longbench/{model_alias}.jsonl
results/raw/longbench_ruler_batch/{suite}/{run_id}/ruler/{model_alias}.jsonl
```

Aggregate CSVs:

```text
results/aggregate/longbench_ruler_batch/{suite}/{run_id}/longbench.csv
results/aggregate/longbench_ruler_batch/{suite}/{run_id}/ruler.csv
```

Figures:

```text
results/figures/longbench_ruler_batch/{suite}/{run_id}/longbench_score_bar.png
results/figures/longbench_ruler_batch/{suite}/{run_id}/ruler_effective_context.png
```

Metadata:

```text
results/raw/longbench_ruler_batch/{suite}/{run_id}/longbench_ruler_batch_metadata.json
```

## Notes

- LongBench preparation reads local `external/LongBench/data/*.jsonl` when present; otherwise it downloads and reads `THUDM/LongBench` `data.zip` through `huggingface_hub`. It does not use the deprecated HuggingFace dataset script path, so `datasets 5.x` is OK.
- LongBench may need network access the first time because `data.zip` has to be downloaded.
- RULER here is the project's marked fallback implementation, not the official RULER repository output.
- Prompts that exceed a model's configured `max_model_len` are written as `skipped_overlength` and treated as completed for resume.
- Aggregation and plotting filter previously removed models by default.
