# Budget Core Runner

`scripts/run_budget_core.py` runs the reduced experiment profile and reuses existing successful results.

Default budget:

- NIAH: 300 samples
- LongBench: 300 samples
- RULER: 240 samples
- PAC-Test: 600 samples
- Models: all 9 framework models

This gives 12,960 model-sample targets before reuse. The script checks existing raw results first, copies successful or terminal-skip rows into a clean budget output directory, and only calls the API for missing/retryable samples.

## Dry Run

```bat
python scripts\run_budget_core.py --dry-run
```

Current dry-run result on this workspace:

```text
selected samples: niah 300, longbench 300, ruler 240, pac 600
model-sample targets: 12960
reused existing: 6665
pending API calls: 6295
```

## Run

Set SiliconFlow key first.

CMD:

```bat
set "SILICONFLOW_API_KEY=your_key"
```

Multiple keys:

```bat
set "SILICONFLOW_API_KEYS=key1,key2,key3"
```

Recommended conservative run:

```bat
python scripts\run_budget_core.py --max-workers 2 --max-in-flight 2
```

If rate limits are low:

```bat
python scripts\run_budget_core.py --max-workers 1 --max-in-flight 1
```

If multiple independent keys are available:

```bat
python scripts\run_budget_core.py --max-workers 4 --max-in-flight 4
```

## Output

Clean budget raw results:

```text
results/raw/budget_core/budget_core_main
```

Selected sample manifest:

```text
data/processed/budget_core/budget_core_main/budget_manifest.jsonl
```

Plan summary:

```text
results/logs/budget_core_plan.json
```

The script is resumable. Rerunning the same command with the same `--run-id` only fills missing/retryable samples.

