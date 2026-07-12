# Official NIAH / RULER Data

This project originally used local synthetic NIAH and local RULER fallback samples. To use official sources instead, keep the model-running code unchanged and import official generated JSONL into this project's standard schema.

## Important Distinction

NIAH and RULER are usually not distributed as one fixed static dataset file.

- NIAH: the commonly cited source is Gregory Kamradt's Needle-in-a-Haystack test framework. It is a generation/evaluation framework, not a single canonical JSONL dataset.
- RULER: the official NVIDIA RULER benchmark provides generation scripts/configs that create JSONL task files at specified context lengths.

So "official dataset" means "official generated samples from the official generator/config", not "download one immutable JSONL file".

## Implemented Official Route

Use the scripts below. NIAH and RULER come from different official sources:

- NIAH: Gregory Kamradt's `needlehaystack` package is used for the haystack context and insertion logic. The project keeps the three required variants: `single`, `multi`, and `sequential`.
- RULER: NVIDIA/RULER's official `scripts/data/synthetic/*.py` generators and `scripts/synthetic.yaml` task configs are used, then imported into this project's schema.

The generated files are still normal project samples with `prompt`, `answer`, `subtask`, and `length_tokens_target`, so the existing inference scripts do not change.

## Generate Official NIAH

Budget-core defaults generate 300 NIAH samples:

- Single-NIAH: 3 lengths x 3 positions x 20 samples = 180
- Multi-NIAH: 2 lengths x 2 distributions x 20 samples = 80
- Sequential-NIAH: 2 lengths x 20 samples = 40

```bat
python scripts\generate_official_niah.py
```

Output:

```text
data\processed\official\niah\samples.jsonl
```

To inspect without running models:

```bat
python scripts\generate_official_niah.py ^
  --single-samples-per-cell 1 ^
  --multi-samples-per-cell 1 ^
  --sequential-samples-per-cell 1 ^
  --output data\processed\official\niah_smoke
```

## Generate Official RULER

Budget-core defaults use official RULER tasks that do not need extra downloaded source assets:

```text
niah_single_1, vt, cwe, fwe
```

That gives 4 tasks x 3 lengths x 20 samples = 240 RULER samples.

```bat
python scripts\generate_official_ruler.py
```

Outputs:

```text
data\official_raw\ruler_budget_core
data\processed\official\ruler\samples.jsonl
```

Dry-run:

```bat
python scripts\generate_official_ruler.py --dry-run
```

More official tasks without external assets:

```bat
python scripts\generate_official_ruler.py --profile no_external_assets
```

Full official task config:

```bat
python scripts\generate_official_ruler.py --profile full_official
```

`full_official` needs extra official source files:

- `external\RULER\scripts\data\synthetic\json\PaulGrahamEssays.json`
- `external\RULER\scripts\data\synthetic\json\squad.json`
- `external\RULER\scripts\data\synthetic\json\hotpotqa.json`

Install the RULER generator extras if needed:

```bat
pip install wonderwords beautifulsoup4 html2text
```

## Import Official RULER

After generating official RULER JSONL files, import them:

```bat
python scripts\import_official_datasets.py ^
  --kind ruler ^
  --source-root D:\path\to\RULER\generated ^
  --output data\processed\official\ruler\samples.jsonl ^
  --subset validation
```

To import only selected official RULER tasks:

```bat
python scripts\import_official_datasets.py ^
  --kind ruler ^
  --source-root D:\path\to\RULER\generated ^
  --output data\processed\official\ruler\samples.jsonl ^
  --tasks vt,cwe,fwe,qa_1,qa_2 ^
  --subset validation
```

Official task name mapping:

| Official task | Project subtask |
|---|---|
| `vt` | `variable_tracking` |
| `cwe` | `common_words_extraction` |
| `fwe` | `freq_words_extraction` |
| `qa_1` | `qa_squad` |
| `qa_2` | `qa_hotpotqa` |
| `niah_*` | `niah` |

## Import Official NIAH

If using official RULER NIAH task families as NIAH:

```bat
python scripts\import_official_datasets.py ^
  --kind niah ^
  --source-root D:\path\to\RULER\generated ^
  --output data\processed\official\niah\samples.jsonl ^
  --tasks niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multikey_2,niah_multikey_3,niah_multivalue,niah_multiquery ^
  --subset validation
```

If using Gregory Kamradt / needlehaystack v2, first export rows that contain a full `prompt` or `input`. If you only have recipe/result rows without full context, reconstruct them first with the official tool before importing.

## Run Budget Core With Official Data

Recommended top-up runner:

```bat
python scripts\run_official_budget_topup.py --dry-run
```

This script uses the agreed per-model targets:

- NIAH: 300
- LongBench: 300
- RULER: 240
- PAC: 600

By default, NIAH and RULER are treated as fresh official-data runs, so old NIAH/RULER results do not count as credits. LongBench and PAC are reused because their datasets did not change: previous successful rows are copied as credits, and only missing/failed rows are called again. Re-running it with the same `--run-id` only fills missing or failed rows.

Run for real:

```bat
python scripts\run_official_budget_topup.py ^
  --run-id official_budget_topup_main ^
  --max-workers 2 --max-in-flight 2
```

Output:

```text
results\raw\official_budget_topup\official_budget_topup_main
```

Manual lower-level route:

Once imported, run:

```bat
python scripts\run_budget_core.py ^
  --niah-data data\processed\official\niah ^
  --ruler-data data\processed\official\ruler ^
  --ruler-tasks niah,variable_tracking,common_words_extraction,freq_words_extraction ^
  --run-id budget_core_official ^
  --dry-run
```

Then run for real:

```bat
python scripts\run_budget_core.py ^
  --niah-data data\processed\official\niah ^
  --ruler-data data\processed\official\ruler ^
  --ruler-tasks niah,variable_tracking,common_words_extraction,freq_words_extraction ^
  --run-id budget_core_official ^
  --max-workers 2 --max-in-flight 2
```

The output will go to:

```text
results/raw/budget_core/budget_core_official
```
