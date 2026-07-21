# Public Benchmark Data Preparation

This document describes how the NIAH, RULER, and LongBench data used by this project were obtained, generated, prepared, and selected. It is intended to support reproducibility and accurate reporting in the thesis or paper.

The authoritative implementation is under `lmaf_experiments/`. This archive keeps the resulting datasets, model outputs, tables, and figures together with the PAC-Test materials.

## 1. Final Data Snapshot

| Benchmark | Final data location in the experiment project | Source and preparation route | Current size |
|---|---|---|---:|
| NIAH | `data/processed/official/niah/samples.jsonl` | Gregory Kamradt Needle-in-a-Haystack context and insertion generator, with project-defined single, multi, and sequential variants | 300 |
| RULER | `data/processed/official/ruler/samples.jsonl` | Official NVIDIA/RULER synthetic generators and task configuration, then imported into the project JSONL schema | 240 |
| LongBench | `data/processed/longbench_ruler_batch/framework_v2/longbench/*.jsonl` | Official THUDM/LongBench test JSONL files, read and wrapped into a common prompt schema | 1,750 prepared rows; 300 selected for the budget-core run |

NIAH and RULER are generator-based benchmarks. Therefore, “official data” in this project means samples generated from the official generator and configuration at a specified context length, rather than one immutable downloaded JSONL file. LongBench is different: it uses an official static test collection and is not synthetically regenerated.

## 2. Shared Preparation Principles

All prepared samples are stored as JSONL records with a consistent interface for downstream inference and scoring. Common fields include:

- `prompt`: complete model input;
- `answer` or `answers`: reference answer(s);
- `sample_id`: reproducible sample identifier;
- `subtask` or `task`: benchmark task label;
- `length_tokens_target` and `length_tokens_actual`: target and observed prompt length;
- source metadata, random seed, and error placeholder fields.

The preparation stage does not call model APIs. Model predictions, latency, token accounting, timestamps, and request errors are appended later as separate raw-result JSONL records.

## 3. NIAH Data Generation

### Source

NIAH uses the context construction and needle insertion logic from Gregory Kamradt's Needle-in-a-Haystack framework:

- Repository: <https://github.com/gkamradt/LLMTest_NeedleInAHaystack>
- Package interface used by this project: `needlehaystack`
- Haystack source: the package's Paul Graham Essays text collection

The project generator is `lmaf_experiments/scripts/generate_official_niah.py`. It uses the official `LLMNeedleHaystackTester` and `LLMMultiNeedleHaystackTester` machinery to read, trim, and insert evidence into the long context. The model object used at generation time is offline-only; no LLM is queried to create the samples.

### Variants

| Variant | Context and insertion route | Project-specific part | Interpretation boundary |
|---|---|---|---|
| Single-NIAH | Official haystack, length control, and single-needle insertion | A deterministic entity-code fact and retrieval question | Standard single-fact retrieval setting. |
| Multi-NIAH | Official multi-needle generator for the uniform setting; official insertion engine for the clustered setting | Three entity-key facts and either uniform or clustered placement | The clustered placement rule is project-defined, although its context construction uses the official engine. |
| Sequential-NIAH | Official haystack and insertion engine | A two-hop forwarding chain ending in a verification code | The chain semantics are project-defined; this is not a fixed official static NIAH task. |

The appropriate paper wording is:

> NIAH samples were constructed using the official Needle-in-a-Haystack haystack and insertion logic, with single-needle, multi-needle, and sequential retrieval variants implemented within the same generation framework.

This wording is more accurate than claiming that every variant is a fixed dataset directly published by the original repository.

### Final NIAH Grid

The official-route NIAH collection contains 300 samples with fixed seed `42`.

| Subset | Conditions | Samples per condition | Total |
|---|---|---:|---:|
| Single-NIAH | 4K, 32K, 64K x positions 10%, 50%, 90% | 20 | 180 |
| Multi-NIAH | 16K, 32K x uniform, clustered | 20 | 80 |
| Sequential-NIAH | 16K, 32K with a two-hop chain | 20 | 40 |
| Total |  |  | 300 |

The generated JSONL records store the actual insertion location in `position_percent_actual`. Analyses should use this stored value when precise placement matters.

### Reproduction

From the configured experiment environment:

```cmd
cd lmaf_experiments
python scripts\generate_official_niah.py
```

This produces `data/processed/official/niah/samples.jsonl` and does not invoke a model API.

## 4. RULER Data Generation

### Source

RULER uses the official NVIDIA/RULER synthetic benchmark repository:

- Repository: <https://github.com/NVIDIA/RULER>
- Configuration: `external/RULER/scripts/synthetic.yaml`
- Generator family: `external/RULER/scripts/data/synthetic/*.py`

The project wrapper `lmaf_experiments/scripts/generate_official_ruler.py` runs the official generator with the selected task configuration, context length, sample count, seed, and answer prefix. `lmaf_experiments/scripts/import_official_datasets.py` then converts the generated JSONL into the project schema while preserving the official task name, source file, target length, prompt, output reference, and answer prefix.

### Final RULER Profile

The final official-route collection uses the `budget_core` profile. It selects official tasks that do not require extra external essay or QA assets.

| Official task name | Project task name | Core official configuration |
|---|---|---|
| `niah_single_1` | `niah` | Noise haystack; one key, one value, one query |
| `vt` | `variable_tracking` | One variable-binding chain with four hops |
| `cwe` | `common_words_extraction` | Frequent/common word extraction with the official count settings |
| `fwe` | `freq_words_extraction` | Frequency-based word extraction with `alpha=2.0` |

The final grid uses 4K, 16K, and 32K contexts, 20 samples per task-length cell, and seed `42`:

```text
4 tasks x 3 lengths x 20 samples = 240 samples
```

Generated official files are retained under `data/official_raw/ruler_budget_core/`; the standardized model-ready data are written to `data/processed/official/ruler/samples.jsonl`.

### Reproduction

```cmd
cd lmaf_experiments
python scripts\generate_official_ruler.py --profile budget_core
```

The full RULER profile may additionally require the official Paul Graham Essays, SQuAD, and HotpotQA assets. Those tasks are not part of the 240-sample budget-core collection.

### Legacy Fallback Boundary

The repository also contains `scripts/run_ruler.py` and `src/lmaf/data/ruler_adapter.py`. These create simplified project fallback tasks and tag them with `implementation=ruler_fallback`. They are useful for smoke tests and early debugging, but they are not official NVIDIA/RULER generator outputs.

Formal reporting rule:

- rows with `source_schema=official_ruler` may be reported as official-generator RULER samples;
- rows with `implementation=ruler_fallback` must be labelled as project fallback data and must not be pooled with official RULER results.

## 5. LongBench Data Preparation

### Source

LongBench uses the official THUDM/LongBench public test data:

- Dataset repository: <https://huggingface.co/datasets/THUDM/LongBench>
- Project reader: `lmaf_experiments/src/lmaf/data/longbench.py`

The preparation code first checks for local task JSONL files under `external/LongBench/` or `external/LongBench/data/`. If they are unavailable, it downloads and reads `data.zip` from the official THUDM/LongBench dataset repository through Hugging Face Hub.

LongBench contexts, questions, and reference answers are never synthetically generated or rewritten by this project. The preparation step only:

1. reads the official test records;
2. retains the original context, question, and answer(s);
3. constructs a model prompt;
4. adds task category and token-length metadata;
5. writes the result as project-standard JSONL.

### Prompt Handling

When a local LongBench `dataset2prompt.json` is available, the project uses its task-specific prompt template. Otherwise, it applies a neutral shared wrapper of the form “context + question + answer”.

Therefore, the correct statement is:

> The study uses official LongBench test samples with a standardized prompt wrapper when the task-specific official template is not locally available.

The data and reference answers remain official LongBench content, but not every run necessarily uses the original task-specific LongBench prompt shell.

### Prepared Task Set and Budget Selection

The Framework V2 preparation stage includes nine representative tasks:

| Category | Tasks |
|---|---|
| Single-document QA | `narrativeqa`, `qasper`, `multifieldqa_en` |
| Multi-document QA | `hotpotqa`, `2wikimqa`, `musique` |
| Summarization | `gov_report`, `qmsum`, `multi_news` |

Preparation is capped at 200 samples per task. The current prepared collection contains 1,750 rows: 200 rows for eight tasks and 150 rows for `multifieldqa_en`.

For the budget-core run, six representative tasks are selected: `narrativeqa`, `qasper`, `hotpotqa`, `2wikimqa`, `gov_report`, and `multi_news`. The selection is deterministic: the first 50 samples after stable sample-ID ordering are retained for each task, yielding 300 LongBench samples. This is a pre-defined cost-controlled subset, not the full official LongBench test set.

### Reproduction

```cmd
cd lmaf_experiments
python scripts\run_longbench.py ^
  --prepare-only ^
  --tasks narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,gov_report,qmsum,multi_news ^
  --longbench-repo external\LongBench ^
  --sample-limit 200 ^
  --output data\processed\longbench_ruler_batch\framework_v2\longbench
```

The 300-row budget-core selection is implemented in `scripts/run_budget_core.py` and `scripts/run_official_budget_topup.py`. Selection does not alter the original LongBench content.

## 6. Reproducibility and Reporting Rules

- NIAH and RULER generation use fixed seed `42`.
- NIAH records actual needle placement; RULER records official task name and source file; LongBench records task category and actual prompt length.
- The selected 300 NIAH, 300 LongBench, and 240 RULER samples are recorded in `data/processed/official_budget_topup/{run_id}/budget_manifest.jsonl`.
- API failures are operational records, not model answers. They should be reported separately from accuracy or other model-performance metrics.
- Earlier local NIAH samples and RULER fallback samples are retained for development history only. They should not be mixed into the final official-route summary.

## 7. Suggested Thesis Wording

> NIAH samples were generated with the Needle-in-a-Haystack context and insertion framework, using single-needle, multi-needle, and sequential retrieval variants. RULER samples were generated from the official NVIDIA/RULER synthetic task generators at 4K, 16K, and 32K context lengths. LongBench used official THUDM/LongBench test records; the original contexts, questions, and reference answers were retained, while prompts were standardized and a pre-defined representative subset was selected for cost-controlled evaluation.
