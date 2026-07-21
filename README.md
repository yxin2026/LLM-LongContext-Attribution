# Long-Context Attribution Framework

This repository is an experimental framework for **attributing long-context language-model failures to specific mechanisms**. It combines public long-context benchmarks with controlled synthetic tests so that aggregate accuracy can be interpreted in terms of position sensitivity, distractor interference, entity-binding capacity, and multihop reasoning.

The project has four integrated layers:

1. **Public benchmark evaluation** with NIAH, RULER, and LongBench.
2. **PAC-Test**, a controlled synthetic benchmark for diagnosing mechanisms that public benchmark scores alone cannot isolate.
3. **LMAF**, the reusable execution, scoring, aggregation, and reporting pipeline.
4. **Result and paper artifacts**, including compact tables, figures, review material, and workbook templates.

> **Important:** result folders include completed, incomplete, and retryable runs. Read completion-status tables before making cross-model claims. A raw or partial run must not be treated as a final benchmark result.

## Research questions

The repository is intended to support the following questions:

- How does retrieval change as relevant evidence moves through a long context?
- At what distractor density does a model begin to confuse highly similar but invalid evidence with the target?
- How many entity-attribute-value bindings can remain stable under long-context interference?
- Can a model follow a valid multihop evidence chain while rejecting plausible false chains?
- Do conclusions from controlled PAC-Test settings agree with results on NIAH, RULER, and LongBench?

## Repository layout

```text
PAC-Test-Dataset/
├── PAC/                            # PAC-Test v5/v2.1 datasets, manifests, and runners
├── lmaf_experiments/               # Core LMAF evaluation framework
│   ├── configs/                    # Models, runtime, and experiment grids
│   ├── docs/                       # Batch-run and provider instructions
│   ├── external/RULER/             # RULER integration
│   ├── scripts/                    # Generation, inference, aggregation, and reporting
│   ├── src/lmaf/                   # Data, inference, evaluation, and utility modules
│   └── tests/                      # Offline unit tests
├── data_generated/                 # Generated NIAH/PAC data, calibration data, and smoke sets
├── Results/                        # Aggregated outputs, reports, figures, review materials
├── form/                           # Paper-oriented table and figure workbooks
├── docs/                           # Repository-level data and status documentation
└── README.md                       # This overview
```

## Evaluation components

| Component | Role in the project | Main controlled variables |
| --- | --- | --- |
| **NIAH** | Tests retrieval of a target fact across long contexts. | Context length and target position. |
| **RULER** | Tests synthetic long-context capabilities beyond simple retrieval. | Task family and context length. |
| **LongBench** | Tests long-context performance on public, task-oriented benchmarks. | Dataset task and token budget. |
| **PAC-A** | Isolates position effects under high-similarity interference. | Target position. |
| **PAC-B** | Finds the interference-density threshold. | Number and type of distractors. |
| **PAC-C** | Measures entity-property binding capacity. | Entity count and simultaneous query count. |
| **PAC-D** | Tests multihop chain tracking under false-chain interference. | Hop count and false-chain count. |

### PAC-Test versions

PAC-Test is the project-specific diagnostic benchmark. Formal v5 contains 291 samples across PAC-A through PAC-D at approximately 32K tokens. PAC-D v2.1 is a separate 12-sample, strong-model multihop extension using a stricter verifier gate.

Do not aggregate PAC-D v5 and PAC-D v2.1 without explicit version labels. See [`PAC/README.md`](PAC/README.md) for full dataset definitions, metrics, sample format, runner commands, and the v5/v2.1 boundary.

## Setup

The project uses Python 3.11 for the LMAF environment.

```bash
conda create -n lmaf python=3.11 -y
conda activate lmaf
pip install -U pip
pip install -r lmaf_experiments/requirements.txt
```

Run offline tests before running model inference:

```bash
cd lmaf_experiments
pytest tests/ -q
cd ..
```

## Providers and runtime configuration

LMAF supports two OpenAI-compatible inference modes:

- **Local:** a local vLLM or compatible endpoint, configured by default at `http://localhost:8000/v1`.
- **SiliconFlow:** configure `SILICONFLOW_API_KEY` as an environment variable, then use `--provider siliconflow`.

Default generation/evaluation settings are stored in `lmaf_experiments/configs/runtime.yaml`: temperature `0.0`, top-p `1.0`, seed `42`, and fixed retry/backoff behavior. Model aliases and experiment grids are defined in `configs/models.yaml` and `configs/experiments.yaml`.

Do not commit API keys, copied `.env` files, or endpoint credentials.

## Quick-start workflows

Run commands below from the repository root unless otherwise stated.

### Generate a small NIAH smoke set

```bash
python lmaf_experiments/scripts/run_niah.py \
  --generate-only \
  --lengths 4096 \
  --positions 50 \
  --samples-per-cell 2 \
  --output data_generated/smoke_niah
```

### Run a PAC quality pilot

The pilot is the recommended validation step before a full API queue.

```bat
python PAC\run_pac2_pilot.py --run-id pac2_pilot_v5_quality --max-workers 4 --max-in-flight 4 --request-delay-sec 12 --timeout 360 --retry 2 --max-tokens 192
```

Summarize a PAC run without submitting more model requests:

```bat
python PAC\run_pac2_pilot.py --run-id pac2_pilot_v5_quality --summarize-only
```

### Run multi-model batches

Use the documented orchestrators for larger jobs:

| Workflow | Entry point | Documentation |
| --- | --- | --- |
| NIAH | `lmaf_experiments/scripts/run_niah_batch.py` | [`NIAH_BATCH.md`](lmaf_experiments/docs/NIAH_BATCH.md) |
| LongBench + RULER | `lmaf_experiments/scripts/run_longbench_ruler_batch.py` | [`LONGBENCH_RULER_BATCH.md`](lmaf_experiments/docs/LONGBENCH_RULER_BATCH.md) |
| PAC | `lmaf_experiments/scripts/run_pac_batch.py` | [`PAC_BATCH.md`](lmaf_experiments/docs/PAC_BATCH.md) |
| Official NIAH/RULER data | generation/import scripts | [`OFFICIAL_NIAH_RULER.md`](lmaf_experiments/docs/OFFICIAL_NIAH_RULER.md) |
| SiliconFlow | provider health check and runners | [`SILICONFLOW_API.md`](lmaf_experiments/docs/SILICONFLOW_API.md) |

The PAC queue runners are resumable: reusing the same run ID skips completed rows and retries configured transient failures. Use a small pilot first and retain raw output rather than overwriting it.

## Data provenance and versioning

| Data category | Location | Notes |
| --- | --- | --- |
| Formal PAC data | `PAC/data/` | Source of truth for PAC v5 and PAC-D v2.1 analyses. |
| Generated datasets | `data_generated/` | Includes smoke, audit, debug, calibration, and batch data; not every subdirectory is formal benchmark data. |
| Public benchmark data | LMAF data workflow | Generated/imported through documented NIAH, RULER, and LongBench preparation steps. |
| Result artifacts | `Results/` | Compact aggregate outputs and reports; inspect completion state before interpretation. |

[`docs/PUBLIC_BENCHMARK_DATA_README.md`](docs/PUBLIC_BENCHMARK_DATA_README.md) documents the provenance and preparation boundary for NIAH, RULER, and LongBench. Every experiment should record the dataset version, model ID, provider, context length, runtime settings, run ID, and random seed.

## Results and reporting

`Results/` contains:

- `experiment_summary_20260705/`: completion-status tables, cleaned aggregate tables, and figures derived from existing raw outputs;
- `lmaf_results_summary/`: result reports, workbooks, figures, and paper-facing summaries;
- `niah_review_20260705/`: NIAH examples and review tables;
- `source_questions.docx`, `source_media/`, and `work/`: supporting review materials.

The project uses exact accuracy, field accuracy/retention, partial rate, and decoy capture rate for PAC diagnostics. API failures are operational errors, not model-answer errors, and must be reported separately.

To regenerate outputs, use the LMAF aggregation and reporting scripts:

```bash
python lmaf_experiments/scripts/aggregate_results.py --help
python lmaf_experiments/scripts/plot_results.py --help
python lmaf_experiments/scripts/summarize_pac_v21_results.py --help
python lmaf_experiments/scripts/summarize_public_benchmarks.py --help
```

## Reproducibility rules

1. Run offline tests and a provider health check before a large batch.
2. Keep PAC v5 and PAC-D v2.1 results separate unless the comparison is explicitly versioned.
3. Do not count timeouts, rate limits, or API failures as model mistakes.
4. Do not manually alter prompts, gold answers, or model predictions; create a new data or run version instead.
5. Preserve run metadata and raw JSONL outputs; derive summaries from a recorded run ID.
6. Report PAC-A/B/C/D at subset level in addition to any overall aggregate.

## License and third-party terms

No repository-wide license is currently declared. Before public release, add a license and review the redistribution, attribution, and usage terms for LongBench, RULER, model weights, APIs, and any external datasets.

## Further reading

- [`PAC/README.md`](PAC/README.md): PAC dataset-specific design and runner reference.
- [`lmaf_experiments/README.md`](lmaf_experiments/README.md): LMAF command reference.
- [`docs/PUBLIC_BENCHMARK_DATA_README.md`](docs/PUBLIC_BENCHMARK_DATA_README.md): public benchmark data preparation and provenance.
- [`docs/STATUS.md`](docs/STATUS.md): repository documentation status and recent work.
