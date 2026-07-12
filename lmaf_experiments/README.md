# LMAF Experiments

This project implements the local LongBench, NIAH, RULER, and PAC-Test experiment pipeline described in the implementation spec.

The pipeline can call either a local vLLM/OpenAI-compatible endpoint or SiliconFlow's OpenAI-compatible API. Use `--provider local` for local deployments and `--provider siliconflow` for SiliconFlow.

For SiliconFlow-specific setup, read [docs/SILICONFLOW_API.md](docs/SILICONFLOW_API.md).

For automatic multi-model NIAH runs based on Framework V2.0, read [docs/NIAH_BATCH.md](docs/NIAH_BATCH.md) and use `scripts/run_niah_batch.py`.

For automatic multi-model LongBench + RULER runs, read [docs/LONGBENCH_RULER_BATCH.md](docs/LONGBENCH_RULER_BATCH.md) and use `scripts/run_longbench_ruler_batch.py`.

For automatic multi-model PAC-Test A/B/C/D runs, read [docs/PAC_BATCH.md](docs/PAC_BATCH.md) and use `scripts/run_pac_batch.py`.

## Setup

```bash
conda create -n lmaf python=3.11 -y
conda activate lmaf
pip install -U pip
pip install -r requirements.txt
```

Run offline smoke tests:

```bash
pytest tests/ -q
python scripts/run_niah.py --generate-only --lengths 4096 --positions 50 --samples-per-cell 2 --output data/generated/smoke_niah
```

## Provider Selection

Local vLLM is still the default:

```bash
python scripts/run_niah.py --provider local --model qwen35_9b ...
```

SiliconFlow uses `SILICONFLOW_API_KEY` and defaults to `https://api.siliconflow.cn/v1`:

```bash
export SILICONFLOW_API_KEY="sk-your-key"
python scripts/siliconflow_healthcheck.py --model Qwen/Qwen3.5-9B
python scripts/run_niah.py --provider siliconflow --model Qwen/Qwen3.5-9B --input data/generated/smoke_niah --output results/raw/smoke/siliconflow_qwen35_9b.jsonl --resume
```

Project aliases such as `qwen35_9b` are resolved to SiliconFlow model names when `--provider siliconflow` is used.

## Serve A Local Model

```bash
bash scripts/launch_vllm.sh Qwen/Qwen3.5-9B qwen35_9b 8000 4096 1
```

The scripts use fixed inference parameters by default:

- `temperature=0.0`
- `top_p=1.0`
- `max_tokens=512`
- `seed=42` for generated data

## Stage 1: 4K NIAH Smoke Test

```bash
python scripts/run_niah.py \
  --model qwen35_9b \
  --endpoint http://localhost:8000/v1 \
  --api-key local-token \
  --input data/generated/smoke_niah \
  --output results/raw/smoke/qwen35_9b.jsonl \
  --resume
```

Passing means the endpoint returns stable responses with no service errors. For a production deployment, require 2/2 exact or contains-answer correctness before moving to longer contexts.

## NIAH

```bash
python scripts/run_niah.py \
  --generate-only \
  --lengths 4096,16384,32768,65536 \
  --positions 10,50,90 \
  --samples-per-cell 50 \
  --output data/generated/niah

python scripts/run_niah.py \
  --model qwen35_9b \
  --endpoint http://localhost:8000/v1 \
  --api-key local-token \
  --input data/generated/niah \
  --output results/raw/niah/qwen35_9b.jsonl \
  --resume

python scripts/aggregate_results.py \
  --input results/raw/niah \
  --experiment niah \
  --output results/aggregate/niah_results.csv

python scripts/plot_results.py \
  --input results/aggregate/niah_results.csv \
  --plot niah_position_curve \
  --output results/figures/niah_position_curve.png
```

## PAC-Test

```bash
python scripts/run_pac.py --generate-only --subset A_position --length 16384 --positions 10,25,50,75,90 --samples-per-cell 50 --output data/generated/pac/A_position
python scripts/run_pac.py --model qwen35_9b --endpoint http://localhost:8000/v1 --api-key local-token --subset A_position --input data/generated/pac/A_position --output results/raw/pac/A_position/qwen35_9b.jsonl --resume

python scripts/run_pac.py --generate-only --subset B_interference --length 16384 --position 10 --densities 0,25,50,75,90 --interference-types in_domain,out_domain,random_noise --samples-per-cell 30 --output data/generated/pac/B_interference
python scripts/run_pac.py --generate-only --subset C_overlap --length 16384 --similarities high,medium,low,none --distances near,medium,far --samples-per-cell 20 --output data/generated/pac/C_overlap
python scripts/run_pac.py --generate-only --subset D_multihop --lengths 8192,16384,32768 --hops 2,3,4 --hop-distances 1024,4096,8192 --samples-per-cell 20 --output data/generated/pac/D_multihop
```

Aggregate and plot:

```bash
python scripts/aggregate_results.py --input results/raw/pac --experiment pac --output results/aggregate/pac_all_results.csv
python scripts/plot_results.py --input results/aggregate/pac_all_results.csv --plot pac_A_position_curve --output results/figures/pac_A_position_curve.png
python scripts/plot_results.py --input results/aggregate/pac_all_results.csv --plot pac_B_density_curve --output results/figures/pac_B_density_curve.png
python scripts/plot_results.py --input results/aggregate/pac_all_results.csv --plot pac_C_confusion_matrix --output results/figures/pac_C_confusion_matrix.png
python scripts/plot_results.py --input results/aggregate/pac_all_results.csv --plot pac_D_multihop_decay --output results/figures/pac_D_multihop_decay.png
```

## LongBench

Prepare data from local LongBench JSONL files or HuggingFace `THUDM/LongBench` `data.zip`:

```bash
python scripts/run_longbench.py \
  --prepare-only \
  --tasks narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,gov_report,qmsum,multi_news \
  --sample-limit 200 \
  --output data/processed/longbench
```

Run:

```bash
python scripts/run_longbench.py \
  --model qwen35_9b \
  --endpoint http://localhost:8000/v1 \
  --api-key local-token \
  --tasks narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,gov_report,qmsum,multi_news \
  --input data/processed/longbench \
  --sample-limit 200 \
  --output results/raw/longbench/qwen35_9b.jsonl \
  --resume
```

Samples over `--max-model-len` are written as `skipped_overlength` unless `--truncate middle` is explicitly set.

## RULER

Use the official RULER repository when available. This project also provides a marked fallback:

```bash
python scripts/run_ruler.py \
  --generate-only \
  --lengths 4096,16384,32768 \
  --tasks niah,variable_tracking,common_words_extraction,freq_words_extraction,qa_squad,qa_hotpotqa \
  --samples-per-cell 50 \
  --output data/generated/ruler

python scripts/run_ruler.py \
  --model qwen35_9b \
  --endpoint http://localhost:8000/v1 \
  --api-key local-token \
  --input data/generated/ruler \
  --output results/raw/ruler/qwen35_9b.jsonl \
  --resume

python scripts/aggregate_results.py --input results/raw/ruler --experiment ruler --output results/aggregate/ruler_results.csv
python scripts/plot_results.py --input results/aggregate/ruler_results.csv --plot ruler_effective_context --output results/figures/ruler_effective_context.png
```

Fallback rows include `implementation=ruler_fallback` and should not be mixed with official RULER results.

## Metadata

```bash
python scripts/write_run_metadata.py --output results/logs/run_metadata.json
```

Every raw JSONL row stores the prompt, prediction, answer, score, latency, token counts, seed, timestamp, and error field.
