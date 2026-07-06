# Phase 5 APBS Two-Day Experiment Framework

This folder contains a compact experiment framework for proving the MVP claim:

> On Qwen 9B at 16K NIAH, Adaptive Position-Aware Base Scaling (APBS) improves middle-position retrieval over both baseline RoPE and global NTK scaling, while preserving head/tail accuracy.

## Recommended Two-Day Scope

Day 1:

1. Generate the 16K main dataset.
2. Run `baseline`, `ntk`, and `apbs` on 50 samples at 10%, 50%, and 90%.
3. Aggregate metrics and check the 50% position lift.

The Day 1/Day 2 scripts use `make_niah_hard_dataset.py` for the main experiments. This adds many near-key decoys and similar checksum values to reduce ceiling effects. The older `make_niah_dataset.py` is kept only for lightweight smoke tests.

Day 2:

1. Run gamma sensitivity at the 50% position only: `0.1`, `0.3`, `0.5`.
2. Run a small 32K validation set with 10-20 samples per position.
3. Generate plots, bootstrap confidence intervals, and the short report.

## Install

On the GPU machine:

```powershell
pip install -r requirements.txt
python scripts/check_environment.py
```

## Fast Path

The generated datasets are already included in `data/`. On a PowerShell GPU host, run:

```powershell
.\RUN_DAY1.ps1 -Model "Qwen/Qwen3.5-9B" -LoadIn4bit
.\RUN_DAY2.ps1 -Model "Qwen/Qwen3.5-9B" -LoadIn4bit
```

If the Hugging Face model id is unavailable, replace `-Model` with your local model path.

If downloading from Hugging Face is slow, download the model first and run from the local path:

```powershell
.\DOWNLOAD_MODEL.ps1 -Model "Qwen/Qwen3.5-9B" -LocalDir "D:\hf_models\Qwen3.5-9B"
.\RUN_DAY1.ps1 -Model "D:\hf_models\Qwen3.5-9B" -DType bf16 -LoadIn4bit
```

If the HF mirror is unstable, use ModelScope:

```powershell
.\DOWNLOAD_MODEL_MODELSCOPE.ps1 -Model "Qwen/Qwen3.5-9B" -LocalDir "D:\hf_models\Qwen3.5-9B"
.\RUN_DAY1.ps1 -Model "D:\hf_models\Qwen3.5-9B" -DType bf16 -LoadIn4bit
```

On a Linux GPU host:

```bash
LOAD_IN_4BIT=1 bash RUN_DAY1.sh "Qwen/Qwen3.5-9B"
LOAD_IN_4BIT=1 bash RUN_DAY2.sh "Qwen/Qwen3.5-9B"
```

For manual execution, follow the steps below.

## Resumable Runner

To count existing valid JSONL rows and only run missing samples:

```powershell
.\RUN_RESUME.ps1 -Model "D:\hf_models\Qwen3.5-9B" -ModelKey "qwen35_9b" -Phase day1 -DType bf16 -LoadIn4bit
.\RUN_RESUME.ps1 -Model "D:\hf_models\Qwen3.5-9B" -ModelKey "qwen35_9b" -Phase full -DType bf16 -LoadIn4bit
```

Use `-DryRun` first to print the missing counts without launching model inference.

## 1. Generate Datasets

Main 16K run:

```powershell
python scripts/make_niah_dataset.py --lengths 16384 --positions 10,50,90 --samples-per-cell 50 --output data/niah_16k_main.jsonl
```

Gamma sensitivity, middle only:

```powershell
python scripts/make_niah_dataset.py --lengths 16384 --positions 50 --samples-per-cell 50 --output data/niah_16k_gamma.jsonl
```

Optional 32K small validation:

```powershell
python scripts/make_niah_dataset.py --lengths 32768 --positions 10,50,90 --samples-per-cell 15 --output data/niah_32k_small.jsonl
```

## 2. Run Inference

Set your model name or local model path:

```powershell
$env:MODEL_NAME="Qwen/Qwen3.5-9B"
```

Main 16K matrix:

```powershell
python scripts/run_apbs_niah.py --model $env:MODEL_NAME --dataset data/niah_16k_main.jsonl --method baseline --output results/raw/16k_baseline.jsonl
python scripts/run_apbs_niah.py --model $env:MODEL_NAME --dataset data/niah_16k_main.jsonl --method ntk --output results/raw/16k_ntk.jsonl --target-length 16384
python scripts/run_apbs_niah.py --model $env:MODEL_NAME --dataset data/niah_16k_main.jsonl --method apbs --output results/raw/16k_apbs_g03.jsonl --target-length 16384 --gamma 0.3
```

Gamma sensitivity:

```powershell
python scripts/run_apbs_niah.py --model $env:MODEL_NAME --dataset data/niah_16k_gamma.jsonl --method apbs --output results/raw/16k_apbs_g01_mid.jsonl --target-length 16384 --gamma 0.1
python scripts/run_apbs_niah.py --model $env:MODEL_NAME --dataset data/niah_16k_gamma.jsonl --method apbs --output results/raw/16k_apbs_g05_mid.jsonl --target-length 16384 --gamma 0.5
```

Optional 32K small:

```powershell
python scripts/run_apbs_niah.py --model $env:MODEL_NAME --dataset data/niah_32k_small.jsonl --method baseline --output results/raw/32k_baseline.jsonl
python scripts/run_apbs_niah.py --model $env:MODEL_NAME --dataset data/niah_32k_small.jsonl --method ntk --output results/raw/32k_ntk.jsonl --target-length 32768
python scripts/run_apbs_niah.py --model $env:MODEL_NAME --dataset data/niah_32k_small.jsonl --method apbs --output results/raw/32k_apbs_g03.jsonl --target-length 32768 --gamma 0.3
```

## 3. Aggregate, Plot, Report

```powershell
python scripts/analyze_apbs_results.py --inputs results/raw/*.jsonl --output-dir results/analysis
```

The analyzer writes:

- `metrics_by_method.csv`
- `position_curves.png`
- `gamma_sensitivity.png` if gamma files are present
- `bootstrap_ci.csv`
- `phase5_apbs_report.md`

## Success Criteria For The Two-Day Claim

Use these MVP thresholds:

- `APBS@50% - Baseline@50% >= 0.10`
- `APBS@50% - NTK@50% >= 0.05`
- Head/tail average drop under APBS `<= 0.05`
- U-shape flattening index decreases by at least `30%` vs baseline
- Bootstrap 95% CI for APBS middle lift should preferably stay above zero

If all pass, write the claim narrowly:

> In a Qwen 9B 16K NIAH setting, APBS provides causal intervention evidence that position-aware RoPE base compensation improves middle-position retrieval beyond global NTK scaling.
