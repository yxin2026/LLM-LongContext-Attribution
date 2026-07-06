param(
  [string]$Model = "Qwen/Qwen3.5-9B",
  [string]$DType = "auto",
  [string]$DeviceMap = "auto",
  [switch]$LoadIn4bit
)

$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force -Path data | Out-Null
New-Item -ItemType Directory -Force -Path results/raw | Out-Null
New-Item -ItemType Directory -Force -Path results/analysis | Out-Null

$QuantArgs = @()
if ($LoadIn4bit) {
  $QuantArgs += "--load-in-4bit"
}

Write-Host "== APBS Day 2: gamma sensitivity dataset =="
python scripts/make_niah_hard_dataset.py --lengths 16384 --positions 50 --samples-per-cell 50 --distractors 64 --output data/niah_16k_gamma.jsonl

Write-Host "== APBS Day 2: gamma=0.1 middle =="
python scripts/run_apbs_niah.py --model $Model --dataset data/niah_16k_gamma.jsonl --method apbs --output results/raw/16k_apbs_g01_mid.jsonl --target-length 16384 --gamma 0.1 --dtype $DType --device-map $DeviceMap @QuantArgs

Write-Host "== APBS Day 2: gamma=0.5 middle =="
python scripts/run_apbs_niah.py --model $Model --dataset data/niah_16k_gamma.jsonl --method apbs --output results/raw/16k_apbs_g05_mid.jsonl --target-length 16384 --gamma 0.5 --dtype $DType --device-map $DeviceMap @QuantArgs

Write-Host "== APBS Day 2: optional 32K small dataset =="
python scripts/make_niah_hard_dataset.py --lengths 32768 --positions 10,50,90 --samples-per-cell 15 --distractors 96 --output data/niah_32k_small.jsonl

Write-Host "== APBS Day 2: 32K baseline =="
python scripts/run_apbs_niah.py --model $Model --dataset data/niah_32k_small.jsonl --method baseline --output results/raw/32k_baseline.jsonl --dtype $DType --device-map $DeviceMap @QuantArgs

Write-Host "== APBS Day 2: 32K global NTK =="
python scripts/run_apbs_niah.py --model $Model --dataset data/niah_32k_small.jsonl --method ntk --output results/raw/32k_ntk.jsonl --target-length 32768 --dtype $DType --device-map $DeviceMap @QuantArgs

Write-Host "== APBS Day 2: 32K APBS gamma=0.3 =="
python scripts/run_apbs_niah.py --model $Model --dataset data/niah_32k_small.jsonl --method apbs --output results/raw/32k_apbs_g03.jsonl --target-length 32768 --gamma 0.3 --dtype $DType --device-map $DeviceMap @QuantArgs

Write-Host "== APBS Day 2: final analysis =="
python scripts/analyze_apbs_results.py --inputs "results/raw/*.jsonl" --output-dir results/analysis

Write-Host "Done. Check results/analysis/phase5_apbs_report.md"
