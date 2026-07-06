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

Write-Host "== APBS Day 1: environment smoke =="
python scripts/smoke_patch.py

Write-Host "== APBS Day 1: 4K model smoke dataset =="
python scripts/make_niah_dataset.py --lengths 4096 --positions 50 --samples-per-cell 2 --output data/smoke_4k.jsonl

Write-Host "== APBS Day 1: baseline smoke =="
python scripts/run_apbs_niah.py --model $Model --dataset data/smoke_4k.jsonl --method baseline --output results/raw/smoke_baseline.jsonl --limit 2 --dtype $DType --device-map $DeviceMap @QuantArgs

Write-Host "== APBS Day 1: NTK smoke =="
python scripts/run_apbs_niah.py --model $Model --dataset data/smoke_4k.jsonl --method ntk --output results/raw/smoke_ntk.jsonl --target-length 4096 --limit 2 --dtype $DType --device-map $DeviceMap @QuantArgs

Write-Host "== APBS Day 1: APBS smoke =="
python scripts/run_apbs_niah.py --model $Model --dataset data/smoke_4k.jsonl --method apbs --output results/raw/smoke_apbs.jsonl --target-length 4096 --gamma 0.3 --limit 2 --dtype $DType --device-map $DeviceMap @QuantArgs

Write-Host "== APBS Day 1: generate 16K main dataset =="
python scripts/make_niah_hard_dataset.py --lengths 16384 --positions 10,50,90 --samples-per-cell 50 --distractors 64 --output data/niah_16k_main.jsonl

Write-Host "== APBS Day 1: 16K baseline =="
python scripts/run_apbs_niah.py --model $Model --dataset data/niah_16k_main.jsonl --method baseline --output results/raw/16k_baseline.jsonl --dtype $DType --device-map $DeviceMap @QuantArgs

Write-Host "== APBS Day 1: 16K global NTK =="
python scripts/run_apbs_niah.py --model $Model --dataset data/niah_16k_main.jsonl --method ntk --output results/raw/16k_ntk.jsonl --target-length 16384 --dtype $DType --device-map $DeviceMap @QuantArgs

Write-Host "== APBS Day 1: 16K APBS gamma=0.3 =="
python scripts/run_apbs_niah.py --model $Model --dataset data/niah_16k_main.jsonl --method apbs --output results/raw/16k_apbs_g03.jsonl --target-length 16384 --gamma 0.3 --dtype $DType --device-map $DeviceMap @QuantArgs

Write-Host "== APBS Day 1: analysis =="
python scripts/analyze_apbs_results.py --inputs "results/raw/16k_*.jsonl" --output-dir results/analysis

Write-Host "Done. Check results/analysis/phase5_apbs_report.md"
