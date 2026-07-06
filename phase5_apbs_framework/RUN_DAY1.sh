#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-Qwen/Qwen3.5-9B}"
DTYPE="${DTYPE:-auto}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
QUANT_ARGS=()
if [[ "${LOAD_IN_4BIT:-0}" == "1" ]]; then
  QUANT_ARGS+=(--load-in-4bit)
fi

mkdir -p data results/raw results/analysis

echo "== APBS Day 1: environment smoke =="
python scripts/smoke_patch.py

echo "== APBS Day 1: 4K model smoke dataset =="
python scripts/make_niah_dataset.py --lengths 4096 --positions 50 --samples-per-cell 2 --output data/smoke_4k.jsonl

echo "== APBS Day 1: baseline smoke =="
python scripts/run_apbs_niah.py --model "$MODEL" --dataset data/smoke_4k.jsonl --method baseline --output results/raw/smoke_baseline.jsonl --limit 2 --dtype "$DTYPE" --device-map "$DEVICE_MAP" "${QUANT_ARGS[@]}"

echo "== APBS Day 1: NTK smoke =="
python scripts/run_apbs_niah.py --model "$MODEL" --dataset data/smoke_4k.jsonl --method ntk --output results/raw/smoke_ntk.jsonl --target-length 4096 --limit 2 --dtype "$DTYPE" --device-map "$DEVICE_MAP" "${QUANT_ARGS[@]}"

echo "== APBS Day 1: APBS smoke =="
python scripts/run_apbs_niah.py --model "$MODEL" --dataset data/smoke_4k.jsonl --method apbs --output results/raw/smoke_apbs.jsonl --target-length 4096 --gamma 0.3 --limit 2 --dtype "$DTYPE" --device-map "$DEVICE_MAP" "${QUANT_ARGS[@]}"

echo "== APBS Day 1: generate 16K main dataset =="
python scripts/make_niah_hard_dataset.py --lengths 16384 --positions 10,50,90 --samples-per-cell 50 --distractors 64 --output data/niah_16k_main.jsonl

echo "== APBS Day 1: 16K baseline =="
python scripts/run_apbs_niah.py --model "$MODEL" --dataset data/niah_16k_main.jsonl --method baseline --output results/raw/16k_baseline.jsonl --dtype "$DTYPE" --device-map "$DEVICE_MAP" "${QUANT_ARGS[@]}"

echo "== APBS Day 1: 16K global NTK =="
python scripts/run_apbs_niah.py --model "$MODEL" --dataset data/niah_16k_main.jsonl --method ntk --output results/raw/16k_ntk.jsonl --target-length 16384 --dtype "$DTYPE" --device-map "$DEVICE_MAP" "${QUANT_ARGS[@]}"

echo "== APBS Day 1: 16K APBS gamma=0.3 =="
python scripts/run_apbs_niah.py --model "$MODEL" --dataset data/niah_16k_main.jsonl --method apbs --output results/raw/16k_apbs_g03.jsonl --target-length 16384 --gamma 0.3 --dtype "$DTYPE" --device-map "$DEVICE_MAP" "${QUANT_ARGS[@]}"

echo "== APBS Day 1: analysis =="
python scripts/analyze_apbs_results.py --inputs "results/raw/16k_*.jsonl" --output-dir results/analysis

echo "Done. Check results/analysis/phase5_apbs_report.md"
