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

echo "== APBS Day 2: gamma sensitivity dataset =="
python scripts/make_niah_hard_dataset.py --lengths 16384 --positions 50 --samples-per-cell 50 --distractors 64 --output data/niah_16k_gamma.jsonl

echo "== APBS Day 2: gamma=0.1 middle =="
python scripts/run_apbs_niah.py --model "$MODEL" --dataset data/niah_16k_gamma.jsonl --method apbs --output results/raw/16k_apbs_g01_mid.jsonl --target-length 16384 --gamma 0.1 --dtype "$DTYPE" --device-map "$DEVICE_MAP" "${QUANT_ARGS[@]}"

echo "== APBS Day 2: gamma=0.5 middle =="
python scripts/run_apbs_niah.py --model "$MODEL" --dataset data/niah_16k_gamma.jsonl --method apbs --output results/raw/16k_apbs_g05_mid.jsonl --target-length 16384 --gamma 0.5 --dtype "$DTYPE" --device-map "$DEVICE_MAP" "${QUANT_ARGS[@]}"

echo "== APBS Day 2: optional 32K small dataset =="
python scripts/make_niah_hard_dataset.py --lengths 32768 --positions 10,50,90 --samples-per-cell 15 --distractors 96 --output data/niah_32k_small.jsonl

echo "== APBS Day 2: 32K baseline =="
python scripts/run_apbs_niah.py --model "$MODEL" --dataset data/niah_32k_small.jsonl --method baseline --output results/raw/32k_baseline.jsonl --dtype "$DTYPE" --device-map "$DEVICE_MAP" "${QUANT_ARGS[@]}"

echo "== APBS Day 2: 32K global NTK =="
python scripts/run_apbs_niah.py --model "$MODEL" --dataset data/niah_32k_small.jsonl --method ntk --output results/raw/32k_ntk.jsonl --target-length 32768 --dtype "$DTYPE" --device-map "$DEVICE_MAP" "${QUANT_ARGS[@]}"

echo "== APBS Day 2: 32K APBS gamma=0.3 =="
python scripts/run_apbs_niah.py --model "$MODEL" --dataset data/niah_32k_small.jsonl --method apbs --output results/raw/32k_apbs_g03.jsonl --target-length 32768 --gamma 0.3 --dtype "$DTYPE" --device-map "$DEVICE_MAP" "${QUANT_ARGS[@]}"

echo "== APBS Day 2: final analysis =="
python scripts/analyze_apbs_results.py --inputs "results/raw/*.jsonl" --output-dir results/analysis

echo "Done. Check results/analysis/phase5_apbs_report.md"
