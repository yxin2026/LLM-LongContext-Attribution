#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="$1"
SERVED_NAME="$2"
PORT="${3:-8000}"
MAX_LEN="${4:-32768}"
TP_SIZE="${5:-1}"
QUANTIZATION="${6:-}"

extra_args=()
if [[ -n "$QUANTIZATION" && "$QUANTIZATION" != "none" ]]; then
  extra_args+=(--quantization "$QUANTIZATION")
fi

vllm serve "$MODEL_ID" \
  --served-model-name "$SERVED_NAME" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --api-key local-token \
  --tensor-parallel-size "$TP_SIZE" \
  --gpu-memory-utilization 0.90 \
  --max-model-len "$MAX_LEN" \
  --trust-remote-code \
  "${extra_args[@]}"

