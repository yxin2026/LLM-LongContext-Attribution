from __future__ import annotations

EXCLUDED_MODELS = {
    "deepseek_r1_distill_qwen_14b",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "gemma4_26b_a4b",
    "google/gemma-4-26B-A4B-it",
    "gemma4_31b",
    "google/gemma-4-31B-it",
}


def is_excluded_model(value: object) -> bool:
    return str(value or "") in EXCLUDED_MODELS

