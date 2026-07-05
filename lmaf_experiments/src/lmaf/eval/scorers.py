from __future__ import annotations

import re
from typing import Any

from lmaf.eval.metrics import best_contains, exact_match, normalize_answer, rouge_l, set_f1, token_f1


def score_niah(prediction: str, answer: str) -> dict[str, Any]:
    contains = best_contains(prediction, answer)
    return {
        "score": contains,
        "metric": "exact_match_or_contains",
        "contains_answer": int(contains),
    }


def score_qa(prediction: str, answers: list[str] | str) -> dict[str, Any]:
    em = exact_match(prediction, answers)
    f1 = token_f1(prediction, answers)
    return {"score": max(em, f1), "metric": "qa_f1", "exact_match": em, "f1": f1}


def score_summarization(prediction: str, answers: list[str] | str) -> dict[str, Any]:
    score = rouge_l(prediction, answers)
    return {"score": score, "metric": "rouge_l", "rouge_l": score}


def score_topk(prediction: str, answers: list[str]) -> dict[str, Any]:
    predicted = re.split(r"[,;\n]+", prediction)
    f1 = set_f1(predicted, answers)
    return {
        "score": float(f1 == 1.0),
        "metric": "set_match",
        "partial_f1": f1,
    }


def classify_error(prediction: str, answer: str, distractor_answers: list[str] | None = None) -> str:
    distractor_answers = distractor_answers or []
    if not prediction or not prediction.strip():
        return "refusal_or_empty"
    norm_pred = normalize_answer(prediction)
    if not norm_pred:
        return "format_error"
    if normalize_answer(answer) and normalize_answer(answer) in norm_pred:
        return "correct"
    for distractor in distractor_answers:
        if normalize_answer(distractor) and normalize_answer(distractor) in norm_pred:
            return "confused_with_distractor"
    if not re.search(r"[a-z0-9\u4e00-\u9fff]", norm_pred):
        return "format_error"
    return "hallucination"

