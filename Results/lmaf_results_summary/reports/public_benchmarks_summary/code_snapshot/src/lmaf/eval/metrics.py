from __future__ import annotations

import math
import re
import string
import unicodedata
from collections import Counter
from typing import Iterable


def normalize_answer(text: str) -> str:
    text = (text or "").lower().strip()
    kept: list[str] = []
    for ch in unicodedata.normalize("NFKC", text):
        category = unicodedata.category(ch)
        if category.startswith("P") or category.startswith("Z"):
            continue
        if ch in string.whitespace:
            continue
        kept.append(ch)
    return "".join(kept)


def contains_answer(prediction: str, answer: str) -> bool:
    norm_pred = normalize_answer(prediction)
    norm_answer = normalize_answer(answer)
    return bool(norm_answer) and norm_answer in norm_pred


def exact_match(prediction: str, answers: str | Iterable[str]) -> float:
    if isinstance(answers, str):
        answers = [answers]
    norm_pred = normalize_answer(prediction)
    return float(any(norm_pred == normalize_answer(ans) for ans in answers))


def best_contains(prediction: str, answers: str | Iterable[str]) -> float:
    if isinstance(answers, str):
        answers = [answers]
    return float(any(contains_answer(prediction, ans) for ans in answers))


def token_f1(prediction: str, answers: str | Iterable[str]) -> float:
    if isinstance(answers, str):
        answers = [answers]
    return max((_token_f1_single(prediction, ans) for ans in answers), default=0.0)


def rouge_l(prediction: str, answers: str | Iterable[str]) -> float:
    if isinstance(answers, str):
        answers = [answers]
    return max((_rouge_l_single(prediction, ans) for ans in answers), default=0.0)


def set_f1(predicted: Iterable[str], expected: Iterable[str]) -> float:
    pred_set = {normalize_answer(x) for x in predicted if normalize_answer(x)}
    exp_set = {normalize_answer(x) for x in expected if normalize_answer(x)}
    if not pred_set and not exp_set:
        return 1.0
    if not pred_set or not exp_set:
        return 0.0
    overlap = len(pred_set & exp_set)
    precision = overlap / len(pred_set)
    recall = overlap / len(exp_set)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return math.nan
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x == 0 or den_y == 0:
        return math.nan
    return num / (den_x * den_y)


def linear_slope(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return math.nan
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    den = sum((x - mean_x) ** 2 for x in xs)
    if den == 0:
        return math.nan
    return sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / den


def _tokenize_for_f1(text: str) -> list[str]:
    stripped = text.strip()
    if re.search(r"\s", stripped):
        return [normalize_answer(t) for t in stripped.split() if normalize_answer(t)]
    normalized = normalize_answer(stripped)
    return list(normalized)


def _token_f1_single(prediction: str, answer: str) -> float:
    pred_tokens = _tokenize_for_f1(prediction)
    answer_tokens = _tokenize_for_f1(answer)
    if not pred_tokens and not answer_tokens:
        return 1.0
    if not pred_tokens or not answer_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(answer_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(answer_tokens)
    return 2 * precision * recall / (precision + recall)


def _rouge_l_single(prediction: str, answer: str) -> float:
    pred_tokens = _tokenize_for_f1(prediction)
    answer_tokens = _tokenize_for_f1(answer)
    if not pred_tokens or not answer_tokens:
        return 0.0
    lcs = _lcs_length(pred_tokens, answer_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(answer_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _lcs_length(a: list[str], b: list[str]) -> int:
    prev = [0] * (len(b) + 1)
    for item_a in a:
        cur = [0]
        for j, item_b in enumerate(b, start=1):
            if item_a == item_b:
                cur.append(prev[j - 1] + 1)
            else:
                cur.append(max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]

