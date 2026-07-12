from __future__ import annotations

import random
from typing import Any

from lmaf.data.niah import generate_single_niah
from lmaf.eval.scorers import score_qa, score_topk
from lmaf.utils.token_count import TokenCounter, make_filler


RULER_TASKS = {
    "niah",
    "variable_tracking",
    "common_words_extraction",
    "freq_words_extraction",
    "qa_squad",
    "qa_hotpotqa",
}


def generate_ruler_fallback(
    task: str,
    length: int,
    seed: int = 42,
    sample_index: int = 0,
    counter: TokenCounter | None = None,
) -> dict[str, Any]:
    counter = counter or TokenCounter()
    if task == "niah":
        sample = generate_single_niah(length, 50, seed, sample_index, counter)
        sample.update(
            {
                "experiment": "ruler",
                "subtask": "niah",
                "sample_id": f"ruler_fallback_niah_{length}_seed{seed}_{sample_index:04d}",
                "implementation": "ruler_fallback",
            }
        )
        return sample
    if task == "variable_tracking":
        return _variable_tracking(length, seed, sample_index, counter)
    if task in {"common_words_extraction", "freq_words_extraction"}:
        return _word_frequency(task, length, seed, sample_index, counter)
    if task in {"qa_squad", "qa_hotpotqa"}:
        return _embedded_qa(task, length, seed, sample_index, counter)
    raise ValueError(f"Unsupported RULER fallback task: {task}")


def score_ruler_sample(sample: dict[str, Any], prediction: str) -> dict[str, Any]:
    task = sample.get("subtask")
    if task in {"common_words_extraction", "freq_words_extraction"}:
        return score_topk(prediction, sample.get("answer", []))
    return score_qa(prediction, sample.get("answer", ""))


def _variable_tracking(length: int, seed: int, sample_index: int, counter: TokenCounter) -> dict[str, Any]:
    rng = random.Random(seed + sample_index)
    final_value = f"value-{rng.randrange(1000, 9999)}"
    statements = [f"v0 is assigned {final_value}."]
    for i in range(1, 8):
        statements.append(f"v{i} is assigned the value of v{i - 1}.")
    question = "What is the final value of v7? Answer only the value."
    prompt = _wrap_to_length("\n".join(statements), question, length, seed, counter, forbidden=(final_value,))
    return {
        "experiment": "ruler",
        "subtask": "variable_tracking",
        "implementation": "ruler_fallback",
        "model": None,
        "sample_id": f"ruler_fallback_variable_tracking_{length}_seed{seed}_{sample_index:04d}",
        "length_tokens_target": length,
        "length_tokens_actual": counter.count(prompt),
        "prompt": prompt,
        "answer": final_value,
        "seed": seed,
        "error": None,
    }


def _word_frequency(task: str, length: int, seed: int, sample_index: int, counter: TokenCounter) -> dict[str, Any]:
    rng = random.Random(seed + sample_index)
    words = ["cedar", "marble", "violet", "amber", "quartz"]
    counts = [30, 22, 15, 9, 4]
    rng.shuffle(words)
    pairs = sorted(zip(words, counts), key=lambda item: item[1], reverse=True)
    body_words = []
    for word, count in pairs:
        body_words.extend([word] * count)
    rng.shuffle(body_words)
    if task == "common_words_extraction":
        answer = [pairs[0][0]]
        question = "Which word appears most frequently? Answer only the word."
    else:
        answer = [word for word, _ in pairs[:3]]
        question = "List the three most frequent words, separated by commas."
    prompt = _wrap_to_length(" ".join(body_words), question, length, seed, counter, forbidden=tuple(answer))
    return {
        "experiment": "ruler",
        "subtask": task,
        "implementation": "ruler_fallback",
        "model": None,
        "sample_id": f"ruler_fallback_{task}_{length}_seed{seed}_{sample_index:04d}",
        "length_tokens_target": length,
        "length_tokens_actual": counter.count(prompt),
        "prompt": prompt,
        "answer": answer,
        "seed": seed,
        "error": None,
    }


def _embedded_qa(task: str, length: int, seed: int, sample_index: int, counter: TokenCounter) -> dict[str, Any]:
    answer = f"checkpoint-{seed}-{sample_index}"
    fact = f"The expedition report states that the recovery phrase is {answer}."
    question = "What is the recovery phrase? Answer only the phrase."
    prompt = _wrap_to_length(fact, question, length, seed, counter, forbidden=(answer,))
    return {
        "experiment": "ruler",
        "subtask": task,
        "implementation": "ruler_fallback",
        "model": None,
        "sample_id": f"ruler_fallback_{task}_{length}_seed{seed}_{sample_index:04d}",
        "length_tokens_target": length,
        "length_tokens_actual": counter.count(prompt),
        "prompt": prompt,
        "answer": answer,
        "seed": seed,
        "error": None,
    }


def _wrap_to_length(
    core: str,
    question: str,
    length: int,
    seed: int,
    counter: TokenCounter,
    forbidden: tuple[str, ...] = (),
) -> str:
    prefix_budget = max(0, int(length * 0.35))
    prefix = make_filler(prefix_budget, seed=seed, counter=counter, forbidden=forbidden)
    used = counter.count(prefix) + counter.count(core) + counter.count(question) + 4
    suffix = make_filler(max(0, length - used), seed=seed + 33, counter=counter, forbidden=forbidden)
    return "\n\n".join(part for part in [prefix, core, suffix, question] if part)

