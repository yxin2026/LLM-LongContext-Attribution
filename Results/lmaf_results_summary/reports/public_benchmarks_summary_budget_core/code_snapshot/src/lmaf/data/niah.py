from __future__ import annotations

import random
import uuid
from typing import Any

from lmaf.utils.token_count import TokenCounter, make_filler


ENTITY_NAMES = [
    "Aurora-Lab",
    "Borealis-Lab",
    "Cygnus-Lab",
    "Delta-Archive",
    "Equinox-Station",
    "Fjord-Vault",
    "Granite-Relay",
    "Helios-Node",
]


def generate_single_niah(
    length: int,
    position: int,
    seed: int = 42,
    sample_index: int = 0,
    counter: TokenCounter | None = None,
) -> dict[str, Any]:
    counter = counter or TokenCounter()
    rng = random.Random(seed + sample_index)
    entity = ENTITY_NAMES[(seed + sample_index) % len(ENTITY_NAMES)]
    code = _deterministic_uuid(rng)
    needle = f"The secret verification code for {entity} is {code}."
    question = f"What is the secret verification code for {entity}? Answer only the code."
    prompt = compose_insertions(
        length=length,
        insertions=[(position, needle)],
        question=question,
        seed=seed + sample_index,
        counter=counter,
        forbidden=(entity, code),
    )
    actual = counter.count(prompt)
    actual_pos = _position_percent(prompt, needle, counter)
    return {
        "experiment": "niah",
        "subtask": "single",
        "model": None,
        "sample_id": f"niah_single_{length}_pos{position}_seed{seed}_{sample_index:04d}",
        "length_tokens_target": length,
        "length_tokens_actual": actual,
        "position_percent": position,
        "position_percent_actual": actual_pos,
        "prompt": prompt,
        "answer": code,
        "entity": entity,
        "needle": needle,
        "seed": seed,
        "error": None,
    }


def generate_multi_niah(
    length: int,
    distribution: str,
    seed: int = 42,
    n_needles: int = 3,
    sample_index: int = 0,
    query_mode: str = "single",
    counter: TokenCounter | None = None,
) -> dict[str, Any]:
    counter = counter or TokenCounter()
    rng = random.Random(seed + sample_index)
    entities = ENTITY_NAMES[:n_needles]
    codes = [_deterministic_uuid(rng) for _ in range(n_needles)]
    needles = [f"The access key for {entity} is {code}." for entity, code in zip(entities, codes)]
    positions = _multi_positions(distribution, n_needles)
    if query_mode == "all":
        question = "List the access keys for Aurora-Lab, Borealis-Lab, and Cygnus-Lab in that order. Answer only the keys separated by commas."
        answer: str | list[str] = codes
    else:
        query_idx = sample_index % n_needles
        question = f"What is the access key for {entities[query_idx]}? Answer only the key."
        answer = codes[query_idx]
    prompt = compose_insertions(
        length=length,
        insertions=list(zip(positions, needles)),
        question=question,
        seed=seed + sample_index,
        counter=counter,
        forbidden=tuple(entities + codes),
    )
    return {
        "experiment": "niah",
        "subtask": "multi",
        "model": None,
        "sample_id": f"niah_multi_{distribution}_{length}_seed{seed}_{sample_index:04d}",
        "length_tokens_target": length,
        "length_tokens_actual": counter.count(prompt),
        "position_percent": None,
        "distribution": distribution,
        "prompt": prompt,
        "answer": answer,
        "entities": entities,
        "needles": needles,
        "seed": seed,
        "error": None,
    }


def generate_sequential_niah(
    length: int,
    hop: int = 2,
    seed: int = 42,
    sample_index: int = 0,
    counter: TokenCounter | None = None,
) -> dict[str, Any]:
    counter = counter or TokenCounter()
    rng = random.Random(seed + sample_index)
    entities = [f"Project-{chr(ord('A') + i)}-{sample_index}" for i in range(hop + 1)]
    code = _deterministic_uuid(rng)
    needles = [
        f"{entities[i]} forwards its archive to {entities[i + 1]}."
        for i in range(hop)
    ]
    needles.append(f"{entities[-1]} stores the final checkpoint code {code}.")
    positions = [20 + int(i * (60 / max(1, len(needles) - 1))) for i in range(len(needles))]
    question = f"{entities[0]} eventually points to which final checkpoint code? Answer only the code."
    prompt = compose_insertions(
        length=length,
        insertions=list(zip(positions, needles)),
        question=question,
        seed=seed + sample_index,
        counter=counter,
        forbidden=tuple(entities + [code]),
    )
    return {
        "experiment": "niah",
        "subtask": "sequential",
        "model": None,
        "sample_id": f"niah_sequential_{hop}hop_{length}_seed{seed}_{sample_index:04d}",
        "length_tokens_target": length,
        "length_tokens_actual": counter.count(prompt),
        "position_percent": None,
        "hop": hop,
        "prompt": prompt,
        "answer": code,
        "entities": entities,
        "needles": needles,
        "seed": seed,
        "error": None,
    }


def reconstruct_prompt(sample: dict[str, Any] | str) -> str:
    if isinstance(sample, str):
        return sample
    return str(sample.get("prompt", ""))


def validate_token_length(
    sample: dict[str, Any],
    counter: TokenCounter | None = None,
    length_tolerance: float = 0.05,
    position_tolerance: float = 2.0,
) -> dict[str, Any]:
    counter = counter or TokenCounter()
    prompt = reconstruct_prompt(sample)
    actual = counter.count(prompt)
    target = int(sample.get("length_tokens_target") or actual)
    result: dict[str, Any] = {
        "length_tokens_actual": actual,
        "length_error_ratio": abs(actual - target) / max(1, target),
        "length_ok": abs(actual - target) / max(1, target) <= length_tolerance,
    }
    needle = sample.get("needle")
    position = sample.get("position_percent")
    if needle and position is not None:
        actual_pos = _position_percent(prompt, str(needle), counter)
        result["position_percent_actual"] = actual_pos
        result["position_error_abs"] = abs(actual_pos - float(position))
        result["position_ok"] = abs(actual_pos - float(position)) <= position_tolerance
    return result


def compose_insertions(
    length: int,
    insertions: list[tuple[int | float, str]],
    question: str,
    seed: int,
    counter: TokenCounter,
    forbidden: tuple[str, ...] = (),
) -> str:
    """Compose a token-budgeted prompt with insertions at target percentages."""

    ordered = sorted(insertions, key=lambda item: item[0])
    parts: list[str] = []
    for idx, (percent, text) in enumerate(ordered):
        desired_start = int(length * float(percent) / 100)
        prefix = _join_parts(parts)
        if prefix:
            prefix += "\n\n"
        filler_budget = max(0, desired_start - counter.count(prefix))
        filler = make_filler(filler_budget, seed=seed + idx * 17, forbidden=forbidden, counter=counter)
        if filler:
            parts.append(filler)
        parts.append(text)

    without_tail = _join_parts(parts)
    if without_tail:
        without_tail += "\n\n"
    tail_budget = max(0, length - counter.count(without_tail) - counter.count(question) - 2)
    tail = make_filler(tail_budget, seed=seed + 101, forbidden=forbidden, counter=counter)
    if tail:
        parts.append(tail)
    parts.append(question)
    return _join_parts(parts)


def _deterministic_uuid(rng: random.Random) -> str:
    return str(uuid.UUID(int=rng.getrandbits(128)))


def _join_parts(parts: list[str]) -> str:
    return "\n\n".join(part for part in parts if part)


def _position_percent(prompt: str, needle: str, counter: TokenCounter) -> float:
    idx = prompt.index(needle)
    prefix_tokens = counter.count(prompt[:idx])
    total_tokens = max(1, counter.count(prompt))
    return round(prefix_tokens * 100 / total_tokens, 2)


def _multi_positions(distribution: str, n_needles: int) -> list[int]:
    if distribution == "clustered":
        base = 50 - 3 * (n_needles // 2)
        return [base + i * 3 for i in range(n_needles)]
    if n_needles == 1:
        return [50]
    return [20 + int(i * (60 / (n_needles - 1))) for i in range(n_needles)]

