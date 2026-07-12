from __future__ import annotations

import random
import string
import uuid
from typing import Any

from lmaf.data.niah import ENTITY_NAMES, compose_insertions, generate_single_niah
from lmaf.eval.scorers import classify_error
from lmaf.utils.token_count import TokenCounter, make_filler


PAC_EXTERNAL_SUBTASKS = {
    "A": "A_position",
    "B": "B_interference",
    "C": "C_overlap",
    "D": "D_multihop",
}

PAC_SUBTASK_TO_EXTERNAL = {value: key for key, value in PAC_EXTERNAL_SUBTASKS.items()}

PAC_EXTERNAL_INTERFERENCE_TYPES = {
    "in_domain_related": "in_domain",
    "out_domain_unrelated": "out_domain",
    "random_noise": "random_noise",
}


def generate_pac_a_position(
    length: int,
    position: int,
    seed: int = 42,
    sample_index: int = 0,
    counter: TokenCounter | None = None,
) -> dict[str, Any]:
    sample = generate_single_niah(length, position, seed, sample_index, counter)
    sample.update(
        {
            "experiment": "pac",
            "subtask": "A_position",
            "sample_id": f"pac_A_position_{length}_pos{position}_seed{seed}_{sample_index:04d}",
            "density": 0,
            "interference_type": None,
        }
    )
    return sample


def generate_pac_b_interference(
    length: int,
    position: int,
    density: int,
    interference_type: str,
    seed: int = 42,
    sample_index: int = 0,
    counter: TokenCounter | None = None,
) -> dict[str, Any]:
    counter = counter or TokenCounter()
    rng = random.Random(seed + sample_index + density)
    entity = f"Aurora-Lab-{sample_index}"
    code = _code(rng)
    target = f"The access key for {entity} is {code}."
    distractors = _interference_block(interference_type, density, rng, sample_index)
    question = f"What is the access key for {entity}? Answer only the key."
    insertions: list[tuple[int, str]] = [(position, target)]
    if distractors:
        insertions.append((min(95, position + 8), distractors))
    prompt = compose_insertions(
        length=length,
        insertions=insertions,
        question=question,
        seed=seed + sample_index,
        counter=counter,
        forbidden=(entity, code),
    )
    return {
        "experiment": "pac",
        "subtask": "B_interference",
        "model": None,
        "sample_id": f"pac_B_interference_{interference_type}_{density}_{length}_seed{seed}_{sample_index:04d}",
        "length_tokens_target": length,
        "length_tokens_actual": counter.count(prompt),
        "position_percent": position,
        "density": density,
        "interference_type": interference_type,
        "prompt": prompt,
        "answer": code,
        "distractor_answers": _extract_codes(distractors),
        "seed": seed,
        "error": None,
    }


def generate_pac_c_overlap(
    length: int,
    similarity: str,
    distance: str,
    seed: int = 42,
    sample_index: int = 0,
    counter: TokenCounter | None = None,
) -> dict[str, Any]:
    counter = counter or TokenCounter()
    rng = random.Random(seed + sample_index)
    target_entity, distractor_entity = _overlap_entities(similarity, sample_index)
    target_code = _code(rng)
    distractor_code = _code(rng)
    target = f"The access key for {target_entity} is {target_code}."
    distractor = f"The access key for {distractor_entity} is {distractor_code}."
    target_position = 50
    distractor_position = _distance_position(target_position, distance)
    question = f"What is the access key for {target_entity}? Answer only the key."
    prompt = compose_insertions(
        length=length,
        insertions=[(target_position, target), (distractor_position, distractor)],
        question=question,
        seed=seed + sample_index,
        counter=counter,
        forbidden=(target_entity, distractor_entity, target_code, distractor_code),
    )
    return {
        "experiment": "pac",
        "subtask": "C_overlap",
        "model": None,
        "sample_id": f"pac_C_overlap_{similarity}_{distance}_{length}_seed{seed}_{sample_index:04d}",
        "length_tokens_target": length,
        "length_tokens_actual": counter.count(prompt),
        "position_percent": target_position,
        "similarity": similarity,
        "distance": distance,
        "prompt": prompt,
        "answer": target_code,
        "distractor_answers": [distractor_code],
        "target_entity": target_entity,
        "distractor_entity": distractor_entity,
        "seed": seed,
        "error": None,
    }


def generate_pac_d_multihop(
    length: int,
    hops: int,
    hop_distance: int,
    seed: int = 42,
    sample_index: int = 0,
    counter: TokenCounter | None = None,
) -> dict[str, Any]:
    counter = counter or TokenCounter()
    rng = random.Random(seed + sample_index + hops)
    entities = [f"Entity-{chr(ord('A') + i)}-{sample_index}" for i in range(hops + 1)]
    code = _code(rng)
    facts = [
        f"{entities[i]} transfers the token to {entities[i + 1]}."
        for i in range(hops)
    ]
    facts.append(f"{entities[-1]} stores final code {code}.")
    start_percent = 10
    spacing_percent = max(2, int(hop_distance * 100 / max(1, length)))
    positions = [min(90, start_percent + i * spacing_percent) for i in range(len(facts))]
    question = f"Starting from {entities[0]}, what is the final code? Answer only the code."
    prompt = compose_insertions(
        length=length,
        insertions=list(zip(positions, facts)),
        question=question,
        seed=seed + sample_index,
        counter=counter,
        forbidden=tuple(entities + [code]),
    )
    intermediate_questions = [
        f"Starting from {entities[i]}, which entity is reached after one transfer?"
        for i in range(hops)
    ]
    return {
        "experiment": "pac",
        "subtask": "D_multihop",
        "model": None,
        "sample_id": f"pac_D_multihop_{hops}hop_{hop_distance}_{length}_seed{seed}_{sample_index:04d}",
        "length_tokens_target": length,
        "length_tokens_actual": counter.count(prompt),
        "position_percent": start_percent,
        "hops": hops,
        "hop_distance": hop_distance,
        "prompt": prompt,
        "answer": code,
        "entities": entities,
        "intermediate_answers": entities[1:],
        "intermediate_questions": intermediate_questions,
        "seed": seed,
        "error": None,
    }


def score_pac_sample(sample: dict[str, Any], prediction: str) -> dict[str, Any]:
    label = classify_error(prediction, str(sample.get("answer", "")), sample.get("distractor_answers") or [])
    return {
        "score": float(label == "correct"),
        "metric": "pac_exact_or_error_class",
        "error_type": label,
    }


def adapt_external_pac_sample(
    row: dict[str, Any],
    counter: TokenCounter | None = None,
    count_tokens: bool = False,
) -> dict[str, Any]:
    """Convert the standalone PAC-Test-Dataset JSONL schema into this project's schema."""

    subset = str(row.get("subset") or "").strip()
    subtask = PAC_EXTERNAL_SUBTASKS.get(subset)
    if not subtask:
        raise ValueError(f"Unsupported external PAC subset: {subset!r}")

    context = str(row.get("context") or "")
    question = str(row.get("question") or "")
    prompt = build_external_pac_prompt(context, question)
    adapted = dict(row)
    adapted.update(
        {
            "experiment": "pac",
            "subtask": subtask,
            "model": row.get("model"),
            "prompt": prompt,
            "length_tokens_target": row.get("total_length"),
            "length_tokens_actual": _external_length(row, prompt, counter, count_tokens),
            "source_schema": "pac_test_dataset_v3",
            "error": row.get("error"),
        }
    )

    if subtask == "A_position":
        position = row.get("position_ratio")
        if position is not None:
            adapted["position_percent"] = _ratio_to_percent(position)
    elif subtask == "B_interference":
        density = row.get("noise_density")
        if density is not None:
            adapted["density"] = _ratio_to_percent(density)
        kind = str(row.get("dilution_type") or "")
        adapted["interference_type"] = PAC_EXTERNAL_INTERFERENCE_TYPES.get(kind, kind or None)
        adapted["interference_type_raw"] = kind or None
        adapted["position_percent"] = 0
    elif subtask == "C_overlap":
        adapted["similarity"] = row.get("similarity_level")
        adapted["distance"] = row.get("distance_level")
    elif subtask == "D_multihop":
        adapted["hops"] = row.get("num_hops")
        adapted["hop_distance"] = row.get("distance_level")
        adapted["distance"] = row.get("distance_level")
        if "fact_chain" in row and isinstance(row["fact_chain"], list):
            adapted["intermediate_answers"] = [item.get("target") for item in row["fact_chain"][:-1] if isinstance(item, dict)]

    return adapted


def _build_external_pac_prompt_legacy(context: str, question: str) -> str:
    return (
        "请根据以下长上下文回答问题。只输出最终答案，不要解释。\n\n"
        f"[长上下文]\n{context}\n\n"
        f"[问题]\n{question}\n\n"
        "[答案]"
    )


def build_external_pac_prompt(context: str, question: str) -> str:
    return (
        "Answer the question using only the following long context. "
        "Output only the final answer, with no explanation.\n\n"
        f"[Context]\n{context}\n\n"
        f"[Question]\n{question}\n\n"
        "[Answer]"
    )


def is_external_pac_sample(row: dict[str, Any]) -> bool:
    return "prompt" not in row and row.get("subset") in PAC_EXTERNAL_SUBTASKS and "context" in row and "question" in row


def _external_length(
    row: dict[str, Any],
    prompt: str,
    counter: TokenCounter | None,
    count_tokens: bool,
) -> int | Any:
    if count_tokens:
        counter = counter or TokenCounter()
        return counter.count(prompt)
    return row.get("total_length")


def _ratio_to_percent(value: Any) -> float | int:
    number = float(value)
    percent = number * 100 if number <= 1 else number
    return int(percent) if percent.is_integer() else percent


def _code(rng: random.Random) -> str:
    return str(uuid.UUID(int=rng.getrandbits(128)))


def _interference_block(kind: str, density: int, rng: random.Random, sample_index: int) -> str:
    if density <= 0:
        return ""
    n_items = max(1, density // 5)
    if kind == "in_domain":
        return "\n".join(
            f"The access key for Decoy-Lab-{sample_index}-{i} is {_code(rng)}."
            for i in range(n_items)
        )
    if kind == "random_noise":
        alphabet = string.ascii_lowercase + string.digits
        return " ".join(
            "".join(rng.choice(alphabet) for _ in range(12))
            for _ in range(n_items * 4)
        )
    return make_filler(n_items * 24, seed=sample_index + density, counter=TokenCounter())


def _extract_codes(text: str) -> list[str]:
    parts = []
    for token in text.replace("\n", " ").split():
        token = token.strip(".,;:")
        if token.count("-") >= 4:
            parts.append(token)
    return parts


def _overlap_entities(similarity: str, sample_index: int) -> tuple[str, str]:
    if similarity == "high":
        return f"Aurora-Lab-{sample_index:02d}", f"Aurora-Lab-{sample_index + 1:02d}"
    if similarity == "medium":
        return f"Aurora-Lab-{sample_index}", f"Borealis-Lab-{sample_index}"
    if similarity == "low":
        return f"Aurora-Lab-{sample_index}", f"Harbor-City-{sample_index}"
    return f"Aurora-Lab-{sample_index}", f"Unrelated-Depot-{sample_index}"


def _distance_position(target_position: int, distance: str) -> int:
    if distance == "near":
        return min(95, target_position + 3)
    if distance == "medium":
        return min(95, target_position + 18)
    return min(95, target_position + 35)
