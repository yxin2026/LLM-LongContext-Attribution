from __future__ import annotations

import argparse
import json
import random
import string
from pathlib import Path


FILLER_SENTENCES = [
    "The audit paragraph lists routine batch numbers, locations, and irrelevant archive labels.",
    "A maintenance memo records shipment codes, sensor readings, and ordinary scheduling notes.",
    "The observation log contains project dates, staff initials, and non-target reference strings.",
    "A registry paragraph mentions calibration values and unrelated checksum-like identifiers.",
    "The field note includes inventory updates, weather records, and obsolete control numbers.",
    "A compliance note repeats procedural language and several distractor identifiers.",
]


def parse_csv_ints(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def code(rng: random.Random, prefix: str, n: int = 8) -> str:
    alphabet = string.ascii_uppercase + string.digits
    return f"{prefix}-" + "".join(rng.choice(alphabet) for _ in range(n))


def mutate_key(rng: random.Random, key: str) -> str:
    chars = list(key)
    idxs = [i for i, c in enumerate(chars) if c.isalnum()]
    idx = rng.choice(idxs)
    alphabet = string.ascii_uppercase + string.digits
    chars[idx] = rng.choice([c for c in alphabet if c != chars[idx]])
    return "".join(chars)


def make_record(key: str, answer: str, label: str) -> str:
    return (
        f"Registry record {label}: lookup_key={key}; checksum_value={answer}; "
        "status=active; use exact lookup_key matching only."
    )


def make_sample(length: int, position: int, idx: int, seed: int, distractors: int) -> dict:
    rng = random.Random(seed + length * 1009 + position * 97 + idx)
    sample_id = f"hard_niah_L{length}_P{position}_S{idx:04d}"
    target_key = code(rng, "K")
    answer = code(rng, "V", 10)
    target_record = make_record(target_key, answer, "TARGET")

    distractor_records = []
    for j in range(distractors):
        distractor_key = mutate_key(rng, target_key) if j < max(6, distractors // 4) else code(rng, "K")
        distractor_records.append(make_record(distractor_key, code(rng, "V", 10), f"DECOY-{j:03d}"))

    target_words = max(512, int(length * 0.74))
    target_start = int(target_words * position / 100)
    pre_decoys = distractor_records[: distractors // 2]
    post_decoys = distractor_records[distractors // 2 :]

    def filler_words(n_words: int, injected: list[str]) -> str:
        words: list[str] = []
        injections = injected[:]
        rng.shuffle(injections)
        next_injection_at = max(40, n_words // max(len(injections) + 1, 1))
        while len(words) < n_words:
            if injections and len(words) >= next_injection_at:
                words.extend(injections.pop().split())
                next_injection_at += max(35, n_words // max(len(injections) + 2, 1))
            else:
                words.extend(rng.choice(FILLER_SENTENCES).split())
        return " ".join(words[:n_words])

    before_words = max(0, target_start)
    after_words = max(0, target_words - before_words - len(target_record.split()))
    context = f"{filler_words(before_words, pre_decoys)} {target_record} {filler_words(after_words, post_decoys)}"
    question = f"What is the checksum_value for the exact lookup_key {target_key}?"
    prompt = (
        "You are given a long registry context with many similar decoy records. "
        "Answer using only the record whose lookup_key exactly matches the question. "
        "Return only the checksum_value and no other text.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    return {
        "sample_id": sample_id,
        "length": length,
        "position": position,
        "answer": answer,
        "prompt": prompt,
        "fact": target_record,
        "question": question,
        "distractors": distractors,
        "difficulty": "hard_decoy_exact_key",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", required=True, type=parse_csv_ints)
    parser.add_argument("--positions", required=True, type=parse_csv_ints)
    parser.add_argument("--samples-per-cell", required=True, type=int)
    parser.add_argument("--distractors", default=48, type=int)
    parser.add_argument("--seed", default=20260706, type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for length in args.lengths:
            for position in args.positions:
                for idx in range(args.samples_per_cell):
                    f.write(json.dumps(make_sample(length, position, idx, args.seed, args.distractors), ensure_ascii=False) + "\n")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()

