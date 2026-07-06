from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


FILLER_SENTENCES = [
    "The archive note discusses routine calibration logs and contains no answer.",
    "A committee memo records ordinary shipment labels and irrelevant reference codes.",
    "The field report mentions weather, inventory, and maintenance details only.",
    "A planning paragraph lists names, dates, and locations unrelated to the query.",
    "The project ledger summarizes standard observations without the target fact.",
]


def make_fact(rng: random.Random, sample_id: str) -> tuple[str, str]:
    key = f"APBS-{rng.randint(100000, 999999)}"
    answer = f"token-{rng.randint(10000000, 99999999)}"
    fact = f"Important fact for {sample_id}: the retrieval key {key} has secret value {answer}."
    question = f"What is the secret value for retrieval key {key}?"
    return fact, answer, question


def approximate_context(length_tokens: int, position_pct: int, fact: str, rng: random.Random) -> str:
    target_words = max(256, int(length_tokens * 0.72))
    fact_words = len(fact.split())
    fact_start = int(target_words * position_pct / 100)
    before_words = max(0, fact_start)
    after_words = max(0, target_words - before_words - fact_words)

    def filler(n_words: int) -> str:
        words = []
        while len(words) < n_words:
            sent = rng.choice(FILLER_SENTENCES)
            words.extend(sent.split())
        return " ".join(words[:n_words])

    return f"{filler(before_words)} {fact} {filler(after_words)}"


def build_sample(length: int, position: int, idx: int, seed: int) -> dict:
    rng = random.Random(seed + length * 1000 + position * 100 + idx)
    sample_id = f"niah_L{length}_P{position}_S{idx:04d}"
    fact, answer, question = make_fact(rng, sample_id)
    context = approximate_context(length, position, fact, rng)
    prompt = (
        "You are given a long context. Answer the question using only the context. "
        "Return only the secret value, with no explanation.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    return {
        "sample_id": sample_id,
        "length": length,
        "position": position,
        "answer": answer,
        "prompt": prompt,
        "fact": fact,
        "question": question,
    }


def parse_csv_ints(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", required=True, type=parse_csv_ints)
    parser.add_argument("--positions", required=True, type=parse_csv_ints)
    parser.add_argument("--samples-per-cell", required=True, type=int)
    parser.add_argument("--seed", default=20260705, type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for length in args.lengths:
            for position in args.positions:
                for idx in range(args.samples_per_cell):
                    row = build_sample(length, position, idx, args.seed)
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {output}")


if __name__ == "__main__":
    main()

