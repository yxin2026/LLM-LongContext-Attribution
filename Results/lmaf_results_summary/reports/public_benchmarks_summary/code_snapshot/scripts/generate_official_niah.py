from __future__ import annotations

import argparse
import asyncio
import contextlib
import glob
import io
import os
import random
import sys
import uuid
import warnings
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lmaf.utils.io import write_jsonl
from lmaf.utils.token_count import TokenCounter


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


class OfflineNeedleHaystackModel:
    """Minimal model provider used only for the official generator."""

    model_name = "offline-official-needlehaystack-generator"

    def __init__(self, counter: TokenCounter) -> None:
        self.counter = counter

    async def evaluate_model(self, prompt: str) -> str:
        raise RuntimeError("This provider is for data generation only.")

    def generate_prompt(self, context: str, retrieval_question: str) -> str:
        return (
            "You are a helpful AI bot that answers questions for a user. "
            "Keep your response short and direct.\n\n"
            f"{context}\n\n"
            f"{retrieval_question} Don't give information outside the document or repeat your findings"
        )

    def encode_text_to_tokens(self, text: str) -> list[int]:
        return self.counter.encode(text)

    def decode_tokens(self, tokens: list[int], context_length: int | None = None) -> str:
        if context_length is not None:
            tokens = tokens[:context_length]
        return self.counter.decode(tokens)


def main() -> None:
    warnings.filterwarnings("ignore")
    args = parse_args()
    rows = generate(args)
    out = output_path(args.output)
    write_jsonl(out, rows)
    print(f"Wrote {len(rows)} official NIAH samples to {out}")
    print("source=Gregory Kamradt needlehaystack package")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate this project's three NIAH variants with the official "
            "Gregory Kamradt needlehaystack context/insertion generator."
        )
    )
    parser.add_argument("--output", default="data/processed/official/niah")
    parser.add_argument("--variants", default="single,multi,sequential")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--haystack-dir", default="PaulGrahamEssays")
    parser.add_argument("--final-context-length-buffer", type=int, default=200)

    parser.add_argument("--lengths-single", default="4096,32768,65536")
    parser.add_argument("--positions", default="10,50,90")
    parser.add_argument("--single-samples-per-cell", type=int, default=20)

    parser.add_argument("--lengths-multi", default="16384,32768")
    parser.add_argument("--distributions", default="uniform,clustered")
    parser.add_argument("--multi-samples-per-cell", type=int, default=20)
    parser.add_argument("--multi-needles", type=int, default=3)

    parser.add_argument("--lengths-sequential", default="16384,32768")
    parser.add_argument("--sequential-samples-per-cell", type=int, default=20)
    parser.add_argument("--sequential-hop", type=int, default=2)
    return parser.parse_args()


def generate(args: argparse.Namespace) -> list[dict[str, Any]]:
    require_official_package()
    counter = TokenCounter(args.tokenizer)
    model = OfflineNeedleHaystackModel(counter)
    rows: list[dict[str, Any]] = []
    variants = parse_csv(args.variants)
    valid_variants = {"single", "multi", "sequential"}
    unknown = set(variants) - valid_variants
    if unknown:
        raise SystemExit(f"Unknown NIAH variants: {', '.join(sorted(unknown))}")

    if "single" in variants:
        for length in parse_ints(args.lengths_single):
            for position in parse_ints(args.positions):
                for sample_index in range(args.single_samples_per_cell):
                    rows.append(generate_single(args, model, counter, length, position, sample_index))

    if "multi" in variants:
        distributions = parse_csv(args.distributions)
        valid_distributions = {"uniform", "clustered"}
        unknown_distributions = set(distributions) - valid_distributions
        if unknown_distributions:
            raise SystemExit(f"Unknown Multi-NIAH distributions: {', '.join(sorted(unknown_distributions))}")
        for length in parse_ints(args.lengths_multi):
            for distribution in distributions:
                for sample_index in range(args.multi_samples_per_cell):
                    rows.append(generate_multi(args, model, counter, length, distribution, sample_index))

    if "sequential" in variants:
        for length in parse_ints(args.lengths_sequential):
            for sample_index in range(args.sequential_samples_per_cell):
                rows.append(generate_sequential(args, model, counter, length, sample_index))

    return rows


def generate_single(
    args: argparse.Namespace,
    model: OfflineNeedleHaystackModel,
    counter: TokenCounter,
    length: int,
    position: int,
    sample_index: int,
) -> dict[str, Any]:
    rng = random.Random(args.seed + sample_index)
    entity = ENTITY_NAMES[(args.seed + sample_index) % len(ENTITY_NAMES)]
    code = deterministic_uuid(rng)
    needle = f"The secret verification code for {entity} is {code}."
    question = f"What is the secret verification code for {entity}? Answer only the code."
    context = official_single_context(args, model, length, position, needle, question)
    prompt = model.generate_prompt(context, question)
    return {
        "experiment": "niah",
        "subtask": "single",
        "source_schema": "official_needlehaystack",
        "official_generator": "gkamradt/LLMTest_NeedleInAHaystack needlehaystack package",
        "model": None,
        "sample_id": f"official_niah_single_{length}_pos{position}_seed{args.seed}_{sample_index:04d}",
        "length_tokens_target": length,
        "length_tokens_actual": counter.count(prompt),
        "position_percent": position,
        "position_percent_actual": position_percent(context, needle, counter),
        "prompt": prompt,
        "answer": code,
        "entity": entity,
        "needle": needle,
        "seed": args.seed,
        "error": None,
    }


def generate_multi(
    args: argparse.Namespace,
    model: OfflineNeedleHaystackModel,
    counter: TokenCounter,
    length: int,
    distribution: str,
    sample_index: int,
) -> dict[str, Any]:
    rng = random.Random(args.seed + 10_000 + sample_index)
    n_needles = args.multi_needles
    entities = ENTITY_NAMES[:n_needles]
    codes = [deterministic_uuid(rng) for _ in range(n_needles)]
    needles = [f"The access key for {entity} is {code}." for entity, code in zip(entities, codes)]
    query_idx = sample_index % n_needles
    question = f"What is the access key for {entities[query_idx]}? Answer only the key."
    if distribution == "uniform":
        context = official_multi_context(args, model, length, depth_percent=20, needles=needles, question=question)
        actual_positions = [round(value, 2) for value in getattr(official_multi_context.last_tester, "insertion_percentages", [])]
    else:
        positions = clustered_positions(n_needles)
        context = official_context_with_insertions(args, model, length, list(zip(positions, needles)), question)
        actual_positions = [position_percent(context, needle, counter) for needle in needles]
    prompt = model.generate_prompt(context, question)
    return {
        "experiment": "niah",
        "subtask": "multi",
        "source_schema": "official_needlehaystack",
        "official_generator": "gkamradt/LLMTest_NeedleInAHaystack needlehaystack package",
        "model": None,
        "sample_id": f"official_niah_multi_{distribution}_{length}_seed{args.seed}_{sample_index:04d}",
        "length_tokens_target": length,
        "length_tokens_actual": counter.count(prompt),
        "position_percent": None,
        "position_percent_actual": actual_positions,
        "distribution": distribution,
        "prompt": prompt,
        "answer": codes[query_idx],
        "entities": entities,
        "needles": needles,
        "seed": args.seed,
        "error": None,
    }


def generate_sequential(
    args: argparse.Namespace,
    model: OfflineNeedleHaystackModel,
    counter: TokenCounter,
    length: int,
    sample_index: int,
) -> dict[str, Any]:
    rng = random.Random(args.seed + 20_000 + sample_index)
    hop = args.sequential_hop
    entities = [f"Project-{chr(ord('A') + i)}-{sample_index}" for i in range(hop + 1)]
    code = deterministic_uuid(rng)
    needles = [f"{entities[i]} forwards its archive to {entities[i + 1]}." for i in range(hop)]
    needles.append(f"{entities[-1]} stores the final checkpoint code {code}.")
    positions = sequential_positions(len(needles))
    question = f"{entities[0]} eventually points to which final checkpoint code? Answer only the code."
    context = official_context_with_insertions(args, model, length, list(zip(positions, needles)), question)
    prompt = model.generate_prompt(context, question)
    return {
        "experiment": "niah",
        "subtask": "sequential",
        "source_schema": "official_needlehaystack_plus_project_chain",
        "official_generator": "gkamradt/LLMTest_NeedleInAHaystack context/insertion generator",
        "model": None,
        "sample_id": f"official_niah_sequential_{hop}hop_{length}_seed{args.seed}_{sample_index:04d}",
        "length_tokens_target": length,
        "length_tokens_actual": counter.count(prompt),
        "position_percent": None,
        "position_percent_actual": [position_percent(context, needle, counter) for needle in needles],
        "hop": hop,
        "prompt": prompt,
        "answer": code,
        "entities": entities,
        "needles": needles,
        "seed": args.seed,
        "error": None,
    }


def official_single_context(
    args: argparse.Namespace,
    model: OfflineNeedleHaystackModel,
    length: int,
    depth_percent: int | float,
    needle: str,
    question: str,
) -> str:
    with contextlib.redirect_stderr(io.StringIO()):
        from needlehaystack.llm_needle_haystack_tester import LLMNeedleHaystackTester

    class WindowsSafeNeedleHaystackTester(LLMNeedleHaystackTester):
        read_context_files = read_context_files_utf8

    tester = WindowsSafeNeedleHaystackTester(
        model_to_test=model,
        needle=needle,
        haystack_dir=args.haystack_dir,
        retrieval_question=question,
        context_lengths=[length],
        document_depth_percents=[depth_percent],
        save_results=False,
        save_contexts=False,
        final_context_length_buffer=args.final_context_length_buffer,
        print_ongoing_status=False,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        return asyncio.run(tester.generate_context(length, depth_percent))


def official_multi_context(
    args: argparse.Namespace,
    model: OfflineNeedleHaystackModel,
    length: int,
    depth_percent: int | float,
    needles: list[str],
    question: str,
) -> str:
    with contextlib.redirect_stderr(io.StringIO()):
        from needlehaystack.llm_multi_needle_haystack_tester import LLMMultiNeedleHaystackTester

    class WindowsSafeMultiNeedleHaystackTester(LLMMultiNeedleHaystackTester):
        read_context_files = read_context_files_utf8

    tester = WindowsSafeMultiNeedleHaystackTester(
        model_to_test=model,
        needles=needles,
        needle=needles[0],
        haystack_dir=args.haystack_dir,
        retrieval_question=question,
        context_lengths=[length],
        document_depth_percents=[depth_percent],
        save_results=False,
        save_contexts=False,
        final_context_length_buffer=args.final_context_length_buffer,
        print_ongoing_status=False,
    )
    official_multi_context.last_tester = tester
    with contextlib.redirect_stdout(io.StringIO()):
        return asyncio.run(tester.generate_context(length, depth_percent))


official_multi_context.last_tester = None  # type: ignore[attr-defined]


def official_context_with_insertions(
    args: argparse.Namespace,
    model: OfflineNeedleHaystackModel,
    length: int,
    insertions: list[tuple[int | float, str]],
    question: str,
) -> str:
    with contextlib.redirect_stderr(io.StringIO()):
        from needlehaystack.llm_needle_haystack_tester import LLMNeedleHaystackTester

    class WindowsSafeNeedleHaystackTester(LLMNeedleHaystackTester):
        read_context_files = read_context_files_utf8

    first_needle = insertions[0][1]
    tester = WindowsSafeNeedleHaystackTester(
        model_to_test=model,
        needle=first_needle,
        haystack_dir=args.haystack_dir,
        retrieval_question=question,
        context_lengths=[length],
        document_depth_percents=[insertions[0][0]],
        save_results=False,
        save_contexts=False,
        final_context_length_buffer=args.final_context_length_buffer,
        print_ongoing_status=False,
    )
    context = tester.read_context_files()
    context = tester.encode_and_trim(context, length)
    for depth_percent, needle in sorted(insertions, key=lambda item: item[0]):
        tester.needle = needle
        context = tester.insert_needle(context, depth_percent, length)
    return context


def read_context_files_utf8(self) -> str:
    with contextlib.redirect_stderr(io.StringIO()):
        import needlehaystack.llm_needle_haystack_tester as official_module

    context = ""
    max_context_length = max(self.context_lengths)
    base_dir = os.path.abspath(os.path.dirname(official_module.__file__))
    while self.get_context_length_in_tokens(context) < max_context_length:
        for file in glob.glob(os.path.join(base_dir, self.haystack_dir, "*.txt")):
            with open(file, "r", encoding="utf-8", errors="ignore") as f:
                context += f.read()
    return context


def require_official_package() -> None:
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            import needlehaystack  # noqa: F401
    except Exception as exc:
        raise SystemExit(
            "Official NIAH generation requires the `needlehaystack` package. "
            "Install it in the same environment you use for experiments."
        ) from exc


def deterministic_uuid(rng: random.Random) -> str:
    return str(uuid.UUID(int=rng.getrandbits(128)))


def position_percent(text: str, needle: str, counter: TokenCounter) -> float:
    idx = text.index(needle)
    prefix_tokens = counter.count(text[:idx])
    total_tokens = max(1, counter.count(text))
    return round(prefix_tokens * 100 / total_tokens, 2)


def clustered_positions(n_needles: int) -> list[int]:
    base = 50 - 3 * (n_needles // 2)
    return [base + i * 3 for i in range(n_needles)]


def sequential_positions(count: int) -> list[int]:
    if count == 1:
        return [50]
    return [20 + int(i * (60 / (count - 1))) for i in range(count)]


def output_path(output: str) -> Path:
    out = ROOT / output
    if out.suffix == ".jsonl":
        return out
    out.mkdir(parents=True, exist_ok=True)
    return out / "samples.jsonl"


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    main()
