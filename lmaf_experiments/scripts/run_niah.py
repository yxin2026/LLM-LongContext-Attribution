from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lmaf.data.niah import generate_multi_niah, generate_sequential_niah, generate_single_niah
from lmaf.eval.metrics import best_contains
from lmaf.eval.scorers import score_niah
from lmaf.inference.client import create_inference_client
from lmaf.utils.io import append_jsonl, collect_jsonl, load_success_ids, utc_timestamp, write_jsonl
from lmaf.utils.token_count import TokenCounter


def main() -> None:
    args = parse_args()
    if args.generate_only:
        generate(args)
    else:
        run_inference(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate or run NIAH experiments.")
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--lengths", default="4096")
    parser.add_argument("--positions", default="50")
    parser.add_argument("--samples-per-cell", type=int, default=2)
    parser.add_argument("--variants", default="single", help="Comma list: single,multi,sequential")
    parser.add_argument("--distributions", default="uniform,clustered")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--provider", choices=["local", "siliconflow", "custom"], default="local")
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--thinking-budget", type=int, default=None)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def generate(args: argparse.Namespace) -> None:
    counter = TokenCounter(args.tokenizer)
    rows: list[dict[str, Any]] = []
    lengths = _parse_ints(args.lengths)
    positions = _parse_ints(args.positions)
    variants = {item.strip() for item in args.variants.split(",") if item.strip()}
    sample_index = 0
    if "single" in variants:
        for length in lengths:
            for position in positions:
                for i in range(args.samples_per_cell):
                    rows.append(generate_single_niah(length, position, args.seed, i, counter))
                    sample_index += 1
    if "multi" in variants:
        for length in lengths:
            for distribution in [x.strip() for x in args.distributions.split(",") if x.strip()]:
                for i in range(args.samples_per_cell):
                    rows.append(generate_multi_niah(length, distribution, args.seed, i, counter=counter))
                    sample_index += 1
    if "sequential" in variants:
        for length in lengths:
            for i in range(args.samples_per_cell):
                rows.append(generate_sequential_niah(length, 2, args.seed, i, counter))
                sample_index += 1

    out = _samples_output_path(args.output)
    write_jsonl(out, rows)
    print(f"Wrote {len(rows)} NIAH samples to {out}")


def run_inference(args: argparse.Namespace) -> None:
    if not args.input:
        raise SystemExit("--input is required unless --generate-only is set")
    if not args.model:
        raise SystemExit("--model is required for inference")

    samples = collect_jsonl(args.input)
    completed = load_success_ids(args.output) if args.resume else set()
    client = create_inference_client(
        provider=args.provider,
        model_name=args.model,
        endpoint=args.endpoint,
        api_key=args.api_key,
        timeout=args.timeout,
        enable_thinking=args.enable_thinking,
        thinking_budget=args.thinking_budget,
    )
    for sample in samples:
        sample_id = str(sample["sample_id"])
        if sample_id in completed:
            continue
        if args.max_model_len and int(sample.get("length_tokens_target") or 0) > args.max_model_len:
            row = dict(sample)
            row.update(
                {
                    "model": args.model,
                    "provider": args.provider,
                    "api_model": client.served_model_name,
                    "prediction": "",
                    "score": 0.0,
                    "metric": "skipped_by_model_length",
                    "latency_sec": 0.0,
                    "prompt_tokens": sample.get("length_tokens_actual"),
                    "completion_tokens": 0,
                    "timestamp": utc_timestamp(),
                    "error": "skipped_by_model_length",
                }
            )
            append_jsonl(args.output, row)
            continue
        result = client.generate(
            prompt=str(sample["prompt"]),
            request_id=sample_id,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        row = dict(sample)
        row.update(
            {
                "model": args.model,
                "provider": args.provider,
                "api_model": client.served_model_name,
                "prediction": result.response_text,
                "latency_sec": result.latency_sec,
                "prompt_tokens": result.prompt_tokens or sample.get("length_tokens_actual"),
                "completion_tokens": result.completion_tokens,
                "timestamp": utc_timestamp(),
                "error": result.error,
            }
        )
        if result.error:
            row.update({"score": 0.0, "metric": "request_error"})
        else:
            row.update(_score_answer(result.response_text, sample.get("answer")))
        append_jsonl(args.output, row)


def _score_answer(prediction: str, answer: Any) -> dict[str, Any]:
    if isinstance(answer, list):
        score = float(all(best_contains(prediction, item) for item in answer))
        return {"score": score, "metric": "all_contains", "contains_answer": int(score)}
    return score_niah(prediction, str(answer))


def _parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _samples_output_path(output: str) -> Path:
    out = Path(output)
    if out.suffix == ".jsonl":
        return out
    out.mkdir(parents=True, exist_ok=True)
    return out / "samples.jsonl"


if __name__ == "__main__":
    main()
