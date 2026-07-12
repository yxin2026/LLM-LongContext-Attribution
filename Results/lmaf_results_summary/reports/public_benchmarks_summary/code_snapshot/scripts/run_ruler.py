from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lmaf.data.ruler_adapter import RULER_TASKS, generate_ruler_fallback, score_ruler_sample
from lmaf.inference.client import create_inference_client, resolve_provider_model
from lmaf.utils.io import append_jsonl, collect_jsonl, load_success_ids, utc_timestamp, write_jsonl
from lmaf.utils.token_count import TokenCounter


def main() -> None:
    args = parse_args()
    if args.generate_only:
        generate(args)
    else:
        run_inference(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate or run RULER fallback tasks.")
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--lengths", default="4096,16384,32768")
    parser.add_argument("--tasks", default="niah,variable_tracking,common_words_extraction,freq_words_extraction,qa_squad,qa_hotpotqa")
    parser.add_argument("--samples-per-cell", type=int, default=2)
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
    tasks = _parse_strs(args.tasks)
    unknown = sorted(set(tasks) - RULER_TASKS)
    if unknown:
        raise SystemExit(f"Unsupported RULER fallback tasks: {', '.join(unknown)}")
    lengths = _parse_ints(args.lengths)
    if not tasks:
        raise SystemExit("--tasks must contain at least one RULER task")
    if not lengths:
        raise SystemExit("--lengths must contain at least one integer length")
    if any(length <= 0 for length in lengths):
        raise SystemExit("--lengths must be positive integers")
    if args.samples_per_cell <= 0:
        raise SystemExit("--samples-per-cell must be positive")
    rows = []
    for task, length, i in itertools.product(tasks, lengths, range(args.samples_per_cell)):
        rows.append(generate_ruler_fallback(task, length, args.seed, i, counter))
    out = _samples_output_path(args.output)
    write_jsonl(out, rows)
    print(f"Wrote {len(rows)} RULER fallback samples to {out}")


def run_inference(args: argparse.Namespace) -> None:
    if not args.input:
        raise SystemExit("--input is required unless --generate-only is set")
    if not args.model:
        raise SystemExit("--model is required for inference")

    samples = collect_jsonl(args.input)
    if not samples:
        raise SystemExit(f"No RULER samples found under {args.input}. Run with --generate-only first.")
    completed = load_success_ids(args.output) if args.resume else set()
    counter = TokenCounter(args.tokenizer)
    client = None
    for sample in samples:
        sample_id = str(sample["sample_id"])
        if sample_id in completed:
            continue
        prompt = str(sample["prompt"])
        prompt_tokens = counter.count(prompt)
        if args.max_model_len and prompt_tokens > args.max_model_len:
            append_jsonl(args.output, _skipped_row(sample, args, prompt_tokens, "skipped_overlength"))
            continue
        if client is None:
            client = create_inference_client(
                provider=args.provider,
                model_name=args.model,
                endpoint=args.endpoint,
                api_key=args.api_key,
                timeout=args.timeout,
                enable_thinking=args.enable_thinking,
                thinking_budget=args.thinking_budget,
            )
        result = client.generate(
            prompt=prompt,
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
                "prompt_tokens": result.prompt_tokens or prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "timestamp": utc_timestamp(),
                "error": result.error,
            }
        )
        if result.error:
            row.update({"score": 0.0, "metric": "request_error"})
        else:
            row.update(score_ruler_sample(sample, result.response_text))
        append_jsonl(args.output, row)


def _skipped_row(sample: dict, args: argparse.Namespace, prompt_tokens: int, reason: str) -> dict:
    row = dict(sample)
    row.update(
        {
            "model": args.model,
            "provider": args.provider,
            "api_model": resolve_provider_model(args.provider, args.model),
            "prediction": "",
            "latency_sec": 0.0,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": 0,
            "timestamp": utc_timestamp(),
            "score": 0.0,
            "metric": reason,
            "error": reason,
        }
    )
    return row


def _parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_strs(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _samples_output_path(output: str) -> Path:
    out = Path(output)
    if out.suffix == ".jsonl":
        return out
    out.mkdir(parents=True, exist_ok=True)
    return out / "samples.jsonl"


if __name__ == "__main__":
    main()
