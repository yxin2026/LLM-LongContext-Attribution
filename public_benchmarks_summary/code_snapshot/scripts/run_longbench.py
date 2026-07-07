from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lmaf.data.longbench import longbench_metric_kind, prepare_longbench
from lmaf.eval.scorers import score_qa, score_summarization
from lmaf.inference.client import create_inference_client, resolve_provider_model
from lmaf.utils.io import append_jsonl, collect_jsonl, iter_jsonl_paths, load_success_ids, utc_timestamp
from lmaf.utils.token_count import TokenCounter


def main() -> None:
    args = parse_args()
    tasks = _parse_strs(args.tasks)
    if not tasks:
        raise SystemExit("--tasks must contain at least one LongBench task")
    if args.sample_limit is not None and args.sample_limit <= 0:
        raise SystemExit("--sample-limit must be positive when provided")
    if args.prepare_only:
        written = prepare_longbench(
            tasks=tasks,
            output_dir=args.output,
            repo_dir=args.longbench_repo,
            sample_limit=args.sample_limit,
            tokenizer_name=args.tokenizer,
        )
        print("Prepared LongBench files:")
        for path in written:
            print(path)
        return
    run_inference(args, tasks)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare or run LongBench.")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--tasks", default="narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,gov_report,qmsum,multi_news")
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--longbench-repo", default="external/LongBench")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--input", default="data/processed/longbench")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--provider", choices=["local", "siliconflow", "custom"], default="local")
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--thinking-budget", type=int, default=None)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--truncate", choices=["none", "middle"], default="none")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def run_inference(args: argparse.Namespace, tasks: list[str]) -> None:
    if not args.model:
        raise SystemExit("--model is required for inference")
    samples = _load_task_samples(args.input, set(tasks))
    if not samples:
        raise SystemExit(f"No LongBench samples found for selected tasks under {args.input}. Run with --prepare-only first.")
    completed = load_success_ids(args.output) if args.resume else set()
    counter = TokenCounter(args.tokenizer)
    client = None
    for sample in samples:
        sample_id = str(sample["sample_id"])
        if sample_id in completed:
            continue
        prompt = str(sample["prompt"])
        truncation = "none"
        if args.max_model_len and counter.count(prompt) > args.max_model_len:
            if args.truncate == "middle":
                prompt = _middle_truncate(prompt, args.max_model_len, counter)
                truncation = "middle"
            else:
                row = _skipped_row(sample, args, counter.count(prompt), "skipped_overlength")
                append_jsonl(args.output, row)
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
                "prompt": prompt,
                "prediction": result.response_text,
                "latency_sec": result.latency_sec,
                "prompt_tokens": result.prompt_tokens or counter.count(prompt),
                "completion_tokens": result.completion_tokens,
                "timestamp": utc_timestamp(),
                "truncation": truncation,
                "error": result.error,
            }
        )
        if result.error:
            row.update({"score": 0.0, "metric": "request_error"})
        elif longbench_metric_kind(str(sample.get("task"))) == "summarization":
            row.update(score_summarization(result.response_text, sample.get("answers", [])))
        else:
            row.update(score_qa(result.response_text, sample.get("answers", [])))
        append_jsonl(args.output, row)


def _load_task_samples(input_path: str, tasks: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in iter_jsonl_paths(input_path):
        if path.stem not in tasks:
            continue
        rows.extend(collect_jsonl(path))
    return rows


def _skipped_row(sample: dict[str, Any], args: argparse.Namespace, prompt_tokens: int, reason: str) -> dict[str, Any]:
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
            "score": 0.0,
            "metric": reason,
            "timestamp": utc_timestamp(),
            "truncation": "none",
            "error": reason,
        }
    )
    return row


def _middle_truncate(prompt: str, max_tokens: int, counter: TokenCounter) -> str:
    marker = "\n\n[... middle truncated ...]\n\n"
    marker_tokens = counter.count(marker)
    keep = max(1, max_tokens - marker_tokens)
    left = keep // 2
    right = keep - left
    tokens = counter.encode(prompt)
    return counter.decode(tokens[:left]) + marker + counter.decode(tokens[-right:])


def _parse_strs(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    main()
