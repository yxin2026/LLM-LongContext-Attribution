from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lmaf.data.pac import (
    generate_pac_a_position,
    generate_pac_b_interference,
    generate_pac_c_overlap,
    generate_pac_d_multihop,
    score_pac_sample,
)
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
    parser = argparse.ArgumentParser(description="Generate or run PAC-Test experiments.")
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--subset", required=True, choices=["A_position", "B_interference", "C_overlap", "D_multihop"])
    parser.add_argument("--length", type=int, default=16384)
    parser.add_argument("--lengths", default=None)
    parser.add_argument("--position", type=int, default=10)
    parser.add_argument("--positions", default="10,25,50,75,90")
    parser.add_argument("--densities", default="0,25,50,75,90")
    parser.add_argument("--interference-types", default="in_domain,out_domain,random_noise")
    parser.add_argument("--similarities", default="high,medium,low,none")
    parser.add_argument("--distances", default="near,medium,far")
    parser.add_argument("--hops", default="2,3,4")
    parser.add_argument("--hop-distances", default="1024,4096,8192")
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
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def generate(args: argparse.Namespace) -> None:
    counter = TokenCounter(args.tokenizer)
    rows = []
    if args.subset == "A_position":
        for position, i in itertools.product(_parse_ints(args.positions), range(args.samples_per_cell)):
            rows.append(generate_pac_a_position(args.length, position, args.seed, i, counter))
    elif args.subset == "B_interference":
        for density, kind, i in itertools.product(
            _parse_ints(args.densities),
            _parse_strs(args.interference_types),
            range(args.samples_per_cell),
        ):
            rows.append(generate_pac_b_interference(args.length, args.position, density, kind, args.seed, i, counter))
    elif args.subset == "C_overlap":
        for similarity, distance, i in itertools.product(
            _parse_strs(args.similarities),
            _parse_strs(args.distances),
            range(args.samples_per_cell),
        ):
            rows.append(generate_pac_c_overlap(args.length, similarity, distance, args.seed, i, counter))
    elif args.subset == "D_multihop":
        lengths = _parse_ints(args.lengths) if args.lengths else [args.length]
        for length, hops, hop_distance, i in itertools.product(
            lengths,
            _parse_ints(args.hops),
            _parse_ints(args.hop_distances),
            range(args.samples_per_cell),
        ):
            rows.append(generate_pac_d_multihop(length, hops, hop_distance, args.seed, i, counter))
    out = _samples_output_path(args.output)
    write_jsonl(out, rows)
    print(f"Wrote {len(rows)} PAC samples to {out}")


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
            row.update({"score": 0.0, "metric": "request_error", "error_type": "request_error"})
        else:
            row.update(score_pac_sample(sample, result.response_text))
        append_jsonl(args.output, row)


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
