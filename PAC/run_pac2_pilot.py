from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PAC_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PAC_ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_pac2_formal as formal


DEFAULT_PILOT_MODELS = "qwen35_9b,qwen35_35b_a3b,qwen35_122b_a10b"


def main() -> None:
    args = parse_args()
    manifest = formal.load_manifest(args.manifest)
    subsets = formal.resolve_subsets(args.subsets, manifest)
    selected_models = formal.resolve_requested_models(args.models)
    all_samples = formal.load_samples(subsets)
    pilot_samples = select_pilot_samples(all_samples, args.samples_per_condition, args.selection_seed)
    write_selection(args, pilot_samples)
    work = formal.build_work(args, manifest, pilot_samples, selected_models)
    formal.write_plan(args, manifest, subsets, work)
    formal.print_plan(work)
    print(f"Pilot unique samples: {len(pilot_samples)}")
    print(f"Pilot estimated calls: {len(work)}")

    if args.dry_run:
        formal.summarize(args)
        return
    if args.summarize_only:
        formal.summarize(args)
        return
    api_keys = formal.resolve_api_keys(args)
    if work and args.provider == "siliconflow" and not api_keys:
        raise SystemExit("SILICONFLOW_API_KEY or SILICONFLOW_API_KEYS is required.")
    if work:
        random.Random(args.shuffle_seed).shuffle(work)
        formal.run_parallel(args, work, api_keys)
    formal.summarize(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fast PAC-Test 2.0 Formal v5 pilot runner for dataset-quality inspection."
    )
    parser.add_argument("--run-id", default="pac2_pilot_v5_quality")
    parser.add_argument("--manifest", default=str(PAC_ROOT / "manifest.json"))
    parser.add_argument("--subsets", default="A,B,C,D")
    parser.add_argument("--models", default=DEFAULT_PILOT_MODELS)
    parser.add_argument(
        "--samples-per-condition",
        type=int,
        default=2,
        help="Samples per controlled cell. Default 2 gives 58 samples x 3 models = 174 calls.",
    )
    parser.add_argument("--selection-seed", type=int, default=20260707)
    parser.add_argument("--provider", choices=["siliconflow", "local", "custom"], default="siliconflow")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--timeout", type=float, default=360)
    parser.add_argument("--retry", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--max-in-flight", type=int, default=4)
    parser.add_argument(
        "--request-delay-sec",
        type=float,
        default=12.0,
        help="Global gap between request starts. Default targets roughly one hour with 174 calls.",
    )
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--thinking-budget", type=int, default=None)
    parser.add_argument("--stop-after", type=int, default=None)
    parser.add_argument("--shuffle-seed", type=int, default=20260707)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--output-root", default=str(ROOT / "results" / "raw" / "pac2_pilot"))
    parser.add_argument("--report-root", default=str(ROOT / "results" / "reports" / "pac2_pilot"))
    parser.add_argument("--plan-output", default=str(ROOT / "results" / "logs" / "pac2_pilot_plan.json"))
    return parser.parse_args()


def select_pilot_samples(
    samples: list[dict[str, Any]],
    samples_per_condition: int,
    seed: int,
) -> list[dict[str, Any]]:
    if samples_per_condition <= 0:
        raise SystemExit("--samples-per-condition must be positive")
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        buckets[condition_key(sample)].append(sample)

    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for key in sorted(buckets):
        bucket = list(buckets[key])
        rng.shuffle(bucket)
        selected.extend(sorted(bucket[:samples_per_condition], key=lambda row: str(row.get("sample_id"))))
    selected.sort(key=lambda row: (str(row.get("formal_subset")), condition_key(row), str(row.get("sample_id"))))
    return selected


def condition_key(sample: dict[str, Any]) -> tuple[Any, ...]:
    subset = str(sample.get("formal_subset"))
    if subset == "PAC-A_position":
        return (subset, int(sample.get("position_bin") or sample.get("position_percent") or 0))
    if subset == "PAC-B_interference":
        return (subset, int(sample.get("decoy_count") or 0))
    if subset == "PAC-C_binding_capacity":
        return (subset, int(sample.get("binding_k") or 0), int(sample.get("query_count") or 0))
    if subset == "PAC-D_multihop_false_chain":
        return (subset, int(sample.get("hop_count") or 0), int(sample.get("false_chain_count") or 0))
    return (subset, str(sample.get("sample_id")))


def write_selection(args: argparse.Namespace, samples: list[dict[str, Any]]) -> None:
    path = Path(args.plan_output).with_name("pac2_pilot_selected_samples.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = defaultdict(int)
    for sample in samples:
        counts[str(condition_key(sample))] += 1
    path.write_text(
        json.dumps(
            {
                "run_id": args.run_id,
                "samples_per_condition": args.samples_per_condition,
                "n_unique_samples": len(samples),
                "selected_sample_ids": [sample["sample_id"] for sample in samples],
                "condition_counts": dict(sorted(counts.items())),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
