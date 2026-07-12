from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from lmaf.data.pac2 import generate_pac2_b_calibration, score_pac2_sample
from lmaf.inference.client import create_inference_client, resolve_provider_model
from lmaf.utils.io import TERMINAL_NONRETRY_ERRORS, append_jsonl, read_jsonl, utc_timestamp, write_jsonl
from lmaf.utils.token_count import TokenCounter
from run_unfinished_fast import FRAMEWORK_MODELS, ModelPlan


DEFAULT_MODELS = "qwen35_9b,qwen35_35b_a3b,qwen35_122b_a10b"
DEFAULT_DECOY_COUNTS = "32,64,128,192"
IDEAL_BANDS = {
    "qwen35_9b": (0.30, 0.60),
    "qwen35_35b_a3b": (0.50, 0.75),
    "qwen35_122b_a10b": (0.70, 0.90),
}

FILE_LOCKS: dict[Path, threading.Lock] = defaultdict(threading.Lock)
THREAD_LOCAL = threading.local()
REQUEST_THROTTLE_LOCK = threading.Lock()
LAST_REQUEST_AT = 0.0


@dataclass(frozen=True)
class WorkItem:
    model: ModelPlan
    output: Path
    sample: dict[str, Any]
    prompt_tokens: int


def main() -> None:
    args = parse_args()
    models = resolve_models(args.models)
    samples_path = ROOT / args.data_root / args.run_id / "samples.jsonl"
    output_root = ROOT / args.output_root / args.run_id
    report_root = ROOT / args.report_root / args.run_id
    counter = TokenCounter(args.tokenizer)

    if not args.summarize_only:
        samples = ensure_samples(args, samples_path, counter)
    else:
        samples = list(read_jsonl(samples_path))
        if not samples:
            raise SystemExit(f"No samples found at {samples_path}. Run without --summarize-only first.")

    work = build_work(args, models, samples, output_root)
    write_plan(args, work, samples_path, output_root, report_root)
    print_plan(work)

    if args.dry_run:
        summarize(args, report_root, output_root)
        return
    if args.summarize_only:
        summarize(args, report_root, output_root)
        return
    if work and args.provider == "siliconflow" and not resolve_api_keys(args):
        raise SystemExit("SILICONFLOW_API_KEY or SILICONFLOW_API_KEYS is required.")
    if work:
        random.shuffle(work)
        run_parallel(args, work, resolve_api_keys(args))
    summarize(args, report_root, output_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "PAC-Test 2.0 difficulty calibration. Generates and runs a small PAC-B "
            "high-similarity interference ladder, then recommends a critical decoy count."
        )
    )
    parser.add_argument("--run-id", default="pac2_calibration_main")
    parser.add_argument("--models", default=DEFAULT_MODELS)
    parser.add_argument("--decoy-counts", default=DEFAULT_DECOY_COUNTS)
    parser.add_argument("--samples-per-cell", type=int, default=5)
    parser.add_argument("--length", type=int, default=32000, help="32K-tier prompt budget with room for chat overhead.")
    parser.add_argument("--position", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--provider", choices=["siliconflow", "local", "custom"], default="siliconflow")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--retry", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--max-in-flight", type=int, default=None)
    parser.add_argument(
        "--request-delay-sec",
        type=float,
        default=0.0,
        help="Minimum gap between API request starts. Useful for large 32K prompts under TPM limits.",
    )
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--thinking-budget", type=int, default=None)
    parser.add_argument("--stop-after", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--force-regenerate", action="store_true")
    parser.add_argument("--data-root", default="data/generated/pac2_calibration")
    parser.add_argument("--output-root", default="results/raw/pac2_calibration")
    parser.add_argument("--report-root", default="results/reports/pac2_calibration")
    parser.add_argument("--plan-output", default="results/logs/pac2_calibration_plan.json")
    return parser.parse_args()


def ensure_samples(args: argparse.Namespace, samples_path: Path, counter: TokenCounter) -> list[dict[str, Any]]:
    if args.skip_generate and samples_path.exists():
        return list(read_jsonl(samples_path))
    if samples_path.exists() and not args.force_regenerate:
        rows = list(read_jsonl(samples_path))
        if rows:
            return rows
    rows: list[dict[str, Any]] = []
    for decoy_count in parse_ints(args.decoy_counts):
        for sample_index in range(args.samples_per_cell):
            rows.append(
                generate_pac2_b_calibration(
                    length=args.length,
                    position=args.position,
                    decoy_count=decoy_count,
                    seed=args.seed,
                    sample_index=sample_index,
                    counter=counter,
                )
            )
    write_jsonl(samples_path, rows)
    print(f"Wrote {len(rows)} PAC2 calibration samples to {samples_path}")
    return rows


def build_work(
    args: argparse.Namespace,
    models: list[ModelPlan],
    samples: list[dict[str, Any]],
    output_root: Path,
) -> list[WorkItem]:
    work: list[WorkItem] = []
    for model in models:
        output = output_root / f"{model.alias}.jsonl"
        completed = completed_ids(output)
        for sample in samples:
            sample_id = str(sample["sample_id"])
            if sample_id in completed:
                continue
            prompt_tokens = int(sample.get("length_tokens_actual") or sample.get("length_tokens_target") or 0)
            if prompt_tokens > model.max_model_len:
                if not args.dry_run:
                    append_terminal_skip(output, sample, args, model, prompt_tokens, "skipped_overlength")
                completed.add(sample_id)
                continue
            work.append(WorkItem(model=model, output=output, sample=sample, prompt_tokens=prompt_tokens))
    if args.stop_after is not None:
        work = work[: args.stop_after]
    return work


def completed_ids(output: Path) -> set[str]:
    completed: set[str] = set()
    if not output.exists():
        return completed
    for row in read_jsonl(output):
        sample_id = row.get("sample_id")
        if sample_id not in (None, "") and row.get("error") in (None, "", *TERMINAL_NONRETRY_ERRORS):
            completed.add(str(sample_id))
    return completed


def append_terminal_skip(
    output: Path,
    sample: dict[str, Any],
    args: argparse.Namespace,
    model: ModelPlan,
    prompt_tokens: int,
    reason: str,
) -> None:
    row = dict(sample)
    row.update(
        {
            "model": model.alias,
            "provider": args.provider,
            "api_model": resolve_provider_model(args.provider, model.alias),
            "prediction": "",
            "latency_sec": 0.0,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": 0,
            "timestamp": utc_timestamp(),
            "score": 0.0,
            "metric": reason,
            "error": reason,
            "error_type": reason,
        }
    )
    append_with_lock(output, row)


def run_parallel(args: argparse.Namespace, work: list[WorkItem], api_keys: list[str]) -> None:
    started = time.perf_counter()
    total = len(work)
    done = 0
    errors = 0
    in_flight_limit = args.max_in_flight or max(args.max_workers, args.max_workers * 2)
    print(f"PAC2 calibration pending API calls: {total}")
    print(f"Workers: {args.max_workers}; max in-flight: {in_flight_limit}")
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        pending = []
        iterator = iter(work)
        while True:
            while len(pending) < in_flight_limit:
                try:
                    item = next(iterator)
                except StopIteration:
                    break
                pending.append(pool.submit(run_one, args, item, choose_key(api_keys)))
            if not pending:
                break
            finished, pending_set = wait(pending, return_when=FIRST_COMPLETED)
            pending = list(pending_set)
            for future in finished:
                ok, label = future.result()
                done += 1
                errors += 0 if ok else 1
                if done % max(1, args.max_workers) == 0 or done == total:
                    elapsed = max(1e-6, time.perf_counter() - started)
                    rate = done / elapsed
                    eta = (total - done) / rate if rate else 0
                    print(
                        f"[{done}/{total}] ok={done - errors} errors={errors} "
                        f"rate={rate:.2f}/s eta={eta/60:.1f}m last={label}",
                        flush=True,
                    )


def run_one(args: argparse.Namespace, item: WorkItem, api_key: str | None) -> tuple[bool, str]:
    client = get_thread_client(args, item.model, api_key)
    throttle_request_starts(args.request_delay_sec)
    result = client.generate(
        prompt=str(item.sample["prompt"]),
        request_id=str(item.sample["sample_id"]),
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    row = dict(item.sample)
    row.update(
        {
            "model": item.model.alias,
            "provider": args.provider,
            "api_model": client.served_model_name,
            "prediction": result.response_text,
            "latency_sec": result.latency_sec,
            "prompt_tokens": result.prompt_tokens or item.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "timestamp": utc_timestamp(),
            "error": result.error,
        }
    )
    if result.error:
        row.update({"score": 0.0, "metric": "request_error", "error_type": "request_error"})
    else:
        row.update(score_pac2_sample(item.sample, result.response_text))
    append_with_lock(item.output, row)
    return result.error is None, f"decoy={item.sample.get('decoy_count')}/{item.model.alias}"


def throttle_request_starts(delay_sec: float) -> None:
    global LAST_REQUEST_AT
    if delay_sec <= 0:
        return
    with REQUEST_THROTTLE_LOCK:
        now = time.perf_counter()
        wait_for = LAST_REQUEST_AT + delay_sec - now
        if wait_for > 0:
            time.sleep(wait_for)
        LAST_REQUEST_AT = time.perf_counter()


def get_thread_client(args: argparse.Namespace, model: ModelPlan, api_key: str | None):
    cache = getattr(THREAD_LOCAL, "clients", None)
    if cache is None:
        cache = {}
        THREAD_LOCAL.clients = cache
    enable_thinking = bool(model.enable_thinking or args.enable_thinking)
    key = (args.provider, model.alias, api_key or "", enable_thinking)
    if key not in cache:
        cache[key] = create_inference_client(
            provider=args.provider,
            model_name=model.alias,
            endpoint=args.endpoint,
            api_key=api_key,
            timeout=args.timeout,
            retry=args.retry,
            backoff=(1, 2, 4, 8),
            enable_thinking=enable_thinking,
            thinking_budget=args.thinking_budget,
        )
    return cache[key]


def summarize(args: argparse.Namespace, report_root: Path, output_root: Path) -> None:
    rows = dedupe_rows(load_result_rows(output_root))
    report_root.mkdir(parents=True, exist_ok=True)
    summary_rows = summarize_by_model_decoy(rows)
    by_decoy_rows, recommendation = summarize_by_decoy(summary_rows)
    write_csv(report_root / "calibration_by_model_decoy.csv", summary_rows)
    write_csv(report_root / "calibration_by_decoy.csv", by_decoy_rows)
    (report_root / "calibration_recommendation.json").write_text(
        json.dumps(recommendation, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_markdown(report_root / "README.md", summary_rows, by_decoy_rows, recommendation)
    print(f"Wrote PAC2 calibration report to {report_root}")
    if recommendation.get("recommended_decoy_count") is not None:
        print(
            "Recommended decoy_count: "
            f"{recommendation['recommended_decoy_count']} ({recommendation.get('reason', '')})"
        )
    else:
        print("Recommended decoy_count: not available yet; run more calibration samples.")


def load_result_rows(output_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not output_root.exists():
        return rows
    for path in sorted(output_root.glob("*.jsonl")):
        rows.extend(read_jsonl(path))
    return rows


def dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[tuple[str, str], dict[str, Any]] = {}
    for idx, row in enumerate(rows):
        key = (str(row.get("model") or ""), str(row.get("sample_id") or ""))
        if not key[0] or not key[1]:
            continue
        current = best.get(key)
        if current is None or row_rank(row, idx) >= row_rank(current, -1):
            best[key] = row
    return list(best.values())


def row_rank(row: dict[str, Any], idx: int) -> tuple[int, int]:
    no_error = row.get("error") in (None, "", *TERMINAL_NONRETRY_ERRORS)
    return (1 if no_error else 0, idx)


def summarize_by_model_decoy(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("experiment") != "pac2":
            continue
        buckets[(str(row.get("model")), int(row.get("decoy_count") or 0))].append(row)

    out: list[dict[str, Any]] = []
    for (model, decoy_count), items in sorted(buckets.items(), key=lambda item: (item[0][1], item[0][0])):
        eval_items = [row for row in items if row.get("error") in (None, "")]
        errors = [row for row in items if row.get("error") not in (None, "")]
        scores = [float(row.get("score") or 0) for row in eval_items]
        field_scores = [float(row.get("field_accuracy", row.get("score") or 0) or 0) for row in eval_items]
        decoy_hits = [row for row in eval_items if row.get("error_type") == "decoy_value_capture"]
        partials = [row for row in eval_items if row.get("error_type") == "partial_triplet"]
        omissions = [row for row in eval_items if row.get("error_type") == "omission"]
        near_misses = [row for row in eval_items if row.get("error_type") == "near_miss_value"]
        out.append(
            {
                "model": model,
                "decoy_count": decoy_count,
                "n_total": len(items),
                "n_eval": len(eval_items),
                "n_api_error": len(errors),
                "accuracy": round(mean(scores), 4) if scores else "",
                "mean_field_accuracy": round(mean(field_scores), 4) if field_scores else "",
                "score_all": round(sum(float(row.get("score") or 0) for row in items) / len(items), 4) if items else "",
                "decoy_capture_rate": round(len(decoy_hits) / len(eval_items), 4) if eval_items else "",
                "partial_rate": round(len(partials) / len(eval_items), 4) if eval_items else "",
                "omission_rate": round(len(omissions) / len(eval_items), 4) if eval_items else "",
                "near_miss_rate": round(len(near_misses) / len(eval_items), 4) if eval_items else "",
                "api_error_rate": round(len(errors) / len(items), 4) if items else "",
                "mean_latency": round(mean([float(row.get("latency_sec") or 0) for row in eval_items]), 3) if eval_items else "",
            }
        )
    return out


def summarize_by_decoy(summary_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_decoy: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        by_decoy[int(row["decoy_count"])].append(row)

    rows: list[dict[str, Any]] = []
    for decoy_count, items in sorted(by_decoy.items()):
        accs = [float(row["accuracy"]) for row in items if row.get("accuracy") not in ("", None)]
        score_alls = [float(row["score_all"]) for row in items if row.get("score_all") not in ("", None)]
        field_accs = [float(row["mean_field_accuracy"]) for row in items if row.get("mean_field_accuracy") not in ("", None)]
        eval_counts = [int(row["n_eval"]) for row in items if row.get("n_eval") not in ("", None)]
        api_error_rates = [float(row["api_error_rate"]) for row in items if row.get("api_error_rate") not in ("", None)]
        model_acc = {row["model"]: row.get("accuracy") for row in items}
        weak = _float(model_acc.get("qwen35_9b"))
        mid = _float(model_acc.get("qwen35_35b_a3b"))
        strong = _float(model_acc.get("qwen35_122b_a10b"))
        spread = ""
        if weak is not None and strong is not None:
            spread = round(strong - weak, 4)
        rows.append(
            {
                "decoy_count": decoy_count,
                "models_observed": len(items),
                "mean_accuracy": round(mean(accs), 4) if accs else "",
                "mean_field_accuracy": round(mean(field_accs), 4) if field_accs else "",
                "mean_score_all": round(mean(score_alls), 4) if score_alls else "",
                "min_accuracy": round(min(accs), 4) if accs else "",
                "max_accuracy": round(max(accs), 4) if accs else "",
                "min_eval_per_model": min(eval_counts) if eval_counts else "",
                "max_api_error_rate": round(max(api_error_rates), 4) if api_error_rates else "",
                "strong_minus_weak": spread,
                "qwen35_9b": model_acc.get("qwen35_9b", ""),
                "qwen35_35b_a3b": model_acc.get("qwen35_35b_a3b", ""),
                "qwen35_122b_a10b": model_acc.get("qwen35_122b_a10b", ""),
                "ideal_band_match": int(_ideal_match(model_acc)),
            }
        )
    return rows, recommend_decoy_count(rows)


def recommend_decoy_count(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"recommended_decoy_count": None, "reason": "no completed calibration rows"}

    stable_rows = [
        row
        for row in rows
        if (_float(row.get("min_eval_per_model")) or 0) >= 3
        and (_float(row.get("max_api_error_rate")) or 0) <= 0.40
    ]
    if not stable_rows:
        return {
            "recommended_decoy_count": None,
            "reason": "not enough stable evaluations; at least 3 successful samples per anchor model are needed",
        }

    max_mean = max((_float(row.get("mean_accuracy")) or 0) for row in stable_rows)
    min_mean = min((_float(row.get("mean_accuracy")) or 0) for row in stable_rows)
    max_spread = max(abs(_float(row.get("strong_minus_weak")) or 0) for row in stable_rows)
    if min_mean >= 0.95 and max_mean >= 0.95 and max_spread <= 0.05:
        return {
            "recommended_decoy_count": None,
            "reason": "ceiling effect: all stable decoy counts are near 100% with almost no model spread",
            "suggested_next_decoy_counts": [256, 384, 448],
        }

    ideal = [row for row in stable_rows if int(row.get("ideal_band_match") or 0) == 1]
    if ideal:
        chosen = max(ideal, key=lambda row: _float(row.get("strong_minus_weak")) or 0)
        return {
            "recommended_decoy_count": int(chosen["decoy_count"]),
            "reason": "all three anchor models are inside the target accuracy bands",
            "row": chosen,
        }

    scored: list[tuple[float, dict[str, Any]]] = []
    for row in stable_rows:
        mean_acc = _float(row.get("mean_accuracy"))
        spread = _float(row.get("strong_minus_weak")) or 0.0
        if mean_acc is None:
            continue
        center_score = 1.0 - abs(mean_acc - 0.65)
        ceiling_penalty = 0.35 if mean_acc > 0.92 else 0.0
        collapse_penalty = 0.35 if mean_acc < 0.25 else 0.0
        score = center_score + spread - ceiling_penalty - collapse_penalty
        scored.append((score, row))
    if not scored:
        return {"recommended_decoy_count": None, "reason": "not enough successful API evaluations"}
    _, chosen = max(scored, key=lambda item: item[0])
    return {
        "recommended_decoy_count": int(chosen["decoy_count"]),
        "reason": "fallback: closest to useful mean accuracy while preserving model spread",
        "row": chosen,
    }


def _ideal_match(model_acc: dict[str, Any]) -> bool:
    for model, (lo, hi) in IDEAL_BANDS.items():
        value = _float(model_acc.get(model))
        if value is None or value < lo or value > hi:
            return False
    return True


def write_markdown(
    path: Path,
    summary_rows: list[dict[str, Any]],
    by_decoy_rows: list[dict[str, Any]],
    recommendation: dict[str, Any],
) -> None:
    lines = [
        "# PAC-Test 2.0 Calibration Report",
        "",
        "This report calibrates 32K high-similarity interference difficulty for PAC-B.",
        "",
        f"Recommended decoy_count: `{recommendation.get('recommended_decoy_count')}`",
        "",
        f"Reason: {recommendation.get('reason', '')}",
        "",
        "## By Decoy Count",
        "",
        _markdown_table(by_decoy_rows),
        "",
        "## By Model And Decoy Count",
        "",
        _markdown_table(summary_rows),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_No rows yet._"
    headers = list(rows[0])
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    return "\n".join(out)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_plan(args: argparse.Namespace, work: list[WorkItem], samples_path: Path, output_root: Path, report_root: Path) -> None:
    path = ROOT / args.plan_output
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter((item.model.alias, int(item.sample.get("decoy_count") or 0)) for item in work)
    rows = [
        {"model": model, "decoy_count": decoy_count, "pending_api_calls": count}
        for (model, decoy_count), count in sorted(counts.items(), key=lambda item: (item[0][1], item[0][0]))
    ]
    payload = {
        "run_id": args.run_id,
        "samples_path": str(samples_path),
        "output_root": str(output_root),
        "report_root": str(report_root),
        "length": args.length,
        "position": args.position,
        "samples_per_cell": args.samples_per_cell,
        "decoy_counts": parse_ints(args.decoy_counts),
        "pending": rows,
        "total_pending_api_calls": sum(item["pending_api_calls"] for item in rows),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote PAC2 calibration plan to {path}")


def print_plan(work: list[WorkItem]) -> None:
    counts = Counter((item.model.alias, int(item.sample.get("decoy_count") or 0)) for item in work)
    print("PAC2 calibration pending work:")
    if not counts:
        print("  no pending API calls")
    for (model, decoy_count), count in sorted(counts.items(), key=lambda item: (item[0][1], item[0][0])):
        print(f"  decoy={decoy_count:<4d} {model:22s} {count}")
    print(f"Total pending API calls: {sum(counts.values())}")


def append_with_lock(output: Path, row: dict[str, Any]) -> None:
    lock = FILE_LOCKS[output]
    with lock:
        append_jsonl(output, row)


def resolve_models(value: str) -> list[ModelPlan]:
    models = []
    for name in parse_csv(value):
        if name not in FRAMEWORK_MODELS:
            raise SystemExit(f"Unknown model alias: {name}")
        models.append(FRAMEWORK_MODELS[name])
    if not models:
        raise SystemExit("--models must include at least one model")
    return models


def resolve_api_keys(args: argparse.Namespace) -> list[str]:
    if args.api_key:
        return [args.api_key]
    keys = os.getenv("SILICONFLOW_API_KEYS")
    if keys:
        return [key.strip() for key in keys.split(",") if key.strip()]
    key = os.getenv("SILICONFLOW_API_KEY")
    return [key] if key else []


def choose_key(api_keys: list[str]) -> str | None:
    if not api_keys:
        return None
    return random.choice(api_keys)


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _float(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    main()
