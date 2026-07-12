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
PAC_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from lmaf.data.pac2 import score_pac2_sample
from lmaf.inference.client import create_inference_client, resolve_provider_model
from lmaf.utils.io import TERMINAL_NONRETRY_ERRORS, append_jsonl, read_jsonl, utc_timestamp
from run_unfinished_fast import FRAMEWORK_MODELS, ModelPlan


SUBSET_ALIASES = {
    "A": "PAC-A_position",
    "B": "PAC-B_interference",
    "C": "PAC-C_binding_capacity",
    "D": "PAC-D_multihop_false_chain",
}

FILE_LOCKS: dict[Path, threading.Lock] = defaultdict(threading.Lock)
THREAD_LOCAL = threading.local()
REQUEST_THROTTLE_LOCK = threading.Lock()
LAST_REQUEST_AT = 0.0


@dataclass(frozen=True)
class WorkItem:
    subset: str
    model: ModelPlan
    output: Path
    sample: dict[str, Any]
    prompt_tokens: int


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    selected_subsets = resolve_subsets(args.subsets, manifest)
    selected_models = resolve_requested_models(args.models)
    samples = load_samples(selected_subsets)
    work = build_work(args, manifest, samples, selected_models)
    write_plan(args, manifest, selected_subsets, work)
    print_plan(work)

    if args.dry_run:
        summarize(args)
        return
    if args.summarize_only:
        summarize(args)
        return
    if work and args.provider == "siliconflow" and not resolve_api_keys(args):
        raise SystemExit("SILICONFLOW_API_KEY or SILICONFLOW_API_KEYS is required.")
    if work:
        random.Random(args.shuffle_seed).shuffle(work)
        run_parallel(args, work, resolve_api_keys(args))
    summarize(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PAC-Test 2.0 Formal v5 with resume and rate control.")
    parser.add_argument("--run-id", default="pac2_formal_v5_main")
    parser.add_argument("--manifest", default=str(PAC_ROOT / "manifest.json"))
    parser.add_argument("--subsets", default="A,B,C,D", help="Comma list: A,B,C,D or full subset names.")
    parser.add_argument("--models", default="auto", help="'auto' uses each sample's models_to_run, or comma model aliases.")
    parser.add_argument("--provider", choices=["siliconflow", "local", "custom"], default="siliconflow")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--timeout", type=float, default=360)
    parser.add_argument("--retry", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument("--max-in-flight", type=int, default=3)
    parser.add_argument(
        "--request-delay-sec",
        type=float,
        default=10.0,
        help="Global minimum gap between API request starts. 10 sec is the balanced default for 32K prompts.",
    )
    parser.add_argument("--enable-thinking", action="store_true", help="Force thinking on for every selected model.")
    parser.add_argument("--thinking-budget", type=int, default=None)
    parser.add_argument("--stop-after", type=int, default=None)
    parser.add_argument("--shuffle-seed", type=int, default=20260707)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--output-root", default=str(ROOT / "results" / "raw" / "pac2_formal"))
    parser.add_argument("--report-root", default=str(ROOT / "results" / "reports" / "pac2_formal"))
    parser.add_argument("--plan-output", default=str(ROOT / "results" / "logs" / "pac2_formal_plan.json"))
    return parser.parse_args()


def load_manifest(path: str) -> dict[str, Any]:
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise SystemExit(f"PAC manifest not found: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def resolve_subsets(value: str, manifest: dict[str, Any]) -> list[str]:
    available = set(manifest.get("subsets") or {})
    out: list[str] = []
    for item in parse_csv(value):
        subset = SUBSET_ALIASES.get(item.upper(), item)
        if subset not in available:
            raise SystemExit(f"Unknown PAC subset: {item}. Available: {', '.join(sorted(available))}")
        out.append(subset)
    return out


def resolve_requested_models(value: str) -> set[str] | None:
    if value.strip().lower() == "auto":
        return None
    models = set(parse_csv(value))
    unknown = sorted(model for model in models if model not in FRAMEWORK_MODELS)
    if unknown:
        raise SystemExit(f"Unknown model alias(es): {', '.join(unknown)}")
    return models


def load_samples(subsets: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for subset in subsets:
        path = PAC_ROOT / "data" / subset / "samples.jsonl"
        if not path.exists():
            raise SystemExit(f"PAC subset file not found: {path}")
        rows.extend(read_jsonl(path))
    return rows


def build_work(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    samples: list[dict[str, Any]],
    selected_models: set[str] | None,
) -> list[WorkItem]:
    output_root = Path(args.output_root) / args.run_id
    work: list[WorkItem] = []
    for sample in samples:
        subset = str(sample.get("formal_subset") or "unknown_subset")
        sample_models = [str(model) for model in sample.get("models_to_run") or []]
        if selected_models is not None:
            sample_models = [model for model in sample_models if model in selected_models]
        for model_alias in sample_models:
            if model_alias not in FRAMEWORK_MODELS:
                raise SystemExit(f"Unknown model alias in PAC sample {sample.get('sample_id')}: {model_alias}")
            model = FRAMEWORK_MODELS[model_alias]
            output = output_root / subset / f"{model.alias}.jsonl"
            completed = completed_ids(output)
            sample_id = str(sample["sample_id"])
            if sample_id in completed:
                continue
            prompt_tokens = int(sample.get("length_tokens_actual") or sample.get("length_tokens_target") or 0)
            if prompt_tokens > model.max_model_len:
                if not args.dry_run and not args.summarize_only:
                    append_terminal_skip(output, sample, args, model, prompt_tokens, "skipped_overlength")
                continue
            work.append(WorkItem(subset=subset, model=model, output=output, sample=sample, prompt_tokens=prompt_tokens))
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
            "field_accuracy": 0.0,
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
    in_flight_limit = args.max_in_flight or max(1, args.max_workers)
    print(f"PAC2 formal pending API calls: {total}")
    print(
        f"Workers={args.max_workers}; max_in_flight={in_flight_limit}; "
        f"request_delay_sec={args.request_delay_sec}"
    )
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
                        f"rate={rate:.3f}/s eta={eta/60:.1f}m last={label}",
                        flush=True,
                    )


def run_one(args: argparse.Namespace, item: WorkItem, api_key: str | None) -> tuple[bool, str]:
    client = get_thread_client(args, item.model, api_key)
    throttle_request_starts(args.request_delay_sec)
    result = client.generate(
        prompt=str(item.sample["prompt"]),
        request_id=f"{item.subset}/{item.model.alias}/{item.sample['sample_id']}",
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
        row.update({"score": 0.0, "field_accuracy": 0.0, "metric": "request_error", "error_type": "request_error"})
    else:
        row.update(score_pac2_sample(item.sample, result.response_text))
    append_with_lock(item.output, row)
    return result.error is None, f"{item.subset}/{item.model.alias}"


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
            backoff=(2, 4, 8, 16),
            enable_thinking=enable_thinking,
            thinking_budget=args.thinking_budget,
        )
    return cache[key]


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


def summarize(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root) / args.run_id
    report_root = Path(args.report_root) / args.run_id
    report_root.mkdir(parents=True, exist_ok=True)
    rows = dedupe_rows(load_result_rows(output_root))
    summary_by_subset_model = summarize_by_subset_model(rows)
    summary_by_condition_model = summarize_by_condition_model(rows)
    error_rows = summarize_errors(rows)
    write_csv(report_root / "summary_by_subset_model.csv", summary_by_subset_model)
    write_csv(report_root / "summary_by_condition_model.csv", summary_by_condition_model)
    write_csv(report_root / "error_types.csv", error_rows)
    write_readme(report_root, summary_by_subset_model, summary_by_condition_model, error_rows)
    print(f"Wrote PAC2 formal report to {report_root}")


def load_result_rows(output_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not output_root.exists():
        return rows
    for path in sorted(output_root.glob("*/*.jsonl")):
        rows.extend(read_jsonl(path))
    return rows


def dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[tuple[str, str, str], tuple[int, dict[str, Any]]] = {}
    for idx, row in enumerate(rows):
        key = (
            str(row.get("formal_subset") or ""),
            str(row.get("model") or ""),
            str(row.get("sample_id") or ""),
        )
        if not all(key):
            continue
        current = best.get(key)
        rank = row_rank(row, idx)
        if current is None or rank >= row_rank(current[1], current[0]):
            best[key] = (idx, row)
    return [item[1] for item in best.values()]


def row_rank(row: dict[str, Any], idx: int) -> tuple[int, int]:
    no_error = row.get("error") in (None, "", *TERMINAL_NONRETRY_ERRORS)
    return (1 if no_error else 0, idx)


def summarize_by_subset_model(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[(str(row.get("formal_subset")), str(row.get("model")))].append(row)
    out = []
    for (subset, model), items in sorted(buckets.items()):
        out.append(summary_row({"subset": subset, "model": model}, items))
    return out


def summarize_by_condition_model(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        subset = str(row.get("formal_subset"))
        model = str(row.get("model"))
        if subset == "PAC-A_position":
            key = (subset, "position", row.get("position_bin"), model)
        elif subset == "PAC-B_interference":
            key = (subset, "decoy_count", row.get("decoy_count"), model)
        elif subset == "PAC-C_binding_capacity":
            key = (subset, "binding_k/query_count", f"{row.get('binding_k')}/{row.get('query_count')}", model)
        elif subset == "PAC-D_multihop_false_chain":
            key = (subset, "hop_count/false_chain_count", f"{row.get('hop_count')}/{row.get('false_chain_count')}", model)
        else:
            key = (subset, "unknown", "", model)
        buckets[key].append(row)
    out = []
    for (subset, condition_name, condition_value, model), items in sorted(buckets.items()):
        out.append(summary_row(
            {
                "subset": subset,
                "condition_name": condition_name,
                "condition_value": condition_value,
                "model": model,
            },
            items,
        ))
    return out


def summary_row(prefix: dict[str, Any], items: list[dict[str, Any]]) -> dict[str, Any]:
    eval_items = [row for row in items if row.get("error") in (None, "")]
    error_items = [row for row in items if row.get("error") not in (None, "")]
    scores = [float(row.get("score") or 0) for row in eval_items]
    field_scores = [float(row.get("field_accuracy", row.get("score") or 0) or 0) for row in eval_items]
    out = dict(prefix)
    out.update(
        {
            "n_total": len(items),
            "n_eval": len(eval_items),
            "n_api_error": len(error_items),
            "accuracy": round(mean(scores), 4) if scores else "",
            "mean_field_accuracy": round(mean(field_scores), 4) if field_scores else "",
            "decoy_capture_rate": rate(eval_items, "decoy_value_capture"),
            "partial_rate": rate(eval_items, "partial_triplet"),
            "omission_rate": rate(eval_items, "omission"),
            "near_miss_rate": rate(eval_items, "near_miss_value"),
            "api_error_rate": round(len(error_items) / len(items), 4) if items else "",
            "mean_latency": round(mean([float(row.get("latency_sec") or 0) for row in eval_items]), 3) if eval_items else "",
        }
    )
    return out


def summarize_errors(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter((row.get("formal_subset"), row.get("model"), row.get("error_type") or "correct") for row in rows)
    out = []
    for (subset, model, error_type), count in sorted(counts.items()):
        out.append({"subset": subset, "model": model, "error_type": error_type, "count": count})
    return out


def rate(rows: list[dict[str, Any]], error_type: str) -> float | str:
    if not rows:
        return ""
    return round(sum(1 for row in rows if row.get("error_type") == error_type) / len(rows), 4)


def write_readme(
    report_root: Path,
    subset_rows: list[dict[str, Any]],
    condition_rows: list[dict[str, Any]],
    error_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# PAC-Test 2.0 Formal Report",
        "",
        "Generated by `PAC/run_pac2_formal.py`.",
        "",
        "## Files",
        "",
        "- `summary_by_subset_model.csv`",
        "- `summary_by_condition_model.csv`",
        "- `error_types.csv`",
        "",
        f"Completed subset/model rows: `{len(subset_rows)}`",
        f"Completed condition/model rows: `{len(condition_rows)}`",
        f"Error-type rows: `{len(error_rows)}`",
        "",
    ]
    (report_root / "README.md").write_text("\n".join(lines), encoding="utf-8")


def write_plan(args: argparse.Namespace, manifest: dict[str, Any], subsets: list[str], work: list[WorkItem]) -> None:
    path = Path(args.plan_output)
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter((item.subset, item.model.alias) for item in work)
    rows = [
        {"subset": subset, "model": model, "pending_api_calls": count}
        for (subset, model), count in sorted(counts.items())
    ]
    path.write_text(
        json.dumps(
            {
                "run_id": args.run_id,
                "selected_subsets": subsets,
                "total_pending_api_calls": len(work),
                "rate_control": {
                    "max_workers": args.max_workers,
                    "max_in_flight": args.max_in_flight,
                    "request_delay_sec": args.request_delay_sec,
                    "timeout": args.timeout,
                    "retry": args.retry,
                },
                "pending_by_subset_model": rows,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def print_plan(work: list[WorkItem]) -> None:
    counts = Counter((item.subset, item.model.alias) for item in work)
    print("PAC2 formal pending work:")
    for (subset, model), count in sorted(counts.items()):
        print(f"  {subset:<30} {model:<24} {count}")
    print(f"Total pending API calls: {len(work)}")


def append_with_lock(path: Path, row: dict[str, Any]) -> None:
    lock = FILE_LOCKS[path]
    with lock:
        append_jsonl(path, row)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


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
    return api_keys[random.randrange(len(api_keys))]


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    main()
