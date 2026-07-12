from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PAC_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PAC_ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_pac2_formal as formal
import run_pac_v21_queue as queue_runner
from run_unfinished_fast import FRAMEWORK_MODELS


FULL_MODELS_NO_HUNYUAN = [
    "qwen35_9b",
    "qwen3_8b",
    "qwen35_27b",
    "qwen35_35b_a3b",
    "qwen35_122b_a10b",
    "seed_oss_36b",
    "qwen3_14b_no_thinking",
    "qwen3_14b_thinking",
]


def main() -> None:
    args = parse_args()
    selected_models = resolve_models(args.models, args.exclude_models)
    samples = override_model_scope(queue_runner.load_selected_samples_from_args(args), selected_models)
    cache = load_result_cache(args, samples, selected_models)
    work = build_work(args, samples, selected_models, cache)

    status = build_status(args, samples, selected_models, cache)
    write_plan(args, selected_models, samples, work, status)
    print_plan(args, selected_models, samples, work, status)

    if args.dry_run:
        summarize_no_hunyuan(args)
        return
    if args.summarize_only:
        summarize_no_hunyuan(args)
        return

    api_keys = queue_runner.resolve_api_keys(args)
    if work and args.provider == "siliconflow" and not api_keys:
        raise SystemExit("SILICONFLOW_API_KEY or SILICONFLOW_API_KEYS is required.")
    if work:
        queue_runner.run_queue(args, work, api_keys)
    else:
        print("No pending PAC v2.1 full no-Hunyuan work.")
    summarize_no_hunyuan(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Queue-based PAC v2.1 full runner. It forces every selected subset to run all "
            "non-Hunyuan framework models, while reusing completed rows and retrying API errors."
        )
    )
    parser.add_argument("--run-id", default="pac_v21_full_queue")
    parser.add_argument("--subsets", default="A,B,C,D21", help="Comma list: A,B,C,D21.")
    parser.add_argument(
        "--models",
        default="all_no_hunyuan",
        help="'all_no_hunyuan' or comma-separated model aliases.",
    )
    parser.add_argument("--exclude-models", default="hunyuan_a13b")
    parser.add_argument("--d21-samples-per-condition", type=int, default=2)
    parser.add_argument("--d21-seed", type=int, default=91021)
    parser.add_argument("--force-generate-d21", action="store_true")
    parser.add_argument("--provider", choices=["siliconflow", "local", "custom"], default="siliconflow")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-keys", default=None, help="Comma-separated API keys. Overrides SILICONFLOW_API_KEYS.")
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--timeout", type=float, default=420)
    parser.add_argument("--retry", type=int, default=1)
    parser.add_argument("--queue-max-attempts", type=int, default=4)
    parser.add_argument("--rate-limit-cooldown-sec", type=float, default=90.0)
    parser.add_argument("--transient-cooldown-sec", type=float, default=25.0)
    parser.add_argument("--connection-cooldown-sec", type=float, default=180.0)
    parser.add_argument("--connection-max-attempts", type=int, default=2)
    parser.add_argument("--connection-burst-threshold", type=int, default=3)
    parser.add_argument("--connection-burst-window-sec", type=float, default=90.0)
    parser.add_argument("--global-connection-cooldown-sec", type=float, default=180.0)
    parser.add_argument(
        "--model-rate-limit-cooldown-sec",
        type=float,
        default=150.0,
        help="When one model/key hits TPM 429, pause only that model/key while other models keep running.",
    )
    parser.add_argument(
        "--global-request-delay-sec",
        type=float,
        default=0.0,
        help="Global minimum gap between API request starts across all workers. Use this for account-level TPM limits.",
    )
    parser.add_argument("--per-key-delay-sec", type=float, default=0.0)
    parser.add_argument("--slots-per-key", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--thinking-budget", type=int, default=None)
    parser.add_argument("--shuffle-seed", type=int, default=20260707)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--stop-after", type=int, default=None)
    parser.add_argument(
        "--rerun-successes",
        action="store_true",
        help="Also rerun rows that already completed successfully. Default only fills missing/API-error rows.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--output-root", default=str(ROOT / "results" / "raw" / "pac_v21_queue"))
    parser.add_argument("--report-root", default=str(ROOT / "results" / "reports" / "pac_v21_full_no_hunyuan_queue"))
    parser.add_argument(
        "--plan-output",
        default=str(ROOT / "results" / "logs" / "pac_v21_full_no_hunyuan_queue_plan.json"),
    )
    return parser.parse_args()


def resolve_models(models_value: str, exclude_value: str) -> list[str]:
    excludes = set(formal.parse_csv(exclude_value))
    if models_value.strip().lower() in {"auto", "all", "all_no_hunyuan", "no_hunyuan"}:
        models = list(FULL_MODELS_NO_HUNYUAN)
    else:
        models = formal.parse_csv(models_value)
    unknown = sorted(model for model in models if model not in FRAMEWORK_MODELS)
    if unknown:
        raise SystemExit(f"Unknown model alias(es): {', '.join(unknown)}")
    return [model for model in models if model not in excludes]


def override_model_scope(samples: list[dict[str, Any]], models: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in samples:
        item = dict(row)
        item["models_to_run"] = list(models)
        item["model_scope"] = "all_no_hunyuan_8"
        rows.append(item)
    return rows


def build_status(
    args: argparse.Namespace,
    samples: list[dict[str, Any]],
    models: list[str],
    cache: dict[Path, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    output_root = Path(args.output_root) / args.run_id
    targets = Counter()
    completed = Counter()
    api_errors = Counter()
    terminal_skips = Counter()
    existing_rows = Counter()
    for sample in samples:
        subset = str(sample["formal_subset"])
        sample_id = str(sample["sample_id"])
        prompt_tokens = int(sample.get("length_tokens_actual") or sample.get("length_tokens_target") or 0)
        for model_alias in models:
            model = FRAMEWORK_MODELS[model_alias]
            key = (subset, model_alias)
            targets[key] += 1
            if prompt_tokens > model.max_model_len:
                terminal_skips[key] += 1
                continue
            output = output_root / subset / f"{model_alias}.jsonl"
            row = cache.get(output, {}).get(sample_id)
            if row is None:
                continue
            existing_rows[key] += 1
            error = row.get("error")
            if error in (None, "", *formal.TERMINAL_NONRETRY_ERRORS):
                completed[key] += 1
            else:
                api_errors[key] += 1
    return {
        "targets": targets,
        "completed": completed,
        "api_errors": api_errors,
        "terminal_skips": terminal_skips,
        "existing_rows": existing_rows,
    }


def build_work(
    args: argparse.Namespace,
    samples: list[dict[str, Any]],
    models: list[str],
    cache: dict[Path, dict[str, dict[str, Any]]],
) -> list[formal.WorkItem]:
    output_root = Path(args.output_root) / args.run_id
    work: list[formal.WorkItem] = []
    for sample in samples:
        subset = str(sample.get("formal_subset") or "unknown_subset")
        sample_id = str(sample["sample_id"])
        prompt_tokens = int(sample.get("length_tokens_actual") or sample.get("length_tokens_target") or 0)
        for model_alias in models:
            model = FRAMEWORK_MODELS[model_alias]
            output = output_root / subset / f"{model.alias}.jsonl"
            if prompt_tokens > model.max_model_len:
                if not args.dry_run and not args.summarize_only:
                    formal.append_terminal_skip(output, sample, args, model, prompt_tokens, "skipped_overlength")
                continue
            row = cache.get(output, {}).get(sample_id)
            if (
                not args.rerun_successes
                and row is not None
                and row.get("error") in (None, "", *formal.TERMINAL_NONRETRY_ERRORS)
            ):
                continue
            work.append(
                formal.WorkItem(
                    subset=subset,
                    model=model,
                    output=output,
                    sample=sample,
                    prompt_tokens=prompt_tokens,
                )
            )
    if args.stop_after is not None:
        work = work[: args.stop_after]
    return work


def load_result_cache(
    args: argparse.Namespace,
    samples: list[dict[str, Any]],
    models: list[str],
) -> dict[Path, dict[str, dict[str, Any]]]:
    output_root = Path(args.output_root) / args.run_id
    paths = {
        output_root / str(sample.get("formal_subset") or "unknown_subset") / f"{model}.jsonl"
        for sample in samples
        for model in models
    }
    cache: dict[Path, dict[str, dict[str, Any]]] = {}
    for path in sorted(paths):
        latest: dict[str, dict[str, Any]] = {}
        if path.exists():
            for row in formal.read_jsonl(path):
                sample_id = row.get("sample_id")
                if sample_id not in (None, ""):
                    latest[str(sample_id)] = row
        cache[path] = latest
    return cache


def write_plan(
    args: argparse.Namespace,
    models: list[str],
    samples: list[dict[str, Any]],
    work: list[formal.WorkItem],
    status: dict[str, Any],
) -> None:
    path = Path(args.plan_output)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending_counts = Counter((item.subset, item.model.alias) for item in work)
    rows = []
    keys = sorted(set(status["targets"]) | set(pending_counts))
    for subset, model in keys:
        rows.append(
            {
                "subset": subset,
                "model": model,
                "target_calls": status["targets"][(subset, model)],
                "completed_success_or_terminal": status["completed"][(subset, model)]
                + status["terminal_skips"][(subset, model)],
                "latest_api_error_rows": status["api_errors"][(subset, model)],
                "pending_api_calls": pending_counts[(subset, model)],
            }
        )
    payload = {
        "run_id": args.run_id,
        "mode": "pac_v21_full_no_hunyuan_queue",
        "selected_models": models,
        "selected_subsets": formal.parse_csv(args.subsets),
        "unique_samples": len(samples),
        "target_api_calls": sum(status["targets"].values()),
        "total_pending_api_calls": len(work),
        "rate_control": {
            "rerun_successes": args.rerun_successes,
            "slots_per_key": args.slots_per_key,
            "global_request_delay_sec": args.global_request_delay_sec,
            "per_key_delay_sec": args.per_key_delay_sec,
            "queue_max_attempts": args.queue_max_attempts,
            "connection_max_attempts": args.connection_max_attempts,
            "connection_cooldown_sec": args.connection_cooldown_sec,
            "connection_burst_threshold": args.connection_burst_threshold,
            "connection_burst_window_sec": args.connection_burst_window_sec,
            "global_connection_cooldown_sec": args.global_connection_cooldown_sec,
            "rate_limit_cooldown_sec": args.rate_limit_cooldown_sec,
            "model_rate_limit_cooldown_sec": args.model_rate_limit_cooldown_sec,
            "transient_cooldown_sec": args.transient_cooldown_sec,
            "timeout": args.timeout,
            "sdk_retry": args.retry,
        },
        "pending_by_subset_model": rows,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def print_plan(
    args: argparse.Namespace,
    models: list[str],
    samples: list[dict[str, Any]],
    work: list[formal.WorkItem],
    status: dict[str, Any],
) -> None:
    pending_counts = Counter((item.subset, item.model.alias) for item in work)
    subset_counts = Counter(str(row["formal_subset"]) for row in samples)
    print("PAC v2.1 full no-Hunyuan queue plan:")
    for subset, count in sorted(subset_counts.items()):
        print(f"  samples {subset:<30} {count}")
    print(f"Models: {', '.join(models)}")
    print(f"Target API calls: {sum(status['targets'].values())}")
    print(f"Rerun successes: {args.rerun_successes}")
    print(f"Already successful/terminal: {sum(status['completed'].values()) + sum(status['terminal_skips'].values())}")
    print(f"Latest API-error rows detected: {sum(status['api_errors'].values())}")
    print("Pending API calls by subset/model:")
    for (subset, model), count in sorted(pending_counts.items()):
        print(f"  {subset:<30} {model:<24} {count}")
    print(f"Total pending API calls: {len(work)}")
    print("Queue mode: each key/slot pulls the next pending sample as soon as it is released.")


def summarize_no_hunyuan(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root) / args.run_id
    report_root = Path(args.report_root) / args.run_id
    report_root.mkdir(parents=True, exist_ok=True)
    rows = [row for row in formal.dedupe_rows(formal.load_result_rows(output_root)) if row.get("model") != "hunyuan_a13b"]
    subset_rows = formal.summarize_by_subset_model(rows)
    condition_rows = formal.summarize_by_condition_model(rows)
    error_rows = formal.summarize_errors(rows)
    formal.write_csv(report_root / "summary_by_subset_model.csv", subset_rows)
    formal.write_csv(report_root / "summary_by_condition_model.csv", condition_rows)
    formal.write_csv(report_root / "error_types.csv", error_rows)
    write_completion_csv(report_root / "completion_by_subset_model.csv", rows)
    write_readme(report_root, len(rows), subset_rows, condition_rows, error_rows)
    print(f"Wrote PAC v2.1 no-Hunyuan report to {report_root}")


def write_completion_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    counts: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: {"rows": 0, "eval": 0, "api_error": 0})
    for row in rows:
        key = (str(row.get("formal_subset") or ""), str(row.get("model") or ""))
        counts[key]["rows"] += 1
        if row.get("error") in (None, ""):
            counts[key]["eval"] += 1
        else:
            counts[key]["api_error"] += 1
    out = []
    for (subset, model), item in sorted(counts.items()):
        out.append({"subset": subset, "model": model, **item})
    write_csv(path, out)


def write_readme(
    report_root: Path,
    n_rows: int,
    subset_rows: list[dict[str, Any]],
    condition_rows: list[dict[str, Any]],
    error_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# PAC v2.1 Full No-Hunyuan Queue Report",
        "",
        "Generated by `PAC/run_pac_v21_full_no_hunyuan_queue.py`.",
        "Hunyuan-A13B is excluded from these summaries.",
        "",
        "## Files",
        "",
        "- `summary_by_subset_model.csv`",
        "- `summary_by_condition_model.csv`",
        "- `error_types.csv`",
        "- `completion_by_subset_model.csv`",
        "",
        f"Deduplicated no-Hunyuan result rows: `{n_rows}`",
        f"Subset/model summary rows: `{len(subset_rows)}`",
        f"Condition/model summary rows: `{len(condition_rows)}`",
        f"Error-type rows: `{len(error_rows)}`",
        "",
    ]
    report_root.joinpath("README.md").write_text("\n".join(lines), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
