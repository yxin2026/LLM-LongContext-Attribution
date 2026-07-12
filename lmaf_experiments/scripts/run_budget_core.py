from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from lmaf.data.pac import adapt_external_pac_sample
from lmaf.utils.io import TERMINAL_NONRETRY_ERRORS, append_jsonl, collect_jsonl, iter_jsonl_paths, read_jsonl, utc_timestamp
from lmaf.utils.token_count import TokenCounter
from run_unfinished_fast import FRAMEWORK_MODELS, PAC_SUBSETS, WorkItem, run_parallel


DEFAULT_MODELS = ",".join(FRAMEWORK_MODELS)
DEFAULT_LONGBENCH_TASKS = "narrativeqa,qasper,hotpotqa,2wikimqa,gov_report,multi_news"
DEFAULT_RULER_TASKS = "niah,variable_tracking,common_words_extraction,qa_hotpotqa"
DEFAULT_RULER_LENGTHS = "4096,16384,32768"
DEFAULT_PAC_SOURCE = os.getenv(
    "PAC_TEST_DATA_DIR",
    r"D:\Workspace\llm-longcontext-attribution\上下文机制探究\PAC-Test-Dataset\data",
)


def main() -> None:
    args = parse_args()
    models = resolve_models(args.models)
    experiments = parse_csv(args.experiments)
    api_keys = resolve_api_keys(args)
    counter = TokenCounter(args.tokenizer)

    selected = select_budget_samples(args, experiments, counter)
    write_manifest(args, selected)
    work, stats = build_budget_work(args, models, selected, counter)
    write_plan(args, selected, work, stats, models, experiments)

    print_plan(selected, work, stats)
    if args.dry_run:
        return
    if args.provider == "siliconflow" and not api_keys:
        raise SystemExit("SILICONFLOW_API_KEY or SILICONFLOW_API_KEYS is required.")
    if not work:
        print("No budget-core API work remains. Existing/skipped rows already cover the selected budget subset.")
        return

    random.shuffle(work)
    run_parallel(args, work, api_keys)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the reduced budget-core experiment profile, reusing existing successful results."
    )
    parser.add_argument("--experiments", default="niah,longbench,ruler,pac")
    parser.add_argument("--models", default=DEFAULT_MODELS, help="Comma-separated framework model aliases.")
    parser.add_argument("--provider", choices=["siliconflow", "local", "custom"], default="siliconflow")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--max-in-flight", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--retry", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--thinking-budget", type=int, default=None)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-after", type=int, default=None)

    parser.add_argument("--run-id", default="budget_core_main")
    parser.add_argument("--output-root", default="results/raw/budget_core")
    parser.add_argument("--manifest-root", default="data/processed/budget_core")
    parser.add_argument("--plan-output", default="results/logs/budget_core_plan.json")
    parser.add_argument("--no-reuse-existing", action="store_true")

    parser.add_argument("--niah-data", default="data/generated/niah_batch/framework_v2_without_fast16k")
    parser.add_argument(
        "--niah-source-output-dir",
        default="results/raw/niah_batch/framework_v2_without_fast16k/framework_v2_extra",
    )
    parser.add_argument("--niah-samples-per-cell", type=int, default=20)

    parser.add_argument("--longbench-data", default="data/processed/longbench_ruler_batch/framework_v2/longbench")
    parser.add_argument(
        "--longbench-source-output-dir",
        default="results/raw/longbench_ruler_batch/framework_v2/longbench_ruler_main/longbench",
    )
    parser.add_argument("--longbench-tasks", default=DEFAULT_LONGBENCH_TASKS)
    parser.add_argument("--longbench-samples-per-task", type=int, default=50)
    parser.add_argument("--longbench-truncate", choices=["none", "middle"], default="none")

    parser.add_argument("--ruler-data", default="data/processed/longbench_ruler_batch/framework_v2/ruler")
    parser.add_argument(
        "--ruler-source-output-dir",
        default="results/raw/longbench_ruler_batch/framework_v2/longbench_ruler_main/ruler",
    )
    parser.add_argument("--ruler-tasks", default=DEFAULT_RULER_TASKS)
    parser.add_argument("--ruler-lengths", default=DEFAULT_RULER_LENGTHS)
    parser.add_argument("--ruler-samples-per-cell", type=int, default=20)

    parser.add_argument("--pac-source-data", default=DEFAULT_PAC_SOURCE)
    parser.add_argument("--pac-source-output-dir", default="results/raw/pac_batch/pac_main")
    parser.add_argument("--pac-subsets", default="A,B,C,D")
    parser.add_argument("--pac-a-samples", type=int, default=200)
    parser.add_argument("--pac-b-samples", type=int, default=200)
    parser.add_argument("--pac-c-samples", type=int, default=100)
    parser.add_argument("--pac-d-samples", type=int, default=100)
    return parser.parse_args()


def resolve_models(value: str) -> list[Any]:
    models = []
    for name in parse_csv(value):
        if name not in FRAMEWORK_MODELS:
            raise SystemExit(f"Unknown model alias: {name}")
        models.append(FRAMEWORK_MODELS[name])
    if not models:
        raise SystemExit("--models must include at least one model alias")
    return models


def resolve_api_keys(args: argparse.Namespace) -> list[str]:
    if args.api_key:
        return [args.api_key]
    keys = os.getenv("SILICONFLOW_API_KEYS")
    if keys:
        return [key.strip() for key in keys.split(",") if key.strip()]
    key = os.getenv("SILICONFLOW_API_KEY")
    return [key] if key else []


def select_budget_samples(
    args: argparse.Namespace,
    experiments: list[str],
    counter: TokenCounter,
) -> dict[str, list[dict[str, Any]]]:
    selected: dict[str, list[dict[str, Any]]] = {}
    if "niah" in experiments:
        selected["niah"] = select_niah_samples(args)
    if "longbench" in experiments:
        selected["longbench"] = select_longbench_samples(args)
    if "ruler" in experiments:
        selected["ruler"] = select_ruler_samples(args)
    if "pac" in experiments:
        selected["pac"] = select_pac_samples(args, counter)
    return selected


def select_niah_samples(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = collect_jsonl(ROOT / args.niah_data)
    return select_grouped(rows, niah_group_key, args.niah_samples_per_cell)


def niah_group_key(row: dict[str, Any]) -> tuple[Any, ...]:
    subtask = row.get("subtask")
    length = int(row.get("length_tokens_target") or 0)
    if subtask == "single":
        return (subtask, length, row.get("position_percent"))
    if subtask == "multi":
        return (subtask, length, row.get("distribution"))
    if subtask == "sequential":
        return (subtask, length, row.get("hop"))
    return (subtask, length)


def select_longbench_samples(args: argparse.Namespace) -> list[dict[str, Any]]:
    tasks = set(parse_csv(args.longbench_tasks))
    rows_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in iter_jsonl_paths(ROOT / args.longbench_data):
        task = path.stem
        if task not in tasks:
            continue
        rows_by_task[task].extend(read_jsonl(path))
    selected: list[dict[str, Any]] = []
    for task in parse_csv(args.longbench_tasks):
        rows = rows_by_task.get(task, [])
        if not rows:
            raise SystemExit(f"No LongBench rows found for task: {task}")
        selected.extend(sorted(rows, key=sample_sort_key)[: args.longbench_samples_per_task])
    return selected


def select_ruler_samples(args: argparse.Namespace) -> list[dict[str, Any]]:
    tasks = set(parse_csv(args.ruler_tasks))
    lengths = {int(item) for item in parse_csv(args.ruler_lengths)}
    rows = [
        row
        for row in collect_jsonl(ROOT / args.ruler_data)
        if row.get("subtask") in tasks and int(row.get("length_tokens_target") or 0) in lengths
    ]
    return select_grouped(
        rows,
        lambda row: (row.get("subtask"), int(row.get("length_tokens_target") or 0)),
        args.ruler_samples_per_cell,
    )


def select_pac_samples(args: argparse.Namespace, counter: TokenCounter) -> list[dict[str, Any]]:
    limits = {
        "A": args.pac_a_samples,
        "B": args.pac_b_samples,
        "C": args.pac_c_samples,
        "D": args.pac_d_samples,
    }
    selected: list[dict[str, Any]] = []
    for subset_key in parse_csv(args.pac_subsets):
        subset_key = subset_key.upper()
        if subset_key not in PAC_SUBSETS:
            raise SystemExit(f"Unsupported PAC subset: {subset_key}")
        _, filename = PAC_SUBSETS[subset_key]
        source = pac_source_for_subset(args.pac_source_data, filename)
        if not source.exists():
            raise SystemExit(f"PAC source file not found: {source}")
        raw_rows = list(read_jsonl(source))
        chosen = stratified_select(raw_rows, lambda row, subset_key=subset_key: pac_group_key(subset_key, row), limits[subset_key])
        selected.extend(adapt_external_pac_sample(row, counter=counter, count_tokens=False) for row in chosen)
    return selected


def pac_group_key(subset_key: str, row: dict[str, Any]) -> tuple[Any, ...]:
    length = row.get("total_length")
    if subset_key == "A":
        return (length, row.get("position_ratio"), row.get("template_type"))
    if subset_key == "B":
        return (length, row.get("noise_density"), row.get("dilution_type"))
    if subset_key == "C":
        return (length, row.get("similarity_level"), row.get("distance_level"))
    if subset_key == "D":
        return (length, row.get("num_hops"), row.get("distance_level"))
    return (length,)


def build_budget_work(
    args: argparse.Namespace,
    models: list[Any],
    selected: dict[str, list[dict[str, Any]]],
    counter: TokenCounter,
) -> tuple[list[WorkItem], Counter[tuple[str, str, str, str]]]:
    work: list[WorkItem] = []
    stats: Counter[tuple[str, str, str, str]] = Counter()
    caches: dict[Path, dict[str, dict[str, Any]]] = {}
    budget_root = ROOT / args.output_root / args.run_id

    for model in models:
        if "niah" in selected:
            output = budget_root / "niah" / f"{model.alias}.jsonl"
            source_output = ROOT / args.niah_source_output_dir / f"{model.alias}.jsonl"
            for sample in selected["niah"]:
                prompt_tokens = int(sample.get("length_tokens_actual") or sample.get("length_tokens_target") or 0)
                add_budget_item(
                    args,
                    work,
                    stats,
                    caches,
                    experiment="niah",
                    dataset_label="budget_niah",
                    model=model,
                    output=output,
                    source_output=source_output,
                    sample=sample,
                    prompt=str(sample["prompt"]),
                    prompt_tokens=prompt_tokens,
                    score_fn="niah",
                    overlength_reason="skipped_by_model_length",
                )

        if "longbench" in selected:
            output = budget_root / "longbench" / f"{model.alias}.jsonl"
            source_output = ROOT / args.longbench_source_output_dir / f"{model.alias}.jsonl"
            for sample in selected["longbench"]:
                prompt = str(sample["prompt"])
                prompt_tokens = counter.count(prompt)
                sample_for_run = sample
                if prompt_tokens > model.max_model_len and args.longbench_truncate == "middle":
                    prompt = middle_truncate(prompt, model.max_model_len, counter)
                    prompt_tokens = counter.count(prompt)
                    sample_for_run = dict(sample)
                    sample_for_run["truncation"] = "middle"
                add_budget_item(
                    args,
                    work,
                    stats,
                    caches,
                    experiment="longbench",
                    dataset_label="budget_longbench",
                    model=model,
                    output=output,
                    source_output=source_output,
                    sample=sample_for_run,
                    prompt=prompt,
                    prompt_tokens=prompt_tokens,
                    score_fn="longbench",
                    overlength_reason="skipped_overlength",
                )

        if "ruler" in selected:
            output = budget_root / "ruler" / f"{model.alias}.jsonl"
            source_output = ROOT / args.ruler_source_output_dir / f"{model.alias}.jsonl"
            for sample in selected["ruler"]:
                prompt = str(sample["prompt"])
                add_budget_item(
                    args,
                    work,
                    stats,
                    caches,
                    experiment="ruler",
                    dataset_label="budget_ruler",
                    model=model,
                    output=output,
                    source_output=source_output,
                    sample=sample,
                    prompt=prompt,
                    prompt_tokens=counter.count(prompt),
                    score_fn="ruler",
                    overlength_reason="skipped_overlength",
                )

        if "pac" in selected:
            for sample in selected["pac"]:
                subset_key = str(sample.get("subset") or "").upper()
                output = budget_root / "pac" / subset_key / f"{model.alias}.jsonl"
                source_output = ROOT / args.pac_source_output_dir / subset_key / f"{model.alias}.jsonl"
                prompt = str(sample["prompt"])
                add_budget_item(
                    args,
                    work,
                    stats,
                    caches,
                    experiment="pac",
                    dataset_label=f"budget_pac_{subset_key}",
                    model=model,
                    output=output,
                    source_output=source_output,
                    sample=sample,
                    prompt=prompt,
                    prompt_tokens=prompt_tokens_from_sample(sample, prompt, counter),
                    score_fn="pac",
                    overlength_reason="skipped_overlength",
                )

    if args.stop_after is not None:
        work = work[: args.stop_after]
    return work, stats


def add_budget_item(
    args: argparse.Namespace,
    work: list[WorkItem],
    stats: Counter[tuple[str, str, str, str]],
    caches: dict[Path, dict[str, dict[str, Any]]],
    *,
    experiment: str,
    dataset_label: str,
    model: Any,
    output: Path,
    source_output: Path,
    sample: dict[str, Any],
    prompt: str,
    prompt_tokens: int,
    score_fn: str,
    overlength_reason: str,
) -> None:
    sample_id = str(sample["sample_id"])
    budget_completed = completed_rows(output, caches)
    stats[(experiment, dataset_label, model.alias, "selected")] += 1
    if sample_id in budget_completed:
        stats[(experiment, dataset_label, model.alias, "already_in_budget")] += 1
        return

    source_completed = {} if args.no_reuse_existing else completed_rows(source_output, caches)
    if sample_id in source_completed:
        stats[(experiment, dataset_label, model.alias, "reused_existing")] += 1
        if not args.dry_run:
            row = dict(source_completed[sample_id])
            row["budget_profile"] = args.run_id
            row["budget_reused_from"] = str(source_output)
            row["budget_reused_at"] = utc_timestamp()
            append_jsonl(output, row)
            budget_completed[sample_id] = row
        return

    if prompt_tokens > model.max_model_len:
        stats[(experiment, dataset_label, model.alias, overlength_reason)] += 1
        if not args.dry_run:
            row = make_terminal_skip_row(args, model, sample, prompt_tokens, overlength_reason)
            append_jsonl(output, row)
            budget_completed[sample_id] = row
        return

    stats[(experiment, dataset_label, model.alias, "pending_api")] += 1
    work.append(
        WorkItem(
            experiment=experiment,
            dataset_label=dataset_label,
            model=model,
            output=output,
            sample=sample,
            prompt=prompt,
            prompt_tokens=prompt_tokens,
            max_tokens=args.max_tokens,
            score_fn=score_fn,
        )
    )


def completed_rows(path: Path, caches: dict[Path, dict[str, dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    if path not in caches:
        completed: dict[str, dict[str, Any]] = {}
        for row in read_jsonl(path):
            sample_id = row.get("sample_id")
            if sample_id not in (None, "") and row.get("error") in (None, "", *TERMINAL_NONRETRY_ERRORS):
                completed[str(sample_id)] = row
        caches[path] = completed
    return caches[path]


def make_terminal_skip_row(
    args: argparse.Namespace,
    model: Any,
    sample: dict[str, Any],
    prompt_tokens: int,
    reason: str,
) -> dict[str, Any]:
    row = dict(sample)
    row.update(
        {
            "model": model.alias,
            "provider": args.provider,
            "api_model": model.api_model,
            "prediction": "",
            "latency_sec": 0.0,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": 0,
            "timestamp": utc_timestamp(),
            "score": 0.0,
            "metric": reason,
            "error": reason,
            "error_type": reason,
            "budget_profile": args.run_id,
        }
    )
    return row


def write_manifest(args: argparse.Namespace, selected: dict[str, list[dict[str, Any]]]) -> None:
    root = ROOT / args.manifest_root / args.run_id
    manifest_rows = []
    for experiment, rows in selected.items():
        for row in rows:
            manifest_rows.append(manifest_row(experiment, row))
    manifest_path = root / "budget_manifest.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        for row in manifest_rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def manifest_row(experiment: str, row: dict[str, Any]) -> dict[str, Any]:
    keep = {
        "experiment": experiment,
        "sample_id": row.get("sample_id"),
        "subtask": row.get("subtask"),
        "task": row.get("task"),
        "category": row.get("category"),
        "subset": row.get("subset"),
        "length_tokens_target": row.get("length_tokens_target"),
        "length_tokens_actual": row.get("length_tokens_actual"),
        "position_percent": row.get("position_percent"),
        "distribution": row.get("distribution"),
        "hop": row.get("hop"),
        "density": row.get("density"),
        "interference_type": row.get("interference_type"),
        "similarity": row.get("similarity"),
        "distance": row.get("distance"),
        "hops": row.get("hops"),
        "answer": row.get("answer"),
    }
    return {key: value for key, value in keep.items() if value not in (None, "")}


def write_plan(
    args: argparse.Namespace,
    selected: dict[str, list[dict[str, Any]]],
    work: list[WorkItem],
    stats: Counter[tuple[str, str, str, str]],
    models: list[Any],
    experiments: list[str],
) -> None:
    path = ROOT / args.plan_output
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = [
        {
            "experiment": experiment,
            "dataset_label": dataset_label,
            "model": model,
            "status": status,
            "count": count,
        }
        for (experiment, dataset_label, model, status), count in sorted(stats.items())
    ]
    selected_counts = {experiment: len(rows) for experiment, rows in selected.items()}
    payload = {
        "run_id": args.run_id,
        "experiments": experiments,
        "models": [asdict(model) for model in models],
        "selected_counts": selected_counts,
        "pending_api_calls": len(work),
        "stats": summary,
        "output_root": str(ROOT / args.output_root / args.run_id),
        "manifest": str(ROOT / args.manifest_root / args.run_id / "budget_manifest.jsonl"),
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)


def print_plan(
    selected: dict[str, list[dict[str, Any]]],
    work: list[WorkItem],
    stats: Counter[tuple[str, str, str, str]],
) -> None:
    print("Budget-core selected samples:")
    for experiment, rows in selected.items():
        print(f"  {experiment}: {len(rows)}")
    print("Budget-core model/sample status:")
    by_experiment_status: Counter[tuple[str, str]] = Counter()
    by_model_pending: Counter[str] = Counter()
    for (experiment, _dataset_label, model, status), count in stats.items():
        by_experiment_status[(experiment, status)] += count
        if status == "pending_api":
            by_model_pending[model] += count
    for (experiment, status), count in sorted(by_experiment_status.items()):
        print(f"  {experiment:<10} {status:<24} {count}")
    print("Pending API calls by model:")
    for model, count in sorted(by_model_pending.items()):
        print(f"  {model:<24} {count}")
    print(f"Total pending API calls: {len(work)}")


def select_grouped(
    rows: list[dict[str, Any]],
    key_fn: Callable[[dict[str, Any]], tuple[Any, ...]],
    limit_per_group: int,
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[key_fn(row)].append(row)
    selected: list[dict[str, Any]] = []
    for key in sorted(groups, key=lambda item: tuple(str(part) for part in item)):
        selected.extend(sorted(groups[key], key=sample_sort_key)[:limit_per_group])
    return selected


def stratified_select(
    rows: list[dict[str, Any]],
    key_fn: Callable[[dict[str, Any]], tuple[Any, ...]],
    limit: int,
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[key_fn(row)].append(row)
    ordered_keys = sorted(groups, key=lambda item: tuple(str(part) for part in item))
    for key in ordered_keys:
        groups[key].sort(key=sample_sort_key)

    selected: list[dict[str, Any]] = []
    offsets = Counter()
    while len(selected) < limit:
        progressed = False
        for key in ordered_keys:
            idx = offsets[key]
            if idx < len(groups[key]):
                selected.append(groups[key][idx])
                offsets[key] += 1
                progressed = True
                if len(selected) >= limit:
                    break
        if not progressed:
            break
    return selected


def sample_sort_key(row: dict[str, Any]) -> str:
    return str(row.get("sample_id") or row.get("id") or row.get("_id") or row.get("question") or "")


def pac_source_for_subset(source_data: str, filename: str) -> Path:
    source = Path(source_data)
    if source.is_file():
        return source
    return source / filename


def prompt_tokens_from_sample(sample: dict[str, Any], prompt: str, counter: TokenCounter) -> int:
    value = sample.get("length_tokens_actual") or sample.get("length_tokens_target") or sample.get("total_length")
    if value not in (None, ""):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            pass
    return counter.count(prompt)


def middle_truncate(prompt: str, max_tokens: int, counter: TokenCounter) -> str:
    marker = "\n\n[... middle truncated ...]\n\n"
    marker_tokens = counter.count(marker)
    keep = max(1, max_tokens - marker_tokens)
    left = keep // 2
    right = keep - left
    tokens = counter.encode(prompt)
    return counter.decode(tokens[:left]) + marker + counter.decode(tokens[-right:])


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    main()
