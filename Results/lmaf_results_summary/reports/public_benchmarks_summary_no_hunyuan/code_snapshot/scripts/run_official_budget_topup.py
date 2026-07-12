from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from lmaf.data.pac import adapt_external_pac_sample
from lmaf.utils.io import TERMINAL_NONRETRY_ERRORS, append_jsonl, collect_jsonl, iter_jsonl_paths, read_jsonl, utc_timestamp
from lmaf.utils.token_count import TokenCounter
from run_budget_core import (
    DEFAULT_LONGBENCH_TASKS,
    DEFAULT_PAC_SOURCE,
    DEFAULT_RULER_LENGTHS,
    manifest_row,
    middle_truncate,
    niah_group_key,
    pac_group_key,
    pac_source_for_subset,
    prompt_tokens_from_sample,
    select_grouped,
    select_longbench_samples,
    stratified_select,
)
from run_unfinished_fast import FRAMEWORK_MODELS, PAC_SUBSETS, WorkItem, run_parallel


DEFAULT_MODELS = ",".join(FRAMEWORK_MODELS)
DEFAULT_EXPERIMENTS = "niah,longbench,ruler,pac"
DEFAULT_RULER_TASKS = "niah,variable_tracking,common_words_extraction,freq_words_extraction"
DEFAULT_REUSE_ROOTS = ",".join(
    [
        "results/raw/budget_core",
        "results/raw/niah_batch",
        "results/raw/longbench_ruler_batch",
        "results/raw/pac_batch",
    ]
)


def main() -> None:
    args = parse_args()
    models = resolve_models(args.models)
    experiments = parse_csv(args.experiments)
    api_keys = resolve_api_keys(args)
    counter = TokenCounter(args.tokenizer)

    ensure_data(args, experiments)
    selected = select_samples(args, experiments, counter)
    write_manifest(args, selected)

    credit_candidates = load_credit_candidates(args, models, experiments)
    work, stats = build_topup_work(args, models, selected, credit_candidates, counter)
    write_plan(args, selected, work, stats, models, experiments)
    print_plan(selected, work, stats)

    if args.dry_run:
        return
    if args.provider == "siliconflow" and not api_keys:
        raise SystemExit("SILICONFLOW_API_KEY or SILICONFLOW_API_KEYS is required.")
    if not work:
        print("No API work remains. Output already reaches the per-model target counts.")
        return

    random.shuffle(work)
    run_parallel(args, work, api_keys)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the agreed budget counts on the official data, counting previous keepable "
            "successful results as credits and only calling the API for the remaining slots."
        )
    )
    parser.add_argument("--experiments", default=DEFAULT_EXPERIMENTS)
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

    parser.add_argument("--run-id", default="official_budget_topup_main")
    parser.add_argument("--output-root", default="results/raw/official_budget_topup")
    parser.add_argument("--manifest-root", default="data/processed/official_budget_topup")
    parser.add_argument("--plan-output", default="results/logs/official_budget_topup_plan.json")
    parser.add_argument("--reuse-roots", default=DEFAULT_REUSE_ROOTS)
    parser.add_argument(
        "--reuse-experiments",
        default="longbench,pac",
        help=(
            "Comma-separated experiments allowed to count previous successful results as credits. "
            "Default keeps NIAH/RULER fresh and reuses only LongBench/PAC."
        ),
    )
    parser.add_argument("--no-reuse-existing", action="store_true")
    parser.add_argument("--reuse-terminal-skips", action="store_true")
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument(
        "--generator-python",
        default=None,
        help=(
            "Python executable used for official data generation. "
            "Default auto-detects the homework conda env, then CONDA_PREFIX, then the current Python."
        ),
    )

    parser.add_argument("--niah-data", default="data/processed/official/niah")
    parser.add_argument("--niah-target", type=int, default=300)
    parser.add_argument("--niah-samples-per-cell", type=int, default=20)

    parser.add_argument("--longbench-data", default="data/processed/longbench_ruler_batch/framework_v2/longbench")
    parser.add_argument("--longbench-target", type=int, default=300)
    parser.add_argument("--longbench-tasks", default=DEFAULT_LONGBENCH_TASKS)
    parser.add_argument("--longbench-samples-per-task", type=int, default=50)
    parser.add_argument("--longbench-truncate", choices=["none", "middle"], default="none")

    parser.add_argument("--ruler-data", default="data/processed/official/ruler")
    parser.add_argument("--ruler-target", type=int, default=240)
    parser.add_argument("--ruler-tasks", default=DEFAULT_RULER_TASKS)
    parser.add_argument("--ruler-lengths", default=DEFAULT_RULER_LENGTHS)
    parser.add_argument("--ruler-samples-per-cell", type=int, default=20)

    parser.add_argument("--pac-source-data", default=DEFAULT_PAC_SOURCE)
    parser.add_argument("--pac-target", type=int, default=600)
    parser.add_argument("--pac-subsets", default="A,B,C,D")
    parser.add_argument("--pac-a-samples", type=int, default=200)
    parser.add_argument("--pac-b-samples", type=int, default=200)
    parser.add_argument("--pac-c-samples", type=int, default=100)
    parser.add_argument("--pac-d-samples", type=int, default=100)
    return parser.parse_args()


def ensure_data(args: argparse.Namespace, experiments: list[str]) -> None:
    if args.skip_generate:
        return
    generator_python = resolve_generator_python(args)
    if "niah" in experiments and not collect_jsonl(ROOT / args.niah_data):
        run_generator([str(generator_python), str(ROOT / "scripts" / "generate_official_niah.py")])
    if "ruler" in experiments and not collect_jsonl(ROOT / args.ruler_data):
        run_generator([str(generator_python), str(ROOT / "scripts" / "generate_official_ruler.py")])


def resolve_generator_python(args: argparse.Namespace) -> Path:
    if args.generator_python:
        return Path(args.generator_python)

    homework_python = Path.home() / ".conda" / "envs" / "homework" / "python.exe"
    if homework_python.exists():
        return homework_python

    conda_prefix = os.getenv("CONDA_PREFIX")
    if conda_prefix:
        conda_python = Path(conda_prefix) / ("python.exe" if os.name == "nt" else "bin/python")
        if conda_python.exists():
            return conda_python

    return Path(sys.executable)


def run_generator(command: list[str]) -> None:
    print("+ " + " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def select_samples(
    args: argparse.Namespace,
    experiments: list[str],
    counter: TokenCounter,
) -> dict[str, list[dict[str, Any]]]:
    selected: dict[str, list[dict[str, Any]]] = {}
    if "niah" in experiments:
        rows = collect_jsonl(ROOT / args.niah_data)
        selected["niah"] = select_grouped(rows, niah_group_key, args.niah_samples_per_cell)
    if "longbench" in experiments:
        selected["longbench"] = select_longbench_samples(args)
    if "ruler" in experiments:
        selected["ruler"] = select_ruler_samples(args)
    if "pac" in experiments:
        selected["pac"] = select_pac_samples(args, counter)
    return selected


def select_ruler_samples(args: argparse.Namespace) -> list[dict[str, Any]]:
    tasks = set(parse_csv(args.ruler_tasks))
    lengths = {int(item) for item in parse_csv(args.ruler_lengths)}
    rows = [
        row
        for row in collect_jsonl(ROOT / args.ruler_data)
        if row.get("subtask") in tasks and ruler_length(row) in lengths
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


def build_topup_work(
    args: argparse.Namespace,
    models: list[Any],
    selected: dict[str, list[dict[str, Any]]],
    credit_candidates: dict[tuple[str, str], list[tuple[Path, dict[str, Any]]]],
    counter: TokenCounter,
) -> tuple[list[WorkItem], Counter[tuple[str, str, str]]]:
    work: list[WorkItem] = []
    stats: Counter[tuple[str, str, str]] = Counter()
    run_root = ROOT / args.output_root / args.run_id
    reuse_experiments = set(parse_csv(args.reuse_experiments))

    for model in models:
        for experiment, rows in selected.items():
            output = run_root / experiment / f"{model.alias}.jsonl"
            target = target_for(args, experiment)
            completed = completed_rows(output, allow_credit_rows=experiment in reuse_experiments)
            stats[(experiment, model.alias, "target")] += target
            stats[(experiment, model.alias, "already_in_output")] += len(completed)

            if not args.no_reuse_existing and experiment in reuse_experiments:
                needed_credit = max(0, target - len(completed))
                copied = copy_credit_rows(
                    args,
                    output,
                    completed,
                    credit_candidates.get((experiment, model.alias), []),
                    needed_credit,
                    experiment,
                    model.alias,
                )
                stats[(experiment, model.alias, "reused_credit")] += copied
            elif experiment not in reuse_experiments:
                stats[(experiment, model.alias, "reuse_disabled_fresh_run")] += 1

            needed_new = max(0, target - len(completed))
            if needed_new:
                added, skipped = add_new_work(args, work, output, completed, rows, experiment, model, needed_new, counter)
                stats[(experiment, model.alias, "pending_api")] += added
                stats[(experiment, model.alias, "terminal_skip")] += skipped
                unfilled = max(0, target - len(completed) - added)
                if unfilled:
                    stats[(experiment, model.alias, "unfilled_no_samples")] += unfilled

    if args.stop_after is not None:
        work = work[: args.stop_after]
    return work, stats


def copy_credit_rows(
    args: argparse.Namespace,
    output: Path,
    completed: dict[str, dict[str, Any]],
    candidates: list[tuple[Path, dict[str, Any]]],
    limit: int,
    experiment: str,
    model_alias: str,
) -> int:
    copied = 0
    for source_path, row in candidates:
        if copied >= limit:
            break
        sample_id = str(row.get("sample_id") or "")
        if not sample_id or sample_id in completed:
            continue
        out_row = dict(row)
        out_row.update(
            {
                "budget_profile": args.run_id,
                "budget_credit_experiment": experiment,
                "budget_credit_model": model_alias,
                "budget_reused_from": str(source_path),
                "budget_reused_at": utc_timestamp(),
            }
        )
        copied += 1
        completed[sample_id] = out_row
        if not args.dry_run:
            append_jsonl(output, out_row)
    return copied


def add_new_work(
    args: argparse.Namespace,
    work: list[WorkItem],
    output: Path,
    completed: dict[str, dict[str, Any]],
    rows: list[dict[str, Any]],
    experiment: str,
    model: Any,
    limit: int,
    counter: TokenCounter,
) -> tuple[int, int]:
    added = 0
    skipped = 0
    for sample in rows:
        if added + skipped >= limit:
            break
        sample_id = str(sample["sample_id"])
        if sample_id in completed:
            continue
        prompt, prompt_tokens, sample_for_run = prompt_for_sample(args, sample, experiment, model, counter)
        if prompt_tokens > model.max_model_len:
            row = make_skip_row(args, sample_for_run, model, prompt_tokens, "skipped_overlength")
            completed[sample_id] = row
            skipped += 1
            if not args.dry_run:
                append_jsonl(output, row)
            continue
        work.append(
            WorkItem(
                experiment=experiment,
                dataset_label=f"official_topup_{experiment}",
                model=model,
                output=output,
                sample=sample_for_run,
                prompt=prompt,
                prompt_tokens=prompt_tokens,
                max_tokens=args.max_tokens,
                score_fn=experiment,
            )
        )
        completed[sample_id] = sample_for_run
        added += 1
    return added, skipped


def prompt_for_sample(
    args: argparse.Namespace,
    sample: dict[str, Any],
    experiment: str,
    model: Any,
    counter: TokenCounter,
) -> tuple[str, int, dict[str, Any]]:
    prompt = str(sample["prompt"])
    prompt_tokens = prompt_tokens_from_sample(sample, prompt, counter)
    sample_for_run = sample
    if experiment == "longbench":
        prompt_tokens = counter.count(prompt)
        if prompt_tokens > model.max_model_len and args.longbench_truncate == "middle":
            prompt = middle_truncate(prompt, model.max_model_len, counter)
            prompt_tokens = counter.count(prompt)
            sample_for_run = dict(sample)
            sample_for_run["truncation"] = "middle"
    elif experiment == "ruler":
        prompt_tokens = counter.count(prompt)
    return prompt, prompt_tokens, sample_for_run


def make_skip_row(args: argparse.Namespace, sample: dict[str, Any], model: Any, prompt_tokens: int, reason: str) -> dict[str, Any]:
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


def load_credit_candidates(
    args: argparse.Namespace,
    models: list[Any],
    experiments: list[str],
) -> dict[tuple[str, str], list[tuple[Path, dict[str, Any]]]]:
    aliases = {model.alias for model in models}
    experiment_set = set(experiments)
    required = {
        (experiment, model.alias): target_for(args, experiment)
        for experiment in experiments
        for model in models
    }
    counts: Counter[tuple[str, str]] = Counter()
    out: dict[tuple[str, str], list[tuple[Path, dict[str, Any]]]] = defaultdict(list)
    seen: set[tuple[str, str, str]] = set()
    for root_value in parse_csv(args.reuse_roots):
        root = ROOT / root_value
        if not root.exists():
            continue
        for path in iter_jsonl_paths(root):
            for row in read_jsonl(path):
                experiment = infer_experiment(row, path)
                model = infer_model(row, path, aliases)
                sample_id = str(row.get("sample_id") or "")
                if experiment not in experiment_set or model not in aliases or not sample_id:
                    continue
                pair = (experiment, model)
                if counts[pair] >= required[pair]:
                    continue
                if not keepable_result(row, args.reuse_terminal_skips):
                    continue
                key = (experiment, model, sample_id)
                if key in seen:
                    continue
                seen.add(key)
                out[pair].append((path, row))
                counts[pair] += 1
                if all(counts[pair_key] >= target for pair_key, target in required.items()):
                    for out_key in out:
                        out[out_key].sort(key=lambda item: (str(item[0]), str(item[1].get("sample_id") or "")))
                    return out
    for key in out:
        out[key].sort(key=lambda item: (str(item[0]), str(item[1].get("sample_id") or "")))
    return out


def ruler_length(row: dict[str, Any]) -> int:
    value = row.get("official_length_dir") or row.get("length_tokens_target") or 0
    return int(value)


def keepable_result(row: dict[str, Any], reuse_terminal_skips: bool) -> bool:
    error = row.get("error")
    if error in (None, ""):
        return row.get("prediction") not in (None, "")
    return reuse_terminal_skips and error in TERMINAL_NONRETRY_ERRORS


def completed_rows(path: Path, allow_credit_rows: bool = True) -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        if not allow_credit_rows and row.get("budget_reused_from"):
            continue
        sample_id = row.get("sample_id")
        if sample_id in (None, ""):
            continue
        if row.get("error") in (None, "", *TERMINAL_NONRETRY_ERRORS):
            completed[str(sample_id)] = row
    return completed


def infer_experiment(row: dict[str, Any], path: Path) -> str:
    value = str(row.get("experiment") or "").lower()
    if value in {"niah", "longbench", "ruler", "pac"}:
        return value
    parts = [part.lower() for part in path.parts]
    for experiment in ("longbench", "niah", "ruler", "pac"):
        if any(experiment in part for part in parts):
            return experiment
    return ""


def infer_model(row: dict[str, Any], path: Path, aliases: set[str]) -> str:
    value = str(row.get("model") or "")
    if value in aliases:
        return value
    if path.stem in aliases:
        return path.stem
    return ""


def target_for(args: argparse.Namespace, experiment: str) -> int:
    return {
        "niah": args.niah_target,
        "longbench": args.longbench_target,
        "ruler": args.ruler_target,
        "pac": args.pac_target,
    }[experiment]


def write_manifest(args: argparse.Namespace, selected: dict[str, list[dict[str, Any]]]) -> None:
    root = ROOT / args.manifest_root / args.run_id
    rows = []
    for experiment, samples in selected.items():
        for sample in samples:
            rows.append(manifest_row(experiment, sample))
    path = root / "budget_manifest.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_plan(
    args: argparse.Namespace,
    selected: dict[str, list[dict[str, Any]]],
    work: list[WorkItem],
    stats: Counter[tuple[str, str, str]],
    models: list[Any],
    experiments: list[str],
) -> None:
    path = ROOT / args.plan_output
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": args.run_id,
        "experiments": experiments,
        "models": [asdict(model) for model in models],
        "selected_counts": {experiment: len(rows) for experiment, rows in selected.items()},
        "pending_api_calls": len(work),
        "output_root": str(ROOT / args.output_root / args.run_id),
        "manifest": str(ROOT / args.manifest_root / args.run_id / "budget_manifest.jsonl"),
        "stats": [
            {"experiment": exp, "model": model, "status": status, "count": count}
            for (exp, model, status), count in sorted(stats.items())
        ],
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)


def print_plan(
    selected: dict[str, list[dict[str, Any]]],
    work: list[WorkItem],
    stats: Counter[tuple[str, str, str]],
) -> None:
    print("Official budget top-up selected samples:")
    for experiment, rows in selected.items():
        print(f"  {experiment}: {len(rows)}")
    print("Per-experiment status:")
    by_status = Counter()
    by_model_pending = Counter()
    for (experiment, model, status), count in stats.items():
        by_status[(experiment, status)] += count
        if status == "pending_api":
            by_model_pending[model] += count
    for (experiment, status), count in sorted(by_status.items()):
        print(f"  {experiment:<10} {status:<20} {count}")
    print("Pending API calls by model:")
    for model, count in sorted(by_model_pending.items()):
        print(f"  {model:<24} {count}")
    print(f"Total pending API calls: {len(work)}")


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


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    main()
