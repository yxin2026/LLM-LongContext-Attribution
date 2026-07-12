from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ModelPlan:
    alias: str
    api_model: str
    max_model_len: int
    priority: int
    enable_thinking: bool = False
    note: str = ""


FRAMEWORK_MODELS: dict[str, ModelPlan] = {
    "qwen35_9b": ModelPlan("qwen35_9b", "Qwen/Qwen3.5-9B", 65536, 1, note="Framework V2 first priority"),
    "qwen3_8b": ModelPlan("qwen3_8b", "Qwen/Qwen3-8B", 32768, 1, note="Generation baseline"),
    "qwen35_27b": ModelPlan("qwen35_27b", "Qwen/Qwen3.5-27B", 32768, 2),
    "qwen35_35b_a3b": ModelPlan("qwen35_35b_a3b", "Qwen/Qwen3.5-35B-A3B", 65536, 3),
    "qwen35_122b_a10b": ModelPlan("qwen35_122b_a10b", "Qwen/Qwen3.5-122B-A10B", 32768, 4),
    "hunyuan_a13b": ModelPlan("hunyuan_a13b", "tencent/Hunyuan-A13B-Instruct", 65536, 3),
    "seed_oss_36b": ModelPlan("seed_oss_36b", "ByteDance-Seed/Seed-OSS-36B-Instruct", 32768, 3),
    "qwen3_14b_no_thinking": ModelPlan("qwen3_14b_no_thinking", "Qwen/Qwen3-14B", 32768, 1),
    "qwen3_14b_thinking": ModelPlan("qwen3_14b_thinking", "Qwen/Qwen3-14B", 32768, 1, enable_thinking=True),
}


MODEL_PROFILES = {
    "minimal": ["qwen35_9b", "qwen3_8b"],
    "single_card": [
        "qwen35_9b",
        "qwen3_8b",
        "qwen35_27b",
        "qwen35_35b_a3b",
        "qwen35_122b_a10b",
    ],
    "all_framework": list(FRAMEWORK_MODELS),
}


DEFAULT_LONGBENCH_TASKS = (
    "narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,gov_report,qmsum,multi_news"
)
DEFAULT_RULER_TASKS = "niah,variable_tracking,common_words_extraction,freq_words_extraction,qa_squad,qa_hotpotqa"


BATCH_SUITES = {
    "smoke": {
        "longbench_tasks": "narrativeqa,hotpotqa,gov_report",
        "longbench_sample_limit": 2,
        "ruler_tasks": "niah,variable_tracking",
        "ruler_lengths": "4096",
        "ruler_samples_per_cell": 1,
    },
    "framework_v2": {
        "longbench_tasks": DEFAULT_LONGBENCH_TASKS,
        "longbench_sample_limit": 200,
        "ruler_tasks": DEFAULT_RULER_TASKS,
        "ruler_lengths": "4096,16384,32768",
        "ruler_samples_per_cell": 50,
    },
}


def main() -> None:
    args = parse_args()
    suite = dict(BATCH_SUITES[args.suite])
    selected_models = resolve_models(args)
    selected_experiments = _parse_strs(args.experiments)
    validate_args(args, selected_experiments)

    longbench_tasks = args.longbench_tasks or suite["longbench_tasks"]
    if args.longbench_full:
        longbench_sample_limit = None
    else:
        longbench_sample_limit = (
            args.longbench_sample_limit if args.longbench_sample_limit is not None else suite["longbench_sample_limit"]
        )
    ruler_tasks = args.ruler_tasks or suite["ruler_tasks"]
    ruler_lengths = args.ruler_lengths or suite["ruler_lengths"]
    ruler_samples_per_cell = (
        args.ruler_samples_per_cell if args.ruler_samples_per_cell is not None else suite["ruler_samples_per_cell"]
    )

    env = os.environ.copy()
    if args.api_key:
        if args.provider == "siliconflow":
            env["SILICONFLOW_API_KEY"] = args.api_key
        elif args.provider == "local":
            env["LOCAL_OPENAI_API_KEY"] = args.api_key
        else:
            env["OPENAI_API_KEY"] = args.api_key

    if args.provider == "siliconflow" and not env.get("SILICONFLOW_API_KEY") and not args.dry_run and not args.generate_only:
        raise SystemExit("SILICONFLOW_API_KEY is required. Set it in the environment or pass --api-key.")

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    data_root = Path(args.data_root) / args.suite
    result_root = Path(args.result_root) / args.suite / run_id
    aggregate_root = Path(args.aggregate_root) / args.suite / run_id
    figure_root = Path(args.figure_root) / args.suite / run_id

    metadata = {
        "run_id": run_id,
        "provider": args.provider,
        "suite": args.suite,
        "profile": args.profile,
        "experiments": selected_experiments,
        "models": [model.__dict__ for model in selected_models],
        "longbench_tasks": longbench_tasks,
        "longbench_sample_limit": longbench_sample_limit,
        "ruler_tasks": ruler_tasks,
        "ruler_lengths": ruler_lengths,
        "ruler_samples_per_cell": ruler_samples_per_cell,
        "data_root": str(data_root),
        "result_root": str(result_root),
        "aggregate_root": str(aggregate_root),
        "figure_root": str(figure_root),
        "dry_run": args.dry_run,
    }

    if "longbench" in selected_experiments:
        run_longbench(args, env, selected_models, data_root, result_root, longbench_tasks, longbench_sample_limit)
    if "ruler" in selected_experiments:
        run_ruler(args, env, selected_models, data_root, result_root, ruler_tasks, ruler_lengths, ruler_samples_per_cell)

    metadata_path = Path(args.metadata_path) if args.metadata_path else result_root / "longbench_ruler_batch_metadata.json"
    if not args.dry_run:
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2, sort_keys=True)
    print(f"LongBench/RULER batch metadata: {metadata_path}")

    if not args.generate_only and not args.skip_aggregate:
        postprocess(args, env, selected_experiments, result_root, aggregate_root, figure_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate/prepare and run LongBench plus RULER for multiple models.")
    parser.add_argument("--suite", choices=sorted(BATCH_SUITES), default="framework_v2")
    parser.add_argument("--profile", choices=sorted(MODEL_PROFILES), default="minimal")
    parser.add_argument("--models", default=None, help="Comma-separated model aliases or exact API model names.")
    parser.add_argument("--experiments", default="longbench,ruler", help="Comma-separated: longbench,ruler")
    parser.add_argument("--provider", choices=["siliconflow", "local", "custom"], default="siliconflow")
    parser.add_argument("--api-key", default=None, help="Optional. Prefer SILICONFLOW_API_KEY environment variable.")
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--data-root", default="data/processed/longbench_ruler_batch")
    parser.add_argument("--result-root", default="results/raw/longbench_ruler_batch")
    parser.add_argument("--aggregate-root", default="results/aggregate/longbench_ruler_batch")
    parser.add_argument("--figure-root", default="results/figures/longbench_ruler_batch")
    parser.add_argument("--metadata-path", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--longbench-tasks", default=None)
    parser.add_argument("--longbench-sample-limit", type=int, default=None)
    parser.add_argument("--longbench-full", action="store_true", help="Use every available LongBench test row.")
    parser.add_argument("--longbench-repo", default="external/LongBench")
    parser.add_argument("--longbench-max-tokens", type=int, default=512)
    parser.add_argument("--longbench-truncate", choices=["none", "middle"], default="none")
    parser.add_argument("--ruler-tasks", default=None)
    parser.add_argument("--ruler-lengths", default=None)
    parser.add_argument("--ruler-samples-per-cell", type=int, default=None)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--ruler-max-tokens", type=int, default=512)
    parser.add_argument("--enable-thinking", action="store_true", help="Force thinking on for every model.")
    parser.add_argument("--thinking-budget", type=int, default=None)
    parser.add_argument("--generate-only", action="store_true", help="Only prepare LongBench/generate RULER data.")
    parser.add_argument("--run-only", action="store_true", help="Skip data preparation/generation and only call models.")
    parser.add_argument("--skip-aggregate", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run_longbench(
    args: argparse.Namespace,
    env: dict[str, str],
    models: list[ModelPlan],
    data_root: Path,
    result_root: Path,
    tasks: str,
    sample_limit: int | None,
) -> None:
    data_dir = data_root / "longbench"
    if not args.run_only:
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "run_longbench.py"),
            "--prepare-only",
            "--tasks",
            tasks,
            "--longbench-repo",
            args.longbench_repo,
            "--output",
            str(data_dir),
        ]
        if sample_limit is not None:
            cmd.extend(["--sample-limit", str(sample_limit)])
        if args.tokenizer:
            cmd.extend(["--tokenizer", args.tokenizer])
        run_command(cmd, env=env, dry_run=args.dry_run)

    if args.generate_only:
        return

    for model in models:
        output = result_root / "longbench" / f"{model.alias}.jsonl"
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "run_longbench.py"),
            "--provider",
            args.provider,
            "--model",
            request_model_name(model),
            "--tasks",
            tasks,
            "--input",
            str(data_dir),
            "--output",
            str(output),
            "--max-model-len",
            str(model.max_model_len),
            "--truncate",
            args.longbench_truncate,
            "--max-tokens",
            str(args.longbench_max_tokens),
            "--timeout",
            str(args.timeout),
            "--resume",
        ]
        add_common_inference_args(cmd, args, model)
        run_command(cmd, env=env, dry_run=args.dry_run)


def run_ruler(
    args: argparse.Namespace,
    env: dict[str, str],
    models: list[ModelPlan],
    data_root: Path,
    result_root: Path,
    tasks: str,
    lengths: str,
    samples_per_cell: int,
) -> None:
    data_dir = data_root / "ruler"
    if not args.run_only:
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "run_ruler.py"),
            "--generate-only",
            "--tasks",
            tasks,
            "--lengths",
            lengths,
            "--samples-per-cell",
            str(samples_per_cell),
            "--output",
            str(data_dir),
        ]
        if args.tokenizer:
            cmd.extend(["--tokenizer", args.tokenizer])
        run_command(cmd, env=env, dry_run=args.dry_run)

    if args.generate_only:
        return

    for model in models:
        output = result_root / "ruler" / f"{model.alias}.jsonl"
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "run_ruler.py"),
            "--provider",
            args.provider,
            "--model",
            request_model_name(model),
            "--input",
            str(data_dir),
            "--output",
            str(output),
            "--max-model-len",
            str(model.max_model_len),
            "--max-tokens",
            str(args.ruler_max_tokens),
            "--timeout",
            str(args.timeout),
            "--resume",
        ]
        add_common_inference_args(cmd, args, model)
        run_command(cmd, env=env, dry_run=args.dry_run)


def postprocess(
    args: argparse.Namespace,
    env: dict[str, str],
    experiments: list[str],
    result_root: Path,
    aggregate_root: Path,
    figure_root: Path,
) -> None:
    for experiment in experiments:
        raw_dir = result_root / experiment
        aggregate_csv = aggregate_root / f"{experiment}.csv"
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "aggregate_results.py"),
            "--input",
            str(raw_dir),
            "--experiment",
            experiment,
            "--output",
            str(aggregate_csv),
        ]
        run_command(cmd, env=env, dry_run=args.dry_run)
        if args.skip_plots:
            continue
        if experiment == "longbench":
            plot = "longbench_score_bar"
            figure = figure_root / "longbench_score_bar.png"
        elif experiment == "ruler":
            plot = "ruler_effective_context"
            figure = figure_root / "ruler_effective_context.png"
        else:
            continue
        plot_cmd = [
            sys.executable,
            str(ROOT / "scripts" / "plot_results.py"),
            "--input",
            str(aggregate_csv),
            "--plot",
            plot,
            "--output",
            str(figure),
        ]
        run_command(plot_cmd, env=env, dry_run=args.dry_run)


def add_common_inference_args(cmd: list[str], args: argparse.Namespace, model: ModelPlan) -> None:
    if args.endpoint:
        cmd.extend(["--endpoint", args.endpoint])
    if model.enable_thinking or args.enable_thinking:
        cmd.append("--enable-thinking")
    if args.thinking_budget is not None:
        cmd.extend(["--thinking-budget", str(args.thinking_budget)])
    if args.tokenizer:
        cmd.extend(["--tokenizer", args.tokenizer])


def resolve_models(args: argparse.Namespace) -> list[ModelPlan]:
    names = [item.strip() for item in (args.models or ",".join(MODEL_PROFILES[args.profile])).split(",") if item.strip()]
    models: list[ModelPlan] = []
    for name in names:
        if name in FRAMEWORK_MODELS:
            models.append(FRAMEWORK_MODELS[name])
        else:
            safe_alias = (
                name.replace("/", "_")
                .replace(":", "_")
                .replace(" ", "_")
                .replace(".", "_")
                .replace("-", "_")
                .lower()
            )
            models.append(ModelPlan(alias=safe_alias, api_model=name, max_model_len=32768, priority=99, note="custom model"))
    return models


def request_model_name(model: ModelPlan) -> str:
    if model.alias in FRAMEWORK_MODELS:
        return model.alias
    return model.api_model


def validate_args(args: argparse.Namespace, experiments: list[str]) -> None:
    unknown = sorted(set(experiments) - {"longbench", "ruler"})
    if unknown:
        raise SystemExit(f"Unsupported experiments: {', '.join(unknown)}")
    if not experiments:
        raise SystemExit("--experiments must include longbench, ruler, or both")
    if args.generate_only and args.run_only:
        raise SystemExit("--generate-only and --run-only cannot be used together")
    if args.longbench_full and args.longbench_sample_limit is not None:
        raise SystemExit("--longbench-full cannot be combined with --longbench-sample-limit")
    if args.longbench_sample_limit is not None and args.longbench_sample_limit <= 0:
        raise SystemExit("--longbench-sample-limit must be positive when provided")
    if args.ruler_samples_per_cell is not None and args.ruler_samples_per_cell <= 0:
        raise SystemExit("--ruler-samples-per-cell must be positive when provided")
    if args.timeout <= 0:
        raise SystemExit("--timeout must be positive")
    if args.longbench_max_tokens <= 0 or args.ruler_max_tokens <= 0:
        raise SystemExit("--longbench-max-tokens and --ruler-max-tokens must be positive")


def run_command(cmd: list[str], env: dict[str, str], dry_run: bool) -> None:
    printable = [redact(item) for item in cmd]
    print("+ " + " ".join(printable))
    if dry_run:
        return
    completed = subprocess.run(cmd, cwd=ROOT, env=env, text=True, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def _parse_strs(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def redact(value: str) -> str:
    if value.startswith("sk-"):
        return "sk-***"
    return value


if __name__ == "__main__":
    main()
