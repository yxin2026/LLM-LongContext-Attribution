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


PAC_SUBSETS = {
    "A": ("A_position", "subset_A.jsonl"),
    "B": ("B_interference", "subset_B.jsonl"),
    "C": ("C_overlap", "subset_C.jsonl"),
    "D": ("D_multihop", "subset_D.jsonl"),
}


def main() -> None:
    args = parse_args()
    selected_models = resolve_models(args)
    selected_subsets = resolve_subsets(args.subsets)
    validate_args(args, selected_subsets)

    env = os.environ.copy()
    if args.api_key:
        if args.provider == "siliconflow":
            env["SILICONFLOW_API_KEY"] = args.api_key
        elif args.provider == "local":
            env["LOCAL_OPENAI_API_KEY"] = args.api_key
        else:
            env["OPENAI_API_KEY"] = args.api_key

    if args.provider == "siliconflow" and not env.get("SILICONFLOW_API_KEY") and not args.dry_run:
        raise SystemExit("SILICONFLOW_API_KEY is required. Set it in the environment or pass --api-key.")

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    result_root = Path(args.result_root) / run_id
    aggregate_root = Path(args.aggregate_root) / run_id
    figure_root = Path(args.figure_root) / run_id
    metadata = {
        "run_id": run_id,
        "provider": args.provider,
        "profile": args.profile,
        "subsets": selected_subsets,
        "models": [model.__dict__ for model in selected_models],
        "source_data": args.source_data,
        "result_root": str(result_root),
        "aggregate_root": str(aggregate_root),
        "figure_root": str(figure_root),
        "sample_limit": args.sample_limit,
        "dry_run": args.dry_run,
    }

    for subset_key in selected_subsets:
        subtask, _ = PAC_SUBSETS[subset_key]
        source = source_for_subset(args.source_data, subset_key)
        for model in selected_models:
            output = result_root / subset_key / f"{model.alias}.jsonl"
            cmd = [
                sys.executable,
                str(ROOT / "scripts" / "run_pac.py"),
                "--subset",
                subtask,
                "--provider",
                args.provider,
                "--model",
                request_model_name(model),
                "--input-format",
                "pac-test",
                "--input",
                str(source),
                "--output",
                str(output),
                "--max-model-len",
                str(model.max_model_len),
                "--max-tokens",
                str(args.max_tokens),
                "--timeout",
                str(args.timeout),
                "--resume",
            ]
            add_common_inference_args(cmd, args, model)
            if args.sample_limit is not None:
                cmd.extend(["--sample-limit", str(args.sample_limit)])
            run_command(cmd, env=env, dry_run=args.dry_run)

    metadata_path = Path(args.metadata_path) if args.metadata_path else result_root / "pac_batch_metadata.json"
    if not args.dry_run:
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2, sort_keys=True)
    print(f"PAC batch metadata: {metadata_path}")

    if not args.skip_aggregate:
        postprocess(args, env, result_root, aggregate_root, figure_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PAC-Test A/B/C/D for multiple models.")
    parser.add_argument(
        "--source-data",
        default=os.getenv("PAC_TEST_DATA_DIR", "data/raw/pac_test"),
        help="Directory containing subset_A/B/C/D.jsonl, or PAC-Test_complete.jsonl.",
    )
    parser.add_argument("--subsets", default="A,B,C,D", help="Comma-separated subset keys: A,B,C,D")
    parser.add_argument("--profile", choices=sorted(MODEL_PROFILES), default="minimal")
    parser.add_argument("--models", default=None, help="Comma-separated model aliases or exact API model names.")
    parser.add_argument("--provider", choices=["siliconflow", "local", "custom"], default="siliconflow")
    parser.add_argument("--api-key", default=None, help="Optional. Prefer SILICONFLOW_API_KEY environment variable.")
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--result-root", default="results/raw/pac_batch")
    parser.add_argument("--aggregate-root", default="results/aggregate/pac_batch")
    parser.add_argument("--figure-root", default="results/figures/pac_batch")
    parser.add_argument("--metadata-path", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--sample-limit", type=int, default=None, help="Optional per-subset inference cap for smoke tests.")
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--enable-thinking", action="store_true", help="Force thinking on for every model.")
    parser.add_argument("--thinking-budget", type=int, default=None)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--skip-aggregate", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def postprocess(args: argparse.Namespace, env: dict[str, str], result_root: Path, aggregate_root: Path, figure_root: Path) -> None:
    aggregate_csv = aggregate_root / "pac.csv"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "aggregate_results.py"),
        "--input",
        str(result_root),
        "--experiment",
        "pac",
        "--output",
        str(aggregate_csv),
    ]
    run_command(cmd, env=env, dry_run=args.dry_run)
    if args.skip_plots:
        return
    plots = [
        ("pac_A_position_curve", "pac_A_position_curve.png"),
        ("pac_B_density_curve", "pac_B_density_curve.png"),
        ("pac_C_confusion_matrix", "pac_C_confusion_matrix.png"),
        ("pac_D_multihop_decay", "pac_D_multihop_decay.png"),
    ]
    for plot, filename in plots:
        plot_cmd = [
            sys.executable,
            str(ROOT / "scripts" / "plot_results.py"),
            "--input",
            str(aggregate_csv),
            "--plot",
            plot,
            "--output",
            str(figure_root / filename),
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


def source_for_subset(source_data: str, subset_key: str) -> Path:
    source = Path(source_data)
    if source.is_file():
        return source
    _, filename = PAC_SUBSETS[subset_key]
    return source / filename


def resolve_subsets(value: str) -> list[str]:
    subsets = [item.strip().upper() for item in value.split(",") if item.strip()]
    unknown = sorted(set(subsets) - set(PAC_SUBSETS))
    if unknown:
        raise SystemExit(f"Unsupported PAC subsets: {', '.join(unknown)}")
    if not subsets:
        raise SystemExit("--subsets must include at least one of A,B,C,D")
    return subsets


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


def validate_args(args: argparse.Namespace, subsets: list[str]) -> None:
    if args.sample_limit is not None and args.sample_limit <= 0:
        raise SystemExit("--sample-limit must be positive when provided")
    if args.timeout <= 0:
        raise SystemExit("--timeout must be positive")
    if args.max_tokens <= 0:
        raise SystemExit("--max-tokens must be positive")
    if args.dry_run:
        return
    source = Path(args.source_data)
    if not source.exists():
        raise SystemExit(f"PAC source data path does not exist: {source}")
    for subset in subsets:
        candidate = source_for_subset(args.source_data, subset)
        if not candidate.exists():
            raise SystemExit(f"PAC source file does not exist for subset {subset}: {candidate}")


def run_command(cmd: list[str], env: dict[str, str], dry_run: bool) -> None:
    printable = [redact(item) for item in cmd]
    print("+ " + " ".join(printable))
    if dry_run:
        return
    completed = subprocess.run(cmd, cwd=ROOT, env=env, text=True, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def redact(value: str) -> str:
    if value.startswith("sk-"):
        return "sk-***"
    return value


if __name__ == "__main__":
    main()
