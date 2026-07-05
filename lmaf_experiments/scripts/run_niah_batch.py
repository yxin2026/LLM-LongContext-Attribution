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
    "deepseek_r1_distill_qwen_14b": ModelPlan(
        "deepseek_r1_distill_qwen_14b",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        32768,
        1,
        note="Distillation baseline",
    ),
    "qwen35_27b": ModelPlan("qwen35_27b", "Qwen/Qwen3.5-27B", 32768, 2),
    "gemma4_31b": ModelPlan("gemma4_31b", "google/gemma-4-31B-it", 32768, 2),
    "qwen35_35b_a3b": ModelPlan("qwen35_35b_a3b", "Qwen/Qwen3.5-35B-A3B", 65536, 3),
    "gemma4_26b_a4b": ModelPlan("gemma4_26b_a4b", "google/gemma-4-26B-A4B-it", 65536, 3),
    "qwen35_122b_a10b": ModelPlan("qwen35_122b_a10b", "Qwen/Qwen3.5-122B-A10B", 32768, 4),
    "hunyuan_a13b": ModelPlan("hunyuan_a13b", "tencent/Hunyuan-A13B-Instruct", 65536, 3),
    "seed_oss_36b": ModelPlan("seed_oss_36b", "ByteDance-Seed/Seed-OSS-36B-Instruct", 32768, 3),
    "qwen3_14b_no_thinking": ModelPlan("qwen3_14b_no_thinking", "Qwen/Qwen3-14B", 32768, 1),
    "qwen3_14b_thinking": ModelPlan("qwen3_14b_thinking", "Qwen/Qwen3-14B", 32768, 1, enable_thinking=True),
}

MODEL_PROFILES = {
    "minimal": ["qwen35_9b", "qwen3_8b", "deepseek_r1_distill_qwen_14b"],
    "single_card": [
        "qwen35_9b",
        "qwen3_8b",
        "deepseek_r1_distill_qwen_14b",
        "qwen35_27b",
        "gemma4_31b",
        "qwen35_35b_a3b",
        "gemma4_26b_a4b",
        "qwen35_122b_a10b",
    ],
    "all_framework": list(FRAMEWORK_MODELS),
}

NIAH_SUITES = {
    "smoke": [
        {
            "name": "single_smoke_4k",
            "variants": "single",
            "lengths": "4096",
            "positions": "50",
            "samples_per_cell": 2,
        }
    ],
    "fast16k": [
        {
            "name": "single_16k",
            "variants": "single",
            "lengths": "16384",
            "positions": "10,50,90",
            "samples_per_cell": 50,
        }
    ],
    "framework_v2": [
        {
            "name": "single",
            "variants": "single",
            "lengths": "4096,16384,32768,65536",
            "positions": "10,50,90",
            "samples_per_cell": 50,
        },
        {
            "name": "multi",
            "variants": "multi",
            "lengths": "16384,32768",
            "distributions": "uniform,clustered",
            "samples_per_cell": 50,
        },
        {
            "name": "sequential",
            "variants": "sequential",
            "lengths": "16384,32768",
            "samples_per_cell": 50,
        },
    ],
}


def main() -> None:
    args = parse_args()
    selected_models = resolve_models(args)
    suite = NIAH_SUITES[args.suite]
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
    metadata = {
        "run_id": run_id,
        "provider": args.provider,
        "suite": args.suite,
        "profile": args.profile,
        "models": [model.__dict__ for model in selected_models],
        "data_root": str(data_root),
        "result_root": str(result_root),
        "dry_run": args.dry_run,
    }

    if not args.run_only:
        for spec in suite:
            output = data_root / spec["name"]
            cmd = [
                sys.executable,
                str(ROOT / "scripts" / "run_niah.py"),
                "--generate-only",
                "--variants",
                spec["variants"],
                "--lengths",
                spec["lengths"],
                "--samples-per-cell",
                str(args.samples_per_cell or spec["samples_per_cell"]),
                "--output",
                str(output),
            ]
            if spec.get("positions"):
                cmd.extend(["--positions", spec["positions"]])
            if spec.get("distributions"):
                cmd.extend(["--distributions", spec["distributions"]])
            run_command(cmd, env=env, dry_run=args.dry_run)

    if not args.generate_only:
        inputs = [data_root / spec["name"] for spec in suite]
        for model in selected_models:
            output = result_root / f"{model.alias}.jsonl"
            cmd = [
                sys.executable,
                str(ROOT / "scripts" / "run_niah.py"),
                "--provider",
                args.provider,
                "--model",
                model.api_model if args.use_api_model_name else model.alias,
                "--input",
                str(data_root),
                "--output",
                str(output),
                "--max-model-len",
                str(model.max_model_len),
                "--timeout",
                str(args.timeout),
                "--resume",
            ]
            if args.endpoint:
                cmd.extend(["--endpoint", args.endpoint])
            if model.enable_thinking or args.enable_thinking:
                cmd.append("--enable-thinking")
            if args.thinking_budget is not None:
                cmd.extend(["--thinking-budget", str(args.thinking_budget)])
            if args.max_tokens != 512:
                cmd.extend(["--max-tokens", str(args.max_tokens)])
            run_command(cmd, env=env, dry_run=args.dry_run)

    metadata_path = Path(args.metadata_path) if args.metadata_path else result_root / "niah_batch_metadata.json"
    if not args.dry_run:
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2, sort_keys=True)
    print(f"NIAH batch metadata: {metadata_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and run the Framework V2.0 NIAH suite for multiple models.")
    parser.add_argument("--suite", choices=sorted(NIAH_SUITES), default="framework_v2")
    parser.add_argument("--profile", choices=sorted(MODEL_PROFILES), default="minimal")
    parser.add_argument("--models", default=None, help="Comma-separated model aliases or exact API model names.")
    parser.add_argument("--provider", choices=["siliconflow", "local", "custom"], default="siliconflow")
    parser.add_argument("--api-key", default=None, help="Optional. Prefer SILICONFLOW_API_KEY environment variable.")
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--data-root", default="data/generated/niah_batch")
    parser.add_argument("--result-root", default="results/raw/niah_batch")
    parser.add_argument("--metadata-path", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--samples-per-cell", type=int, default=None, help="Override suite sample count for quick tests.")
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--enable-thinking", action="store_true", help="Force thinking on for every model.")
    parser.add_argument("--thinking-budget", type=int, default=None)
    parser.add_argument("--use-api-model-name", action="store_true", help="Pass exact API model names instead of project aliases.")
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--run-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


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

