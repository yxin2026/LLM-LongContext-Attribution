from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from import_official_datasets import convert_file, find_source_files, infer_task_name
from lmaf.utils.io import write_jsonl


PROFILE_TASKS = {
    "budget_core": "niah_single_1,vt,cwe,fwe",
    "no_external_assets": "niah_single_1,niah_multikey_2,niah_multikey_3,vt,cwe,fwe",
    "full_official": (
        "niah_single_1,niah_single_2,niah_single_3,"
        "niah_multikey_1,niah_multikey_2,niah_multikey_3,"
        "niah_multivalue,niah_multiquery,vt,cwe,fwe,qa_1,qa_2"
    ),
}


def main() -> None:
    args = parse_args()
    tasks = parse_csv(args.tasks or PROFILE_TASKS[args.profile])
    lengths = parse_ints(args.lengths)
    ruler_repo = ROOT / args.ruler_repo
    raw_root = ROOT / args.raw_output_root
    processed_output = ROOT / args.processed_output

    preflight(args, ruler_repo, tasks, dry_run=args.dry_run)
    config = load_ruler_config(ruler_repo)

    commands = build_commands(args, ruler_repo, raw_root, tasks, lengths, config)
    for command, output_file in commands:
        if args.dry_run:
            print("+ " + " ".join(command))
            continue
        if output_file.exists() and not args.force and count_jsonl(output_file) == args.samples_per_cell:
            print(f"Skip existing official RULER file: {output_file}")
            continue
        print(f"Generate official RULER file: {output_file}")
        if args.verbose:
            print("+ " + " ".join(command))
        run_command(command, cwd=ruler_repo / "scripts" / "data", verbose=args.verbose)

    if args.dry_run:
        return
    rows = import_generated(raw_root, processed_output, tasks, args.subset, args.limit_per_file)
    print(f"Wrote {len(rows)} official RULER samples to {processed_output}")
    print("source=NVIDIA/RULER scripts/data/synthetic")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate official NVIDIA/RULER synthetic data and import it into this project's schema."
    )
    parser.add_argument("--profile", choices=sorted(PROFILE_TASKS), default="budget_core")
    parser.add_argument("--tasks", default=None, help="Override profile task list with official RULER task names.")
    parser.add_argument("--lengths", default="4096,16384,32768")
    parser.add_argument("--samples-per-cell", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subset", default="validation")
    parser.add_argument("--tokenizer-type", default="openai")
    parser.add_argument("--tokenizer-path", default="cl100k_base")
    parser.add_argument("--model-template-type", default="base")
    parser.add_argument("--ruler-repo", default="external/RULER")
    parser.add_argument("--raw-output-root", default="data/official_raw/ruler_budget_core")
    parser.add_argument("--processed-output", default="data/processed/official/ruler/samples.jsonl")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--force", action="store_true", help="Regenerate files even if the expected line count exists.")
    parser.add_argument("--verbose", action="store_true", help="Stream official generator logs.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit-per-file", type=int, default=None)
    return parser.parse_args()


def build_commands(
    args: argparse.Namespace,
    ruler_repo: Path,
    raw_root: Path,
    tasks: list[str],
    lengths: list[int],
    config: dict[str, dict[str, Any]],
) -> list[tuple[list[str], Path]]:
    commands: list[tuple[list[str], Path]] = []
    script_dir = ruler_repo / "scripts" / "data"
    synthetic_dir = script_dir / "synthetic"
    constants = load_python_module(synthetic_dir / "constants.py", "ruler_synthetic_constants")
    templates = load_python_module(script_dir / "template.py", "ruler_templates").Templates
    if args.model_template_type not in templates:
        raise SystemExit(f"Unknown RULER model template: {args.model_template_type}")

    for length in lengths:
        for task in tasks:
            if task not in config:
                raise SystemExit(f"Unknown official RULER task: {task}")
            task_cfg = dict(config[task])
            base_cfg = dict(constants.TASKS[task_cfg["task"]])
            args_cfg = dict(task_cfg.get("args") or {})
            task_template = base_cfg["template"]
            answer_prefix = base_cfg.get("answer_prefix", "")
            template = templates[args.model_template_type].format(task_template=task_template) + answer_prefix
            task_script = synthetic_dir / f"{task_cfg['task']}.py"
            output_file = raw_root / str(length) / task / f"{args.subset}.jsonl"
            command = [
                args.python,
                str(task_script),
                "--save_dir",
                str(raw_root / str(length)),
                "--save_name",
                task,
                "--subset",
                args.subset,
                "--tokenizer_path",
                args.tokenizer_path,
                "--tokenizer_type",
                args.tokenizer_type,
                "--max_seq_length",
                str(length),
                "--tokens_to_generate",
                str(base_cfg["tokens_to_generate"]),
                "--num_samples",
                str(args.samples_per_cell),
                "--random_seed",
                str(args.seed),
                "--template",
                template,
            ]
            for key, value in args_cfg.items():
                command.extend([f"--{key}", str(value)])
            commands.append((command, output_file))
    return commands


def preflight(args: argparse.Namespace, ruler_repo: Path, tasks: list[str], dry_run: bool = False) -> None:
    if not ruler_repo.exists():
        raise SystemExit(
            f"RULER repo not found: {ruler_repo}\n"
            "Clone the official repo first: git clone https://github.com/NVIDIA/RULER.git external/RULER"
        )
    config_path = ruler_repo / "scripts" / "synthetic.yaml"
    if not config_path.exists():
        raise SystemExit(f"RULER synthetic.yaml not found: {config_path}")

    missing_modules = missing_python_modules(required_modules(tasks, args.tokenizer_type))
    if missing_modules and not dry_run:
        raise SystemExit(
            "Official RULER generation is missing Python packages: "
            + ", ".join(missing_modules)
            + "\nInstall them in the experiment environment, for example:\n"
            + "pip install wonderwords beautifulsoup4 html2text"
        )

    config = load_ruler_config(ruler_repo)
    missing_assets = required_assets_missing(ruler_repo, tasks, config)
    if missing_assets and not dry_run:
        raise SystemExit(
            "Official RULER generation is missing source assets:\n"
            + "\n".join(f"- {item}" for item in missing_assets)
            + "\nFor Paul Graham essays, run from external/RULER/scripts/data/synthetic/json:\n"
            + "python download_paulgraham_essay.py\n"
            + "For QA assets, download squad.json and/or hotpotqa.json into the same json directory."
        )


def required_modules(tasks: list[str], tokenizer_type: str) -> list[str]:
    modules = ["yaml"]
    if tokenizer_type == "openai":
        modules.append("tiktoken")
    if any(task.startswith("niah_") for task in tasks):
        modules.append("wonderwords")
    return modules


def missing_python_modules(modules: list[str]) -> list[str]:
    missing = []
    for module in modules:
        if importlib.util.find_spec(module) is None:
            missing.append(module)
    return missing


def required_assets_missing(ruler_repo: Path, tasks: list[str], config: dict[str, dict[str, Any]]) -> list[str]:
    json_dir = ruler_repo / "scripts" / "data" / "synthetic" / "json"
    missing = []
    for task in tasks:
        task_cfg = config[task]
        args_cfg = task_cfg.get("args") or {}
        if task_cfg["task"] == "niah" and args_cfg.get("type_haystack") == "essay":
            path = json_dir / "PaulGrahamEssays.json"
            if not path.exists():
                missing.append(str(path))
        if task_cfg["task"] == "qa":
            dataset = args_cfg.get("dataset")
            path = json_dir / f"{dataset}.json"
            if not path.exists():
                missing.append(str(path))
    return missing


def import_generated(
    raw_root: Path,
    output: Path,
    tasks: list[str],
    subset: str,
    limit_per_file: int | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    import_args = SimpleNamespace(
        kind="ruler",
        subset=subset,
        tasks=",".join(tasks),
        append_answer_prefix=True,
        limit_per_file=limit_per_file,
    )
    task_set = set(tasks)
    for path in find_source_files(raw_root, subset):
        task_name = infer_task_name(raw_root, path)
        if task_name not in task_set:
            continue
        rows.extend(convert_file(path, task_name, import_args))
    if not rows:
        raise SystemExit(f"No official RULER rows imported from {raw_root}")
    write_jsonl(output, rows)
    return rows


def run_command(command: list[str], cwd: Path, verbose: bool) -> None:
    if verbose:
        subprocess.run(command, cwd=cwd, check=True)
        return
    result = subprocess.run(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        raise subprocess.CalledProcessError(result.returncode, command)


def load_ruler_config(ruler_repo: Path) -> dict[str, dict[str, Any]]:
    with (ruler_repo / "scripts" / "synthetic.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_python_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    main()
