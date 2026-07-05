from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lmaf.utils.io import ensure_parent, utc_timestamp


def main() -> None:
    args = parse_args()
    metadata = {
        "git_commit": _cmd(["git", "rev-parse", "HEAD"]),
        "python_version": platform.python_version(),
        "cuda_version": _cmd(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]),
        "vllm_version": _package_version("vllm"),
        "gpu_name": _cmd(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]),
        "model_config": args.model_config,
        "runtime_config": args.runtime_config,
        "started_at": args.started_at or "",
        "finished_at": utc_timestamp(),
    }
    out = ensure_parent(args.output)
    with out.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2, sort_keys=True)
    print(f"Wrote metadata to {out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write run metadata JSON.")
    parser.add_argument("--output", default="results/logs/run_metadata.json")
    parser.add_argument("--model-config", default="configs/models.yaml")
    parser.add_argument("--runtime-config", default="configs/runtime.yaml")
    parser.add_argument("--started-at", default=None)
    return parser.parse_args()


def _cmd(command: list[str]) -> str:
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False, timeout=5)
    except Exception:
        return "unknown"
    if completed.returncode != 0:
        return "unknown"
    return completed.stdout.strip().splitlines()[0] if completed.stdout.strip() else "unknown"


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not_installed"


if __name__ == "__main__":
    main()

