from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from make_niah_dataset import build_sample


@dataclass(frozen=True)
class RunSpec:
    name: str
    length: int
    positions: tuple[int, ...]
    samples_per_cell: int
    method: str
    gamma: float | None = None

    @property
    def target_length(self) -> int:
        return self.length

    @property
    def method_label(self) -> str:
        if self.method == "apbs":
            return f"apbs_g{self.gamma:.1f}"
        return self.method


def sanitize(value: str) -> str:
    value = value.replace("\\", "/").rstrip("/").split("/")[-1] or value
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def jsonl_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def target_rows(spec: RunSpec, seed: int) -> list[dict]:
    rows = []
    for position in spec.positions:
        for idx in range(spec.samples_per_cell):
            rows.append(build_sample(spec.length, position, idx, seed))
    return rows


def result_is_keepable(row: dict, spec: RunSpec, model_key: str, adopt_legacy: bool) -> bool:
    if row.get("method") != spec.method:
        return False
    if int(row.get("length", -1)) != spec.length:
        return False
    if int(row.get("position", -1)) not in spec.positions:
        return False
    if "sample_id" not in row or "prediction" not in row or "correct" not in row:
        return False
    if row.get("correct") not in {0, 1, True, False}:
        return False

    row_model = row.get("model_key")
    if row_model and row_model != model_key:
        return False
    if not row_model and not adopt_legacy:
        return False

    if spec.method == "apbs":
        try:
            return abs(float(row.get("gamma")) - float(spec.gamma)) < 1e-9
        except (TypeError, ValueError):
            return False
    return True


def completed_ids(raw_dir: Path, spec: RunSpec, model_key: str, target_ids: set[str], adopt_legacy: bool) -> set[str]:
    done = set()
    for path in sorted(raw_dir.glob("**/*.jsonl")):
        for row in jsonl_rows(path):
            sample_id = row.get("sample_id")
            if sample_id in target_ids and result_is_keepable(row, spec, model_key, adopt_legacy):
                done.add(sample_id)
    return done


def build_plan(phase: str, main_samples: int, gamma_samples: int, small32_samples: int) -> list[RunSpec]:
    day1 = [
        RunSpec("16k_baseline", 16384, (10, 50, 90), main_samples, "baseline"),
        RunSpec("16k_ntk", 16384, (10, 50, 90), main_samples, "ntk"),
        RunSpec("16k_apbs_g03", 16384, (10, 50, 90), main_samples, "apbs", 0.3),
    ]
    day2 = [
        RunSpec("16k_apbs_g01_mid", 16384, (50,), gamma_samples, "apbs", 0.1),
        RunSpec("16k_apbs_g05_mid", 16384, (50,), gamma_samples, "apbs", 0.5),
        RunSpec("32k_baseline", 32768, (10, 50, 90), small32_samples, "baseline"),
        RunSpec("32k_ntk", 32768, (10, 50, 90), small32_samples, "ntk"),
        RunSpec("32k_apbs_g03", 32768, (10, 50, 90), small32_samples, "apbs", 0.3),
    ]
    if phase == "day1":
        return day1
    if phase == "day2":
        return day2
    return day1 + day2


def run_missing(args, spec: RunSpec, rows: list[dict], model_key: str) -> None:
    missing_path = Path("data") / "resume_missing" / model_key / f"{spec.name}.jsonl"
    output_path = Path("results") / "raw" / model_key / f"{spec.name}.jsonl"
    write_jsonl(missing_path, rows)

    cmd = [
        sys.executable,
        "scripts/run_apbs_niah.py",
        "--model",
        args.model,
        "--model-key",
        model_key,
        "--dataset",
        str(missing_path),
        "--output",
        str(output_path),
        "--method",
        spec.method,
        "--target-length",
        str(spec.target_length),
        "--train-context-length",
        str(args.train_context_length),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--dtype",
        args.dtype,
        "--device-map",
        args.device_map,
        "--append",
    ]
    if spec.method == "apbs":
        cmd.extend(["--gamma", str(spec.gamma)])
    if args.load_in_4bit:
        cmd.append("--load-in-4bit")
    if args.load_in_8bit:
        cmd.append("--load-in-8bit")
    if args.attn_implementation:
        cmd.extend(["--attn-implementation", args.attn_implementation])
    if args.max_memory:
        cmd.extend(["--max-memory", args.max_memory])
    if args.offload_folder:
        cmd.extend(["--offload-folder", args.offload_folder])

    print("RUN " + " ".join(cmd))
    if not args.dry_run:
        subprocess.run(cmd, check=True)


def analyze_if_requested(args) -> None:
    if args.no_analyze or args.dry_run:
        return
    cmd = [
        sys.executable,
        "scripts/analyze_apbs_results.py",
        "--inputs",
        "results/raw/**/*.jsonl",
        "results/raw/*.jsonl",
        "--output-dir",
        "results/analysis",
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF model id or local model path.")
    parser.add_argument("--model-key", default=None)
    parser.add_argument("--phase", choices=["day1", "day2", "full"], default="day1")
    parser.add_argument("--main-samples", type=int, default=50)
    parser.add_argument("--gamma-samples", type=int, default=50)
    parser.add_argument("--small32-samples", type=int, default=15)
    parser.add_argument("--seed", type=int, default=20260705)
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--max-memory", default=None)
    parser.add_argument("--offload-folder", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--train-context-length", type=int, default=4096)
    parser.add_argument("--adopt-legacy-results", action="store_true", default=True)
    parser.add_argument("--no-adopt-legacy-results", dest="adopt_legacy_results", action="store_false")
    parser.add_argument("--no-analyze", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.load_in_4bit and args.load_in_8bit:
        raise SystemExit("Use either --load-in-4bit or --load-in-8bit, not both.")

    model_key = args.model_key or sanitize(args.model)
    raw_dir = Path("results") / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"model={args.model}")
    print(f"model_key={model_key}")
    print(f"phase={args.phase}")
    print(f"adopt_legacy_results={args.adopt_legacy_results}")

    plan = build_plan(args.phase, args.main_samples, args.gamma_samples, args.small32_samples)
    any_missing = False
    for spec in plan:
        rows = target_rows(spec, args.seed)
        target_ids = {row["sample_id"] for row in rows}
        done = completed_ids(raw_dir, spec, model_key, target_ids, args.adopt_legacy_results)
        missing = [row for row in rows if row["sample_id"] not in done]
        print(f"{spec.name}: target={len(rows)} done={len(done)} missing={len(missing)}")
        if not missing:
            continue
        any_missing = True
        run_missing(args, spec, missing, model_key)

    if not any_missing:
        print("All requested runs are already complete.")

    analyze_if_requested(args)


if __name__ == "__main__":
    main()

