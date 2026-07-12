from __future__ import annotations

import csv
import json
import math
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "lmaf_experiments"
OUT = ROOT / "outputs" / "experiment_summary_20260705"
CLEAN_RAW = OUT / "clean_raw"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORT = OUT / "README.md"

TERMINAL_SKIP_ERRORS = {"skipped_by_model_length", "skipped_overlength"}
DEFAULT_MODELS = [
    "qwen35_9b",
    "qwen3_8b",
    "qwen35_27b",
    "qwen35_35b_a3b",
    "qwen35_122b_a10b",
    "hunyuan_a13b",
    "seed_oss_36b",
    "qwen3_14b_no_thinking",
    "qwen3_14b_thinking",
]


@dataclass(frozen=True)
class DatasetSpec:
    label: str
    experiment: str
    raw_dir: Path
    expected_count: int
    clean_dir: Path
    expected_models: list[str]
    notes: str = ""


def main() -> None:
    reset_output_dirs()
    specs = build_specs()
    status_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for spec in specs:
        rows, manifest = clean_spec(spec)
        status_rows.extend(rows)
        manifest_rows.extend(manifest)

    write_csv(TABLES / "completion_status.csv", status_rows, completion_fields())
    write_csv(TABLES / "clean_manifest.csv", manifest_rows, manifest_fields())

    aggregates = aggregate_clean_results()
    overview_rows = write_overviews(aggregates, status_rows)
    make_figures(aggregates, overview_rows)
    write_report(status_rows, overview_rows, aggregates)

    print(f"Wrote summary to {OUT}")
    print(f"Completion table: {TABLES / 'completion_status.csv'}")
    print(f"Report: {REPORT}")


def reset_output_dirs() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    CLEAN_RAW.mkdir(parents=True, exist_ok=True)


def build_specs() -> list[DatasetSpec]:
    longbench_expected = count_jsonl_rows(PROJECT / "data" / "processed" / "longbench_ruler_batch" / "framework_v2" / "longbench")
    niah_expected = count_jsonl_rows(PROJECT / "data" / "generated" / "niah_batch" / "framework_v2_without_fast16k")
    if longbench_expected == 0:
        longbench_expected = 1750
    if niah_expected == 0:
        niah_expected = 750

    pac_expected = {
        "A": count_external_pac_rows("subset_A.jsonl", fallback=1000),
        "B": count_external_pac_rows("subset_B.jsonl", fallback=1800),
        "C": count_external_pac_rows("subset_C.jsonl", fallback=600),
        "D": count_external_pac_rows("subset_D.jsonl", fallback=1160),
    }

    return [
        DatasetSpec(
            label="NIAH framework_v2_without_fast16k",
            experiment="niah",
            raw_dir=PROJECT / "results" / "raw" / "niah_batch" / "framework_v2_without_fast16k" / "framework_v2_extra",
            expected_count=niah_expected,
            clean_dir=CLEAN_RAW / "niah_framework_v2_without_fast16k",
            expected_models=DEFAULT_MODELS,
            notes="This is the no-fast16k continuation suite in the current workspace.",
        ),
        DatasetSpec(
            label="LongBench framework_v2",
            experiment="longbench",
            raw_dir=PROJECT / "results" / "raw" / "longbench_ruler_batch" / "framework_v2" / "longbench_ruler_main" / "longbench",
            expected_count=longbench_expected,
            clean_dir=CLEAN_RAW / "longbench_framework_v2",
            expected_models=DEFAULT_MODELS,
        ),
        DatasetSpec(
            label="RULER framework_v2",
            experiment="ruler",
            raw_dir=PROJECT / "results" / "raw" / "longbench_ruler_batch" / "framework_v2" / "longbench_ruler_main" / "ruler",
            expected_count=900,
            clean_dir=CLEAN_RAW / "ruler_framework_v2",
            expected_models=DEFAULT_MODELS,
            notes="Expected count follows the configured fallback suite: 6 tasks x 3 lengths x 50 samples.",
        ),
        DatasetSpec(
            label="PAC A position",
            experiment="pac",
            raw_dir=PROJECT / "results" / "raw" / "pac_batch" / "pac_main" / "A",
            expected_count=pac_expected["A"],
            clean_dir=CLEAN_RAW / "pac_A_position",
            expected_models=DEFAULT_MODELS,
        ),
        DatasetSpec(
            label="PAC B interference",
            experiment="pac",
            raw_dir=PROJECT / "results" / "raw" / "pac_batch" / "pac_main" / "B",
            expected_count=pac_expected["B"],
            clean_dir=CLEAN_RAW / "pac_B_interference",
            expected_models=DEFAULT_MODELS,
        ),
        DatasetSpec(
            label="PAC C overlap",
            experiment="pac",
            raw_dir=PROJECT / "results" / "raw" / "pac_batch" / "pac_main" / "C",
            expected_count=pac_expected["C"],
            clean_dir=CLEAN_RAW / "pac_C_overlap",
            expected_models=DEFAULT_MODELS,
        ),
        DatasetSpec(
            label="PAC D multihop",
            experiment="pac",
            raw_dir=PROJECT / "results" / "raw" / "pac_batch" / "pac_main" / "D",
            expected_count=pac_expected["D"],
            clean_dir=CLEAN_RAW / "pac_D_multihop",
            expected_models=DEFAULT_MODELS,
        ),
    ]


def count_jsonl_rows(path: Path) -> int:
    if path.is_file() and path.suffix == ".jsonl":
        return count_lines(path)
    if not path.exists():
        return 0
    total = 0
    for file in path.rglob("*.jsonl"):
        if should_ignore_raw_file(file):
            continue
        total += count_lines(file)
    return total


def count_external_pac_rows(filename: str, fallback: int) -> int:
    candidates = [
        Path(r"D:\Workspace\llm-longcontext-attribution\上下文机制探究\PAC-Test-Dataset\data") / filename,
        Path.cwd() / "data" / "raw" / "pac_test" / filename,
        PROJECT / "data" / "raw" / "pac_test" / filename,
    ]
    for candidate in candidates:
        try:
            if candidate.exists():
                return count_lines(candidate)
        except OSError:
            continue
    return fallback


def count_lines(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    except UnicodeDecodeError:
        with path.open("r", encoding="utf-8-sig") as f:
            return sum(1 for line in f if line.strip())


def clean_spec(spec: DatasetSpec) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    spec.clean_dir.mkdir(parents=True, exist_ok=True)
    status_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    seen_models: set[str] = set()
    files = sorted(spec.raw_dir.glob("*.jsonl")) if spec.raw_dir.exists() else []
    for file in files:
        if should_ignore_raw_file(file):
            continue
        model = file.stem
        seen_models.add(model)
        raw_rows = list(read_jsonl(file))
        final_rows, duplicate_rows = dedupe_rows(raw_rows)
        write_jsonl(spec.clean_dir / file.name, [slim_row(row) for row in final_rows])
        status_rows.append(status_for_file(spec, model, file, raw_rows, final_rows, duplicate_rows))
        manifest_rows.append(
            {
                "dataset": spec.label,
                "experiment": spec.experiment,
                "model": model,
                "source_file": str(file.relative_to(PROJECT)),
                "clean_file": str((spec.clean_dir / file.name).relative_to(ROOT)),
            }
        )
    for missing_model in spec.expected_models:
        if missing_model in seen_models:
            continue
        status_rows.append(status_for_missing(spec, missing_model))
    return status_rows, manifest_rows


def should_ignore_raw_file(file: Path) -> bool:
    name = file.name.lower()
    return "metadata" in name or name.startswith(".")


def read_jsonl(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON at {path}:{line_no}") from exc


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def slim_row(row: dict[str, Any]) -> dict[str, Any]:
    keep = {
        "sample_id",
        "experiment",
        "subtask",
        "task",
        "model",
        "provider",
        "api_model",
        "length",
        "length_tokens_target",
        "length_tokens_actual",
        "position_percent",
        "density",
        "interference_type",
        "similarity",
        "similarity_level",
        "distance",
        "distance_level",
        "hops",
        "num_hops",
        "hop_distance",
        "chain_type",
        "implementation",
        "score",
        "f1",
        "partial_f1",
        "rouge_l",
        "metric",
        "error",
        "error_type",
        "latency_sec",
        "prompt_tokens",
        "completion_tokens",
        "timestamp",
        "source_schema",
    }
    return {key: row.get(key) for key in keep if key in row}


def dedupe_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    no_id: list[dict[str, Any]] = []
    for row in rows:
        sample_id = row.get("sample_id")
        if sample_id in (None, ""):
            no_id.append(row)
        else:
            by_id[str(sample_id)].append(row)
    final: list[dict[str, Any]] = []
    duplicate_count = 0
    for items in by_id.values():
        duplicate_count += max(0, len(items) - 1)
        final.append(select_final_row(items))
    final.extend(no_id)
    return sorted(final, key=lambda row: str(row.get("sample_id", ""))), duplicate_count


def select_final_row(items: list[dict[str, Any]]) -> dict[str, Any]:
    for row in reversed(items):
        if is_success(row) or is_terminal_skip(row):
            return row
    return items[-1]


def is_success(row: dict[str, Any]) -> bool:
    return row.get("error") in (None, "")


def is_terminal_skip(row: dict[str, Any]) -> bool:
    return row.get("error") in TERMINAL_SKIP_ERRORS


def is_retryable_error(row: dict[str, Any]) -> bool:
    return row.get("error") not in (None, "", *TERMINAL_SKIP_ERRORS)


def status_for_file(
    spec: DatasetSpec,
    model: str,
    file: Path,
    raw_rows: list[dict[str, Any]],
    final_rows: list[dict[str, Any]],
    duplicate_rows: int,
) -> dict[str, Any]:
    success = sum(1 for row in final_rows if is_success(row))
    terminal_skips = sum(1 for row in final_rows if is_terminal_skip(row))
    retry_errors = sum(1 for row in final_rows if is_retryable_error(row))
    unique_final = len(final_rows)
    missing = max(0, spec.expected_count - unique_final)
    complete = unique_final >= spec.expected_count and retry_errors == 0
    usable = success + terminal_skips
    return {
        "dataset": spec.label,
        "experiment": spec.experiment,
        "model": model,
        "status": "complete" if complete else "incomplete",
        "expected_samples": spec.expected_count,
        "raw_rows": len(raw_rows),
        "unique_final_rows": unique_final,
        "success_rows": success,
        "terminal_skips": terminal_skips,
        "retryable_errors": retry_errors,
        "missing_samples_est": missing,
        "duplicate_rows_removed": duplicate_rows,
        "completion_rate": round(usable / spec.expected_count, 6) if spec.expected_count else "",
        "source_file": str(file.relative_to(PROJECT)),
        "notes": spec.notes,
    }


def status_for_missing(spec: DatasetSpec, model: str) -> dict[str, Any]:
    return {
        "dataset": spec.label,
        "experiment": spec.experiment,
        "model": model,
        "status": "missing",
        "expected_samples": spec.expected_count,
        "raw_rows": 0,
        "unique_final_rows": 0,
        "success_rows": 0,
        "terminal_skips": 0,
        "retryable_errors": 0,
        "missing_samples_est": spec.expected_count,
        "duplicate_rows_removed": 0,
        "completion_rate": 0,
        "source_file": "",
        "notes": spec.notes,
    }


def aggregate_clean_results() -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    aggregate_jobs = [
        ("niah_without_fast16k", "niah", CLEAN_RAW / "niah_framework_v2_without_fast16k"),
        ("longbench_framework_v2", "longbench", CLEAN_RAW / "longbench_framework_v2"),
        ("ruler_framework_v2", "ruler", CLEAN_RAW / "ruler_framework_v2"),
        ("pac_A_position", "pac", CLEAN_RAW / "pac_A_position"),
        ("pac_B_interference", "pac", CLEAN_RAW / "pac_B_interference"),
        ("pac_C_overlap", "pac", CLEAN_RAW / "pac_C_overlap"),
        ("pac_D_multihop", "pac", CLEAN_RAW / "pac_D_multihop"),
    ]
    for name, experiment, input_dir in aggregate_jobs:
        if not input_dir.exists() or not any(input_dir.glob("*.jsonl")):
            continue
        output = TABLES / f"{name}.csv"
        cmd = [
            sys.executable,
            str(PROJECT / "scripts" / "aggregate_results.py"),
            "--input",
            str(input_dir),
            "--experiment",
            experiment,
            "--output",
            str(output),
        ]
        run(cmd)
        outputs[name] = output
    return outputs


def write_overviews(aggregates: dict[str, Path], status_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, path in aggregates.items():
        agg_rows = read_csv(path)
        by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in agg_rows:
            model = row.get("model") or ""
            if not model:
                continue
            by_model[model].append(row)
        for model, model_rows in by_model.items():
            accuracies = [to_float(row.get("accuracy")) for row in model_rows]
            accuracies = [x for x in accuracies if x is not None]
            error_rates = [to_float(row.get("error_rate")) for row in model_rows]
            error_rates = [x for x in error_rates if x is not None]
            rows.append(
                {
                    "aggregate": name,
                    "model": model,
                    "mean_accuracy": round(sum(accuracies) / len(accuracies), 6) if accuracies else "",
                    "mean_error_rate": round(sum(error_rates) / len(error_rates), 6) if error_rates else "",
                    "n_groups": len(model_rows),
                }
            )
    write_csv(TABLES / "model_overview.csv", rows, ["aggregate", "model", "mean_accuracy", "mean_error_rate", "n_groups"])

    dataset_rows = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in status_rows:
        grouped[row["dataset"]].append(row)
    for dataset, items in grouped.items():
        dataset_rows.append(
            {
                "dataset": dataset,
                "models_complete": sum(1 for row in items if row["status"] == "complete"),
                "models_incomplete": sum(1 for row in items if row["status"] == "incomplete"),
                "models_missing": sum(1 for row in items if row["status"] == "missing"),
                "total_retryable_errors": sum(int(row["retryable_errors"]) for row in items),
                "total_missing_samples_est": sum(int(row["missing_samples_est"]) for row in items),
            }
        )
    write_csv(
        TABLES / "dataset_completion_overview.csv",
        dataset_rows,
        [
            "dataset",
            "models_complete",
            "models_incomplete",
            "models_missing",
            "total_retryable_errors",
            "total_missing_samples_est",
        ],
    )
    return rows


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def make_figures(aggregates: dict[str, Path], overview_rows: list[dict[str, Any]]) -> None:
    plot_jobs = [
        ("niah_without_fast16k", "niah_position_curve", "niah_position_curve.png"),
        ("longbench_framework_v2", "longbench_score_bar", "longbench_score_bar.png"),
        ("ruler_framework_v2", "ruler_effective_context", "ruler_effective_context.png"),
        ("pac_A_position", "pac_A_position_curve", "pac_A_position_curve.png"),
        ("pac_B_interference", "pac_B_density_curve", "pac_B_density_curve.png"),
        ("pac_C_overlap", "pac_C_confusion_matrix", "pac_C_confusion_matrix.png"),
        ("pac_D_multihop", "pac_D_multihop_decay", "pac_D_multihop_decay.png"),
    ]
    for aggregate_name, plot_kind, filename in plot_jobs:
        if aggregate_name not in aggregates:
            continue
        cmd = [
            sys.executable,
            str(PROJECT / "scripts" / "plot_results.py"),
            "--input",
            str(aggregates[aggregate_name]),
            "--plot",
            plot_kind,
            "--output",
            str(FIGURES / filename),
        ]
        try:
            run(cmd)
        except subprocess.CalledProcessError as exc:
            print(f"Warning: plotting failed for {aggregate_name}: {exc}", file=sys.stderr)
    make_overview_plot(overview_rows)


def make_overview_plot(overview_rows: list[dict[str, Any]]) -> None:
    if not overview_rows:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return

    grouped: dict[str, list[float]] = defaultdict(list)
    for row in overview_rows:
        value = to_float(row.get("mean_accuracy"))
        if value is not None:
            grouped[row["aggregate"]].append(value)
    labels = list(grouped)
    values = [sum(grouped[label]) / len(grouped[label]) for label in labels]
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(labels)), values)
    plt.xticks(range(len(labels)), labels, rotation=30, ha="right")
    plt.ylabel("Mean model accuracy")
    plt.title("Existing clean-result overview")
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(FIGURES / "overview_mean_accuracy.png", dpi=160)
    plt.close()


def write_report(status_rows: list[dict[str, Any]], overview_rows: list[dict[str, Any]], aggregates: dict[str, Path]) -> None:
    dataset_summary = list(read_csv(TABLES / "dataset_completion_overview.csv"))
    complete_datasets = [row for row in dataset_summary if int(row["models_complete"]) > 0]
    lines = [
        "# Existing Experiment Results Summary",
        "",
        "This folder was generated from the current `lmaf_experiments/results/raw` files. Raw files were not modified.",
        "",
        "## Completion Overview",
        "",
        "| Dataset | Complete Models | Incomplete Models | Missing Models | Retryable Errors | Missing Samples Estimate |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in dataset_summary:
        lines.append(
            f"| {row['dataset']} | {row['models_complete']} | {row['models_incomplete']} | "
            f"{row['models_missing']} | {row['total_retryable_errors']} | {row['total_missing_samples_est']} |"
        )
    lines.extend(
        [
            "",
            "## Generated Tables",
            "",
            "- `tables/completion_status.csv`: per dataset/model completion status.",
            "- `tables/clean_manifest.csv`: mapping from raw JSONL files to cleaned JSONL files.",
            "- `tables/model_overview.csv`: mean scores by model and aggregate.",
            "- `tables/*_framework*.csv` and `tables/pac_*.csv`: cleaned aggregate tables.",
            "",
            "## Generated Figures",
            "",
        ]
    )
    for figure in sorted(FIGURES.glob("*.png")):
        lines.append(f"- `figures/{figure.name}`")
    lines.extend(
        [
            "",
            "## Confirmed Complete Data",
            "",
        ]
    )
    if complete_datasets:
        for row in complete_datasets:
            lines.append(f"- {row['dataset']}: {row['models_complete']} model(s) complete.")
    else:
        lines.append("- No dataset/model pair is fully complete under the current strict definition.")
    lines.extend(
        [
            "",
            "Strict definition: unique final rows >= expected samples and no retryable API errors remain. "
            "Terminal overlength skips count as completed bookkeeping but are excluded by default in aggregate tables.",
            "",
        ]
    )
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def completion_fields() -> list[str]:
    return [
        "dataset",
        "experiment",
        "model",
        "status",
        "expected_samples",
        "raw_rows",
        "unique_final_rows",
        "success_rows",
        "terminal_skips",
        "retryable_errors",
        "missing_samples_est",
        "duplicate_rows_removed",
        "completion_rate",
        "source_file",
        "notes",
    ]


def manifest_fields() -> list[str]:
    return ["dataset", "experiment", "model", "source_file", "clean_file"]


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=PROJECT, check=True)


def to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


if __name__ == "__main__":
    main()
