from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lmaf.utils.models import is_excluded_model


MODEL_ORDER = [
    "qwen35_9b",
    "qwen3_8b",
    "qwen35_27b",
    "qwen35_35b_a3b",
    "qwen35_122b_a10b",
    "qwen3_14b_no_thinking",
    "qwen3_14b_thinking",
    "hunyuan_a13b",
    "seed_oss_36b",
]

LONG_BENCH_TASK_ORDER = [
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "gov_report",
    "qmsum",
    "multi_news",
]

TASK_CATEGORY_ORDER = ["single_doc_qa", "multi_doc_qa", "summarization"]

DEFAULT_INPUTS = {
    "longbench": "results/raw/longbench_ruler_batch/framework_v2/longbench_ruler_main/longbench",
    "niah": "results/raw/niah_batch/framework_v2_without_fast16k/framework_v2_extra",
    "ruler": "results/raw/longbench_ruler_batch/framework_v2/longbench_ruler_main/ruler",
}

CODE_PURPOSES = {
    "scripts/run_longbench.py": "Prepare and run LongBench evaluations.",
    "scripts/run_ruler.py": "Prepare and run RULER/fallback synthetic evaluations.",
    "scripts/run_longbench_ruler_batch.py": "Batch runner for LongBench and RULER model sweeps.",
    "scripts/run_niah.py": "Generate and run NIAH variants.",
    "scripts/run_niah_batch.py": "Batch runner for NIAH suites.",
    "scripts/aggregate_results.py": "Generic JSONL-to-CSV aggregation utility.",
    "scripts/plot_results.py": "Generic plotting utility for aggregated benchmark results.",
    "scripts/summarize_public_benchmarks.py": "Publication-style LongBench/NIAH/RULER summary generator.",
    "scripts/build_public_benchmarks_workbook.mjs": "Excel workbook builder for public benchmark summary tables.",
    "scripts/run_official_budget_topup.py": "Budgeted top-up runner used for later official-data experiments.",
    "scripts/generate_official_niah.py": "Official-style NIAH data generator.",
    "scripts/generate_official_ruler.py": "Official-style RULER data generator.",
    "src/lmaf/data/longbench.py": "LongBench data preparation logic.",
    "src/lmaf/data/niah.py": "NIAH sample construction logic.",
    "src/lmaf/data/ruler.py": "RULER/fallback sample construction logic.",
    "src/lmaf/eval/metrics.py": "Shared scoring and statistics helpers.",
    "src/lmaf/providers.py": "Provider/API dispatch layer.",
    "src/lmaf/utils/models.py": "Model filtering and excluded-model registry.",
    "requirements.txt": "Python dependency list.",
    "pyproject.toml": "Project metadata.",
    "README.md": "Project usage notes.",
}


def main() -> None:
    args = parse_args()
    output = (ROOT / args.output).resolve()
    tables_dir = output / "tables"
    figures_dir = output / "figures"
    code_dir = output / "code_snapshot"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    code_dir.mkdir(parents=True, exist_ok=True)

    rows, source_rows = load_rows(args)
    if not rows:
        raise SystemExit("No benchmark rows found. Check input roots.")
    raw = pd.DataFrame(rows)
    raw = normalize_dataframe(raw)

    outputs: dict[str, Path] = {}
    outputs.update(write_core_tables(raw, source_rows, tables_dir))
    outputs.update(write_longbench_tables(raw, tables_dir))
    outputs.update(write_niah_tables(raw, tables_dir))
    outputs.update(write_ruler_tables(raw, tables_dir))
    outputs.update(write_figures(raw, figures_dir))
    outputs.update(write_code_inventory(code_dir, tables_dir))
    outputs["summary_report"] = write_report(raw, output, outputs, source_rows)
    write_workbook_payload(output, tables_dir, outputs)

    print(f"Wrote public benchmark summary to {output}")
    for key, path in sorted(outputs.items()):
        print(f"{key}: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize LongBench, NIAH, and RULER results for paper-style reporting.")
    parser.add_argument("--output", default="results/reports/public_benchmarks_summary")
    parser.add_argument("--longbench-root", default=DEFAULT_INPUTS["longbench"])
    parser.add_argument("--niah-root", default=DEFAULT_INPUTS["niah"])
    parser.add_argument("--ruler-root", default=DEFAULT_INPUTS["ruler"])
    parser.add_argument("--include-excluded-models", action="store_true")
    parser.add_argument(
        "--score-errors-as-zero",
        action="store_true",
        help="Use score_all as the main score. Default is success-only scores with coverage reported separately.",
    )
    return parser.parse_args()


def load_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    roots = {
        "longbench": ROOT / args.longbench_root,
        "niah": ROOT / args.niah_root,
        "ruler": ROOT / args.ruler_root,
    }
    chosen: dict[tuple[str, str, str], dict[str, Any]] = {}
    source_rows: list[dict[str, Any]] = []
    for benchmark, root in roots.items():
        files = sorted(root.rglob("*.jsonl")) if root.exists() else []
        source_count = 0
        source_errors = 0
        for path in files:
            for row in iter_jsonl(path):
                source_count += 1
                model = str(row.get("model") or path.stem)
                api_model = str(row.get("api_model") or "")
                if not args.include_excluded_models and (is_excluded_model(model) or is_excluded_model(api_model)):
                    continue
                sample_id = str(row.get("sample_id") or "")
                if not sample_id:
                    continue
                normalized = normalize_row(row, benchmark, model, path, args.score_errors_as_zero)
                if normalized["error"]:
                    source_errors += 1
                key = (benchmark, model, sample_id)
                previous = chosen.get(key)
                if previous is None or row_priority(normalized) > row_priority(previous):
                    chosen[key] = normalized
        source_rows.append(
            {
                "benchmark": benchmark,
                "root": str(root.relative_to(ROOT)) if root.exists() else str(root),
                "included": root.exists(),
                "jsonl_files": len(files),
                "raw_rows_seen": source_count,
                "raw_errors_seen": source_errors,
                "last_modified": latest_mtime(files),
                "note": "primary public-benchmark source selected for paper summary",
            }
        )
    return list(chosen.values()), source_rows


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                yield {
                    "sample_id": f"json_error::{path.name}::{line_no}",
                    "model": path.stem,
                    "error": f"JSONDecodeError: {exc}",
                    "score": 0.0,
                }


def normalize_row(row: dict[str, Any], benchmark: str, model: str, path: Path, score_errors_as_zero: bool) -> dict[str, Any]:
    error = row.get("error")
    success = error in (None, "")
    score = to_float(row.get("score"))
    score_success = score if success else math.nan
    score_all = score if success and score is not None else 0.0
    task = row.get("task") or row.get("subtask") or row.get("variant") or ""
    category = row.get("category") or infer_longbench_category(str(task))
    length = row.get("length_tokens_target") or row.get("length")
    return {
        "benchmark": benchmark,
        "experiment": row.get("experiment") or benchmark,
        "model": model,
        "api_model": row.get("api_model"),
        "provider": row.get("provider"),
        "sample_id": row.get("sample_id"),
        "task": str(task),
        "category": category,
        "subtask": str(row.get("subtask") or task),
        "length": normalize_number(length),
        "position": normalize_number(row.get("position_percent")),
        "position_actual": normalize_number(row.get("position_percent_actual")),
        "distribution": row.get("distribution"),
        "implementation": row.get("implementation"),
        "metric": row.get("metric"),
        "score": score,
        "score_success": score_success,
        "score_all": score_all,
        "score_main": score_all if score_errors_as_zero else score_success,
        "f1": to_float(row.get("f1") or row.get("partial_f1")),
        "rouge_l": to_float(row.get("rouge_l")),
        "exact_match": to_float(row.get("exact_match")),
        "latency_sec": to_float(row.get("latency_sec")),
        "prompt_tokens": to_float(row.get("prompt_tokens")),
        "completion_tokens": to_float(row.get("completion_tokens")),
        "error": "" if success else str(error),
        "error_type": row.get("error_type"),
        "success": bool(success),
        "source_file": str(path.relative_to(ROOT)),
        "source_mtime": path.stat().st_mtime,
    }


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["length", "position", "position_actual", "score", "score_success", "score_all", "score_main", "f1", "rouge_l", "exact_match", "latency_sec"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["success"] = df["success"].astype(bool)
    df["model"] = pd.Categorical(df["model"], categories=MODEL_ORDER, ordered=True)
    return df.sort_values(["benchmark", "model", "task", "length", "position", "sample_id"], kind="stable")


def row_priority(row: dict[str, Any]) -> tuple[int, int, float]:
    return (
        1 if row.get("success") else 0,
        1 if to_float(row.get("score")) is not None else 0,
        float(row.get("source_mtime") or 0),
    )


def write_core_tables(raw: pd.DataFrame, source_rows: list[dict[str, Any]], tables: Path) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    outputs["source_runs"] = write_csv(pd.DataFrame(source_rows), tables / "source_runs.csv")
    outputs["raw_public_rows_dedup"] = write_csv(
        raw[
            [
                "benchmark",
                "model",
                "sample_id",
                "task",
                "category",
                "subtask",
                "length",
                "position",
                "distribution",
                "implementation",
                "metric",
                "success",
                "score",
                "score_success",
                "score_all",
                "f1",
                "rouge_l",
                "exact_match",
                "latency_sec",
                "error",
                "error_type",
                "source_file",
            ]
        ],
        tables / "raw_public_rows_dedup.csv",
    )
    coverage = summarize(raw, ["benchmark", "model"], ["benchmark", "model"])
    outputs["coverage_by_benchmark_model"] = write_csv(coverage, tables / "coverage_by_benchmark_model.csv")
    benchmark_summary = summarize(raw, ["benchmark"], ["benchmark"])
    benchmark_summary["n_models"] = raw.groupby("benchmark")["model"].nunique().reindex(benchmark_summary["benchmark"]).to_numpy()
    benchmark_summary["n_tasks"] = raw.groupby("benchmark")["task"].nunique().reindex(benchmark_summary["benchmark"]).to_numpy()
    outputs["benchmark_summary"] = write_csv(benchmark_summary, tables / "benchmark_summary.csv")
    return outputs


def write_longbench_tables(raw: pd.DataFrame, tables: Path) -> dict[str, Path]:
    df = raw[raw["benchmark"] == "longbench"].copy()
    outputs: dict[str, Path] = {}
    if df.empty:
        return outputs
    outputs["longbench_by_task_model"] = write_csv(summarize(df, ["model", "category", "task"], ["task", "model"]), tables / "longbench_by_task_model.csv")
    outputs["longbench_by_category_model"] = write_csv(summarize(df, ["model", "category"], ["category", "model"]), tables / "longbench_by_category_model.csv")
    outputs["longbench_by_model"] = write_csv(summarize(df, ["model"], ["score_mean"], ascending=[False]), tables / "longbench_by_model.csv")
    return outputs


def write_niah_tables(raw: pd.DataFrame, tables: Path) -> dict[str, Path]:
    df = raw[raw["benchmark"] == "niah"].copy()
    outputs: dict[str, Path] = {}
    if df.empty:
        return outputs
    outputs["niah_by_subtask_model"] = write_csv(summarize(df, ["model", "subtask"], ["subtask", "model"]), tables / "niah_by_subtask_model.csv")
    outputs["niah_by_condition_model"] = write_csv(
        summarize(df, ["model", "subtask", "length", "position", "distribution"], ["subtask", "length", "position", "distribution", "model"]),
        tables / "niah_by_condition_model.csv",
    )
    single = df[(df["subtask"] == "single") & df["position"].notna()]
    if not single.empty:
        outputs["niah_single_position"] = write_csv(summarize(single, ["model", "length", "position"], ["length", "position", "model"]), tables / "niah_single_position.csv")
        outputs["niah_middle_drop"] = write_csv(compute_middle_drop(single), tables / "niah_middle_drop.csv")
    return outputs


def write_ruler_tables(raw: pd.DataFrame, tables: Path) -> dict[str, Path]:
    df = raw[raw["benchmark"] == "ruler"].copy()
    outputs: dict[str, Path] = {}
    if df.empty:
        return outputs
    outputs["ruler_by_task_model"] = write_csv(summarize(df, ["model", "task"], ["task", "model"]), tables / "ruler_by_task_model.csv")
    outputs["ruler_by_task_length_model"] = write_csv(summarize(df, ["model", "task", "length"], ["task", "length", "model"]), tables / "ruler_by_task_length_model.csv")
    outputs["ruler_effective_context"] = write_csv(compute_ruler_ecl(df), tables / "ruler_effective_context.csv")
    return outputs


def summarize(df: pd.DataFrame, group_cols: list[str], sort_cols: list[str], ascending: list[bool] | None = None) -> pd.DataFrame:
    grouped = df.groupby(group_cols, dropna=False, observed=False)
    out = grouped.agg(
        n_total=("sample_id", "count"),
        n_eval=("success", "sum"),
        n_error=("success", lambda s: int((~s.astype(bool)).sum())),
        score_mean=("score_main", "mean"),
        score_all=("score_all", "mean"),
        f1_mean=("f1", "mean"),
        rouge_l_mean=("rouge_l", "mean"),
        exact_match_mean=("exact_match", "mean"),
        latency_mean=("latency_sec", "mean"),
        prompt_tokens_mean=("prompt_tokens", "mean"),
    ).reset_index()
    out["coverage"] = out["n_eval"] / out["n_total"].replace(0, np.nan)
    out["error_rate"] = out["n_error"] / out["n_total"].replace(0, np.nan)
    ordered = group_cols + [
        "n_total",
        "n_eval",
        "n_error",
        "coverage",
        "error_rate",
        "score_mean",
        "score_all",
        "f1_mean",
        "rouge_l_mean",
        "exact_match_mean",
        "latency_mean",
        "prompt_tokens_mean",
    ]
    out = out[ordered]
    if ascending is None:
        ascending = [True] * len(sort_cols)
    return out.sort_values(sort_cols, ascending=ascending, kind="stable")


def compute_middle_drop(single: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = summarize(single, ["model", "length", "position"], ["model", "length", "position"])
    for (model, length), part in grouped.groupby(["model", "length"], observed=False):
        pos = {int(row.position): row.score_mean for row in part.itertuples() if not pd.isna(row.score_mean)}
        if 50 not in pos:
            continue
        edge_values = [pos[p] for p in (10, 90) if p in pos]
        if not edge_values:
            continue
        edge_mean = float(np.mean(edge_values))
        middle = float(pos[50])
        rows.append(
            {
                "model": model,
                "length": length,
                "edge_mean_10_90": edge_mean,
                "middle_50": middle,
                "middle_drop": edge_mean - middle,
                "relative_middle_drop": (edge_mean - middle) / edge_mean if edge_mean else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["length", "model"], kind="stable")


def compute_ruler_ecl(df: pd.DataFrame) -> pd.DataFrame:
    grouped = summarize(df, ["model", "task", "length"], ["model", "task", "length"])
    rows: list[dict[str, Any]] = []
    for (model, task), part in grouped.groupby(["model", "task"], observed=False):
        part = part.dropna(subset=["length", "score_mean"]).sort_values("length")
        if part.empty:
            continue
        base = float(part.iloc[0]["score_mean"])
        rel_threshold = 0.85 * base
        abs_threshold = 0.80
        rel_ok = part[part["score_mean"] >= rel_threshold]
        abs_ok = part[part["score_mean"] >= abs_threshold]
        rows.append(
            {
                "model": model,
                "task": task,
                "base_length": int(part.iloc[0]["length"]),
                "base_accuracy": base,
                "relative_threshold_85pct_base": rel_threshold,
                "ecl_relative": int(rel_ok["length"].max()) if not rel_ok.empty else np.nan,
                "ecl_abs80": int(abs_ok["length"].max()) if not abs_ok.empty else np.nan,
                "n_lengths": len(part),
                "mean_accuracy": float(part["score_mean"].mean()),
                "min_accuracy": float(part["score_mean"].min()),
            }
        )
    return pd.DataFrame(rows).sort_values(["task", "model"], kind="stable")


def write_figures(raw: pd.DataFrame, figures: Path) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    coverage = summarize(raw, ["benchmark", "model"], ["benchmark", "model"])
    outputs["fig_coverage_heatmap"] = heatmap(
        coverage,
        row="model",
        col="benchmark",
        value="coverage",
        row_order=MODEL_ORDER,
        col_order=["longbench", "niah", "ruler"],
        title="Evaluation coverage by benchmark and model",
        output=figures / "coverage_heatmap.png",
        vmin=0,
        vmax=1,
        fmt=".0%",
    )

    lb = raw[raw["benchmark"] == "longbench"]
    if not lb.empty:
        lb_task = summarize(lb, ["model", "task"], ["task", "model"])
        outputs["fig_longbench_task_heatmap"] = heatmap(
            lb_task,
            row="model",
            col="task",
            value="score_mean",
            row_order=MODEL_ORDER,
            col_order=[x for x in LONG_BENCH_TASK_ORDER if x in set(lb_task["task"])],
            title="LongBench score by task and model",
            output=figures / "longbench_task_heatmap.png",
        )
        lb_cat = summarize(lb, ["model", "category"], ["category", "model"])
        outputs["fig_longbench_category_heatmap"] = heatmap(
            lb_cat,
            row="model",
            col="category",
            value="score_mean",
            row_order=MODEL_ORDER,
            col_order=[x for x in TASK_CATEGORY_ORDER if x in set(lb_cat["category"])],
            title="LongBench score by task family",
            output=figures / "longbench_category_heatmap.png",
        )
        outputs["fig_longbench_model_ranking"] = horizontal_bar(
            summarize(lb, ["model"], ["score_mean"], ascending=[False]),
            "model",
            "score_mean",
            "LongBench mean score",
            figures / "longbench_model_ranking.png",
            "Mean score",
        )

    niah = raw[raw["benchmark"] == "niah"]
    if not niah.empty:
        niah_subtask = summarize(niah, ["model", "subtask"], ["subtask", "model"])
        outputs["fig_niah_subtask_heatmap"] = heatmap(
            niah_subtask,
            row="model",
            col="subtask",
            value="score_mean",
            row_order=MODEL_ORDER,
            col_order=[x for x in ["single", "multi", "sequential"] if x in set(niah_subtask["subtask"])],
            title="NIAH accuracy by variant and model",
            output=figures / "niah_variant_heatmap.png",
        )
        single = niah[(niah["subtask"] == "single") & niah["position"].notna()]
        if not single.empty:
            chosen_length = choose_length(single)
            single_len = single[single["length"] == chosen_length]
            outputs["fig_niah_single_position_heatmap"] = heatmap(
                summarize(single_len, ["model", "position"], ["position", "model"]),
                row="model",
                col="position",
                value="score_mean",
                row_order=MODEL_ORDER,
                col_order=sorted_dropna(single_len["position"]),
                title=f"NIAH single-needle position curve ({int(chosen_length)} tokens)",
                output=figures / f"niah_single_position_{int(chosen_length)}.png",
            )
            outputs["fig_niah_single_position_lines"] = line_by_model(
                summarize(single_len, ["model", "position"], ["position", "model"]),
                x="position",
                y="score_mean",
                title=f"NIAH single-needle accuracy by position ({int(chosen_length)} tokens)",
                output=figures / f"niah_single_position_lines_{int(chosen_length)}.png",
                x_label="Needle position (%)",
                y_label="Accuracy",
            )

    ruler = raw[raw["benchmark"] == "ruler"]
    if not ruler.empty:
        ruler_task = summarize(ruler, ["model", "task"], ["task", "model"])
        outputs["fig_ruler_task_heatmap"] = heatmap(
            ruler_task,
            row="model",
            col="task",
            value="score_mean",
            row_order=MODEL_ORDER,
            col_order=sorted_dropna(ruler_task["task"]),
            title="RULER accuracy by task and model",
            output=figures / "ruler_task_heatmap.png",
        )
        ruler_length = summarize(ruler, ["task", "length"], ["task", "length"])
        outputs["fig_ruler_task_length_heatmap"] = heatmap(
            ruler_length,
            row="task",
            col="length",
            value="score_mean",
            row_order=sorted_dropna(ruler_length["task"]),
            col_order=sorted_dropna(ruler_length["length"]),
            title="RULER mean accuracy by task and context length",
            output=figures / "ruler_task_length_heatmap.png",
        )
        outputs["fig_ruler_context_length_lines"] = line_by_task(
            ruler_length,
            x="length",
            y="score_mean",
            series="task",
            title="RULER accuracy as context length increases",
            output=figures / "ruler_context_length_lines.png",
            x_label="Context length (tokens)",
            y_label="Mean accuracy",
        )
    return outputs


def heatmap(
    df: pd.DataFrame,
    row: str,
    col: str,
    value: str,
    row_order: list[Any],
    col_order: list[Any],
    title: str,
    output: Path,
    vmin: float = 0,
    vmax: float = 1,
    fmt: str = ".2f",
) -> Path:
    if df.empty:
        return output
    pivot = df.pivot_table(index=row, columns=col, values=value, aggfunc="mean", observed=False)
    row_order = [x for x in row_order if x in pivot.index]
    col_order = [x for x in col_order if x in pivot.columns]
    if not row_order or not col_order:
        return output
    pivot = pivot.reindex(index=row_order, columns=col_order)
    data = pivot.to_numpy(dtype=float)
    width = max(7.5, 1.1 * len(col_order) + 3)
    height = max(4.8, 0.48 * len(row_order) + 1.8)
    fig, ax = plt.subplots(figsize=(width, height), dpi=180)
    masked = np.ma.masked_invalid(data)
    cmap = matplotlib.colormaps["YlGnBu"].copy()
    cmap.set_bad("#F3F4F6")
    im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_title(title, fontsize=12, weight="bold", pad=12)
    ax.set_xticks(np.arange(len(col_order)))
    ax.set_xticklabels([short_label(x) for x in col_order], rotation=30, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(row_order)))
    ax.set_yticklabels([short_label(x) for x in row_order], fontsize=8)
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-0.5, len(col_order), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_order), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(len(row_order)):
        for j in range(len(col_order)):
            val = data[i, j]
            if not math.isnan(val):
                label = format(val, fmt)
                color = "white" if val >= 0.62 else "#111827"
                ax.text(j, i, label, ha="center", va="center", fontsize=7, color=color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.ax.tick_params(labelsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def horizontal_bar(df: pd.DataFrame, label_col: str, value_col: str, title: str, output: Path, x_label: str) -> Path:
    plot_df = df[[label_col, value_col]].dropna().sort_values(value_col, ascending=True)
    fig, ax = plt.subplots(figsize=(8.6, max(4.6, 0.45 * len(plot_df) + 1.3)), dpi=180)
    vals = plot_df[value_col].astype(float).to_numpy()
    y = np.arange(len(plot_df))
    ax.barh(y, vals, color="#2563EB")
    ax.set_yticks(y)
    ax.set_yticklabels([short_label(x) for x in plot_df[label_col]], fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel(x_label, fontsize=9)
    ax.set_title(title, fontsize=12, weight="bold", pad=12)
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.8)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    for yi, val in zip(y, vals):
        ax.text(min(0.99, val + 0.012), yi, f"{val:.2f}", va="center", fontsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def line_by_model(df: pd.DataFrame, x: str, y: str, title: str, output: Path, x_label: str, y_label: str) -> Path:
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=180)
    for model in MODEL_ORDER:
        part = df[df["model"].astype(str) == model].dropna(subset=[x, y]).sort_values(x)
        if part.empty:
            continue
        ax.plot(part[x], part[y], marker="o", linewidth=1.7, label=short_label(model))
    ax.set_title(title, fontsize=12, weight="bold", pad=12)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_ylim(0, 1.05)
    ax.grid(color="#E5E7EB")
    ax.legend(fontsize=7, ncol=2, frameon=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def line_by_task(df: pd.DataFrame, x: str, y: str, series: str, title: str, output: Path, x_label: str, y_label: str) -> Path:
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=180)
    for name in sorted_dropna(df[series]):
        part = df[df[series].astype(str) == str(name)].dropna(subset=[x, y]).sort_values(x)
        if part.empty:
            continue
        ax.plot(part[x], part[y], marker="o", linewidth=1.7, label=short_label(name))
    ax.set_title(title, fontsize=12, weight="bold", pad=12)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_ylim(0, 1.05)
    ax.grid(color="#E5E7EB")
    ax.legend(fontsize=7, ncol=2, frameon=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def write_code_inventory(code_dir: Path, tables: Path) -> dict[str, Path]:
    rows: list[dict[str, Any]] = []
    for rel, purpose in CODE_PURPOSES.items():
        source = ROOT / rel
        exists = source.exists()
        target = code_dir / rel
        if exists:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        rows.append(
            {
                "path": rel,
                "exists": exists,
                "purpose": purpose,
                "size_bytes": source.stat().st_size if exists else "",
                "line_count": count_lines(source) if exists and source.is_file() else "",
                "sha256": sha256_file(source) if exists and source.is_file() else "",
                "last_modified": fmt_time(source.stat().st_mtime) if exists else "",
                "snapshot_path": str(target.relative_to(code_dir.parent)) if exists else "",
            }
        )
    inventory = pd.DataFrame(rows)
    path = write_csv(inventory, tables / "code_inventory.csv")
    (code_dir / "README.md").write_text(
        "\n".join(
            [
                "# Code Snapshot",
                "",
                "This folder contains a lightweight snapshot of scripts and source files used to generate or summarize the public benchmark results.",
                "It is intended for paper/report reproducibility, not as a replacement for the full repository.",
            ]
        ),
        encoding="utf-8",
    )
    return {"code_inventory": path}


def write_report(raw: pd.DataFrame, output: Path, outputs: dict[str, Path], source_rows: list[dict[str, Any]]) -> Path:
    coverage = summarize(raw, ["benchmark", "model"], ["benchmark", "model"])
    bench = summarize(raw, ["benchmark"], ["benchmark"])
    top_lines = []
    for benchmark in ["longbench", "niah", "ruler"]:
        part = raw[raw["benchmark"] == benchmark]
        if part.empty:
            continue
        by_model = summarize(part, ["model"], ["score_mean"], ascending=[False]).dropna(subset=["score_mean"])
        if by_model.empty:
            continue
        best = by_model.iloc[0]
        top_lines.append(f"- {benchmark}: 当前成功样本均分最高的模型为 `{best['model']}`，score={best['score_mean']:.3f}，coverage={best['coverage']:.1%}。")

    niah_line = ""
    niah = raw[raw["benchmark"] == "niah"]
    if not niah.empty:
        niah_summary = summarize(niah, ["subtask"], ["subtask"])
        easy = niah_summary["score_mean"].mean()
        niah_line = f"NIAH 普通检索任务的平均成功样本准确率约为 {easy:.3f}，可用于说明基础检索能力较强，但不宜单独作为复杂记忆衰减证据。"

    ruler_line = ""
    ruler = raw[raw["benchmark"] == "ruler"]
    if not ruler.empty:
        ruler_summary = summarize(ruler, ["task"], ["task"])
        ruler_line = f"RULER 当前普通/ fallback 配置平均准确率约为 {ruler_summary['score_mean'].mean():.3f}，主要用于有效上下文边界与基础 synthetic 任务筛查。"

    source_lines = [f"- {row['benchmark']}: `{row['root']}`" for row in source_rows]
    lines = [
        "# Public Benchmark Result Summary",
        "",
        "## Scope",
        "",
        "本报告整理 LongBench、普通 NIAH 和普通 RULER 的已完成实验结果，用作论文/报告中公开基准阶段的基础证据。分数默认只在成功 API 调用上计算；API 额度、限流或连接错误不计为模型能力错误，而是在 coverage/error_rate 中单独呈现。",
        "",
        "## Included Result Roots",
        "",
        *source_lines,
        "",
        "## High-Level Findings",
        "",
        *top_lines,
        "",
        niah_line,
        "",
        ruler_line,
        "",
        "整体解释口径：LongBench 作为通用长上下文能力基线，NIAH 验证基础检索能力，RULER 检查普通 synthetic 任务和有效上下文边界。若这些公开基准出现高分或区分度不足，应解释为公开基准天花板效应，而不是实验失败；这正是后续 PAC-Test 2.0 转向高相似干扰、实体绑定和多跳假链的动机。",
        "",
        "## Tables",
        "",
    ]
    for key, path in sorted(outputs.items()):
        if path.suffix.lower() == ".csv":
            lines.append(f"- {key}: `{path.relative_to(output)}`")
    lines.extend(["", "## Figures", ""])
    for key, path in sorted(outputs.items()):
        if path.suffix.lower() == ".png":
            lines.append(f"- {key}: `{path.relative_to(output)}`")
    lines.extend(["", "## Coverage Snapshot", "", dataframe_to_markdown(coverage), "", "## Benchmark Summary", "", dataframe_to_markdown(bench), ""])
    path = output / "summary_report.md"
    path.write_text("\n".join(lines), encoding="utf-8-sig")
    return path


def write_workbook_payload(output: Path, tables: Path, outputs: dict[str, Path]) -> Path:
    sheet_specs = [
        ("Overview", tables / "benchmark_summary.csv", 80),
        ("Coverage", tables / "coverage_by_benchmark_model.csv", 200),
        ("LongBench_Model", tables / "longbench_by_model.csv", 100),
        ("LongBench_Task", tables / "longbench_by_task_model.csv", 300),
        ("NIAH_Subtask", tables / "niah_by_subtask_model.csv", 200),
        ("NIAH_Condition", tables / "niah_by_condition_model.csv", 500),
        ("RULER_Task", tables / "ruler_by_task_model.csv", 250),
        ("RULER_Length", tables / "ruler_by_task_length_model.csv", 500),
        ("RULER_ECL", tables / "ruler_effective_context.csv", 300),
        ("Code_Inventory", tables / "code_inventory.csv", 200),
    ]
    sheets = []
    for name, path, max_rows in sheet_specs:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        sheets.append({"name": name, "rows": dataframe_to_rows(df.head(max_rows))})
    payload = {
        "title": "Public Benchmark Summary",
        "generated_at": fmt_time(time.time()),
        "sheets": sheets,
        "figures": {key: str(path.relative_to(output)) for key, path in outputs.items() if path.suffix.lower() == ".png"},
    }
    path = output / "workbook_payload.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def dataframe_to_rows(df: pd.DataFrame) -> list[list[Any]]:
    rows = [df.columns.tolist()]
    for record in df.where(pd.notna(df), None).to_dict(orient="records"):
        rows.append([json_safe(record.get(col)) for col in df.columns])
    return rows


def dataframe_to_markdown(df: pd.DataFrame, max_rows: int = 40) -> str:
    if df.empty:
        return "_No rows._"
    view = df.head(max_rows).copy()
    headers = [str(col) for col in view.columns]
    body = []
    for record in view.where(pd.notna(view), "").to_dict(orient="records"):
        body.append([markdown_cell(record.get(col)) for col in view.columns])
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
    if len(df) > max_rows:
        lines.append("| " + " | ".join(["..."] * len(headers)) + " |")
    return "\n".join(lines)


def markdown_cell(value: Any) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.4f}"
    text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if pd.isna(value):
        return None
    return value


def write_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].round(6)
    out.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def infer_longbench_category(task: str) -> str:
    if task in {"narrativeqa", "qasper", "multifieldqa_en"}:
        return "single_doc_qa"
    if task in {"hotpotqa", "2wikimqa", "musique"}:
        return "multi_doc_qa"
    if task in {"gov_report", "qmsum", "multi_news"}:
        return "summarization"
    return ""


def choose_length(df: pd.DataFrame) -> float:
    counts = df.groupby("length", dropna=True).size().sort_values(ascending=False)
    if counts.empty:
        return float(df["length"].dropna().iloc[0])
    preferred = [32768, 65536, 16384, 4096]
    available = set(int(x) for x in counts.index if not pd.isna(x))
    for length in preferred:
        if length in available:
            return float(length)
    return float(counts.index[0])


def sorted_dropna(values: pd.Series) -> list[Any]:
    vals = [x for x in values.dropna().unique().tolist() if x != ""]
    return sorted(vals, key=lambda x: (0, float(x)) if is_number(x) else (1, str(x)))


def normalize_number(value: Any) -> Any:
    number = to_float(value)
    if number is None:
        return np.nan
    if abs(number - int(number)) < 1e-9:
        return int(number)
    return number


def to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def is_number(value: Any) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def short_label(value: Any) -> str:
    text = str(value)
    replacements = {
        "qwen35_": "Q3.5-",
        "qwen3_": "Q3-",
        "_no_thinking": "-noT",
        "_thinking": "-T",
        "_a": "-A",
        "hunyuan_a13b": "Hunyuan-A13B",
        "seed_oss_36b": "Seed-OSS-36B",
        "common_words_extraction": "common words",
        "freq_words_extraction": "freq words",
        "variable_tracking": "var tracking",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    if len(text) > 28:
        text = text[:25] + "..."
    return text


def latest_mtime(files: list[Path]) -> str:
    if not files:
        return ""
    return fmt_time(max(path.stat().st_mtime for path in files))


def fmt_time(value: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(value))


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        return sum(1 for _ in f)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    main()
