from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lmaf.data.longbench import LONGBENCH_CATEGORIES
from lmaf.utils.io import iter_jsonl_paths, read_jsonl
from lmaf.utils.models import is_excluded_model
from run_unfinished_fast import FRAMEWORK_MODELS


DEFAULT_ROOTS = ",".join(
    [
        "results/raw/official_budget_topup/official_budget_topup_main",
    ]
)

MODEL_ORDER = list(FRAMEWORK_MODELS)
LONG_BENCH_TASK_ORDER = [
    "narrativeqa",
    "qasper",
    "hotpotqa",
    "2wikimqa",
    "gov_report",
    "multi_news",
]
PAC_SUBTASK_ORDER = ["A_position", "B_interference", "C_overlap", "D_multihop"]
PAC_SUBTASK_LABELS = {
    "A_position": "A: Position",
    "B_interference": "B: Interference",
    "C_overlap": "C: Entity overlap",
    "D_multihop": "D: Multi-hop",
}


def main() -> None:
    args = parse_args()
    output = ROOT / args.output
    tables = output / "tables"
    figures = output / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args)
    if not rows:
        raise SystemExit("No LongBench/PAC rows found. Check --input-roots.")

    raw = pd.DataFrame(rows)
    raw.to_csv(tables / "raw_longbench_pac_dedup.csv", index=False, encoding="utf-8-sig")

    longbench = raw[raw["experiment"] == "longbench"].copy()
    pac = raw[raw["experiment"] == "pac"].copy()

    outputs: dict[str, Path] = {}
    if not longbench.empty:
        outputs.update(write_longbench_tables(longbench, tables))
        outputs.update(plot_longbench(longbench, figures))
    if not pac.empty:
        outputs.update(write_pac_tables(pac, tables))
        outputs.update(plot_pac(pac, figures))

    write_readme(output, raw, outputs)
    print(f"Wrote LongBench/PAC report assets to {output}")
    for key, path in sorted(outputs.items()):
        print(f"{key}: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create readable LongBench and PAC result tables/figures.")
    parser.add_argument("--input-roots", default=DEFAULT_ROOTS)
    parser.add_argument("--output", default="results/reports/longbench_pac")
    parser.add_argument("--include-excluded-models", action="store_true")
    parser.add_argument("--include-errors-in-main-score", action="store_true")
    return parser.parse_args()


def load_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    chosen: dict[tuple[str, str, str], dict[str, Any]] = {}
    source_rank = 0
    for root_value in parse_csv(args.input_roots):
        root = ROOT / root_value
        if not root.exists():
            source_rank += 1
            continue
        for path in iter_jsonl_paths(root):
            for row in read_jsonl(path):
                exp = str(row.get("experiment") or infer_experiment_from_path(path))
                if exp not in {"longbench", "pac"}:
                    continue
                model = str(row.get("model") or path.stem)
                api_model = str(row.get("api_model") or "")
                if not args.include_excluded_models and (is_excluded_model(model) or is_excluded_model(api_model)):
                    continue
                sample_id = str(row.get("sample_id") or "")
                if not sample_id:
                    continue
                normalized = normalize_row(row, path, source_rank, exp, model, args.include_errors_in_main_score)
                key = (exp, model, sample_id)
                previous = chosen.get(key)
                if previous is None or row_priority(normalized) > row_priority(previous):
                    chosen[key] = normalized
        source_rank += 1
    return list(chosen.values())


def normalize_row(
    row: dict[str, Any],
    path: Path,
    source_rank: int,
    experiment: str,
    model: str,
    include_errors_in_main_score: bool,
) -> dict[str, Any]:
    error = row.get("error")
    success = error in (None, "")
    score = to_float(row.get("score"))
    score_success = score if success else None
    score_all = score if (success and score is not None) else 0.0
    main_score = score_all if include_errors_in_main_score else score_success
    task = row.get("task") or row.get("subtask") or ""
    subtask = row.get("subtask") or task
    length = row.get("length_tokens_target") or row.get("total_length") or row.get("length")
    normalized = {
        "experiment": experiment,
        "model": model,
        "api_model": row.get("api_model"),
        "provider": row.get("provider"),
        "sample_id": row.get("sample_id"),
        "task": task,
        "category": row.get("category") or LONGBENCH_CATEGORIES.get(str(task), ""),
        "subtask": subtask,
        "subset": row.get("subset") or subset_from_subtask(str(subtask)),
        "length": normalize_number(length),
        "position": normalize_number(row.get("position_percent")),
        "density": normalize_number(row.get("density")),
        "interference_type": row.get("interference_type"),
        "similarity": row.get("similarity"),
        "distance": row.get("distance"),
        "hops": normalize_number(row.get("hops")),
        "hop_distance": row.get("hop_distance"),
        "chain_type": row.get("chain_type"),
        "metric": row.get("metric"),
        "score": score,
        "score_main": main_score,
        "score_all": score_all,
        "f1": to_float(row.get("f1") or row.get("partial_f1")),
        "rouge_l": to_float(row.get("rouge_l")),
        "latency_sec": to_float(row.get("latency_sec")),
        "error": error,
        "error_type": row.get("error_type"),
        "success": success,
        "source_file": str(path),
        "source_rank": source_rank,
        "budget_reused_from": row.get("budget_reused_from"),
    }
    return normalized


def row_priority(row: dict[str, Any]) -> tuple[int, int, int]:
    success = 1 if row.get("success") else 0
    has_score = 1 if row.get("score") is not None else 0
    source_rank_score = -int(row.get("source_rank") or 0)
    return (success, has_score, source_rank_score)


def write_longbench_tables(df: pd.DataFrame, tables: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    by_task = summarize(
        df,
        ["model", "category", "task"],
        sort_cols=["task", "model"],
    )
    by_model = summarize(df, ["model"], sort_cols=["score_mean"], ascending=[False])
    by_category = summarize(df, ["model", "category"], sort_cols=["category", "model"])

    paths["longbench_by_task_model"] = write_csv(by_task, tables / "longbench_by_task_model.csv")
    paths["longbench_by_model"] = write_csv(by_model, tables / "longbench_by_model.csv")
    paths["longbench_by_category_model"] = write_csv(by_category, tables / "longbench_by_category_model.csv")
    return paths


def write_pac_tables(df: pd.DataFrame, tables: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    by_subset = summarize(df, ["model", "subset", "subtask"], sort_cols=["subtask", "model"])
    paths["pac_by_subset_model"] = write_csv(by_subset, tables / "pac_by_subset_model.csv")

    subset_specs = {
        "A_position": ["model", "length", "position"],
        "B_interference": ["model", "length", "interference_type", "density"],
        "C_overlap": ["model", "length", "similarity", "distance"],
        "D_multihop": ["model", "length", "hops", "hop_distance"],
    }
    for subtask, group_cols in subset_specs.items():
        part = df[df["subtask"] == subtask]
        if part.empty:
            continue
        name = f"pac_{subtask}_details"
        paths[name] = write_csv(
            summarize(part, group_cols, sort_cols=group_cols),
            tables / f"{name}.csv",
        )
    return paths


def summarize(
    df: pd.DataFrame,
    group_cols: list[str],
    sort_cols: list[str],
    ascending: list[bool] | None = None,
) -> pd.DataFrame:
    grouped = df.groupby(group_cols, dropna=False)
    out = grouped.agg(
        n_total=("sample_id", "count"),
        n_success=("success", "sum"),
        n_error=("success", lambda s: int((~s.astype(bool)).sum())),
        score_mean=("score_main", "mean"),
        score_all=("score_all", "mean"),
        f1_mean=("f1", "mean"),
        rouge_l_mean=("rouge_l", "mean"),
        latency_mean=("latency_sec", "mean"),
    ).reset_index()
    out["coverage"] = out["n_success"] / out["n_total"]
    out["error_rate"] = out["n_error"] / out["n_total"]
    ordered = group_cols + [
        "n_total",
        "n_success",
        "n_error",
        "coverage",
        "error_rate",
        "score_mean",
        "score_all",
        "f1_mean",
        "rouge_l_mean",
        "latency_mean",
    ]
    out = out[ordered]
    if ascending is None:
        ascending = [True] * len(sort_cols)
    return out.sort_values(sort_cols, ascending=ascending, kind="stable")


def plot_longbench(df: pd.DataFrame, figures: Path) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    by_task = summarize(df, ["model", "task"], ["task", "model"])
    task_order = [task for task in LONG_BENCH_TASK_ORDER if task in set(by_task["task"])]
    outputs["fig_longbench_task_heatmap"] = heatmap(
        by_task,
        row="model",
        col="task",
        value="score_mean",
        row_order=MODEL_ORDER,
        col_order=task_order,
        title="LongBench score by task and model",
        output=figures / "longbench_task_heatmap.png",
        fmt=".2f",
    )

    by_category = summarize(df, ["model", "category"], ["category", "model"])
    category_order = [c for c in ["single_doc_qa", "multi_doc_qa", "summarization"] if c in set(by_category["category"])]
    outputs["fig_longbench_category_heatmap"] = heatmap(
        by_category,
        row="model",
        col="category",
        value="score_mean",
        row_order=MODEL_ORDER,
        col_order=category_order,
        title="LongBench score by task family",
        output=figures / "longbench_category_heatmap.png",
        fmt=".2f",
    )

    by_model = summarize(df, ["model"], ["score_mean"], ascending=[False])
    outputs["fig_longbench_model_ranking"] = horizontal_bar(
        by_model,
        label_col="model",
        value_col="score_mean",
        title="LongBench overall model ranking",
        output=figures / "longbench_model_ranking.png",
        x_label="Mean score",
    )
    return outputs


def plot_pac(df: pd.DataFrame, figures: Path) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    by_subset = summarize(df, ["model", "subtask"], ["subtask", "model"])
    outputs["fig_pac_subset_heatmap"] = heatmap(
        by_subset,
        row="model",
        col="subtask",
        value="score_mean",
        row_order=MODEL_ORDER,
        col_order=[x for x in PAC_SUBTASK_ORDER if x in set(by_subset["subtask"])],
        col_labels=PAC_SUBTASK_LABELS,
        title="PAC accuracy by subset and model",
        output=figures / "pac_subset_heatmap.png",
        fmt=".2f",
    )

    part = df[df["subtask"] == "A_position"]
    if not part.empty:
        a_pos = summarize(part, ["model", "position"], ["position", "model"])
        outputs["fig_pac_A_position_heatmap"] = heatmap(
            a_pos,
            row="model",
            col="position",
            value="score_mean",
            row_order=MODEL_ORDER,
            col_order=sorted_dropna(a_pos["position"]),
            title="PAC-A position effect (avg across lengths)",
            output=figures / "pac_A_position_heatmap.png",
            fmt=".2f",
        )
        a_len = summarize(part, ["model", "length"], ["length", "model"])
        outputs["fig_pac_A_length_heatmap"] = heatmap(
            a_len,
            row="model",
            col="length",
            value="score_mean",
            row_order=MODEL_ORDER,
            col_order=sorted_dropna(a_len["length"]),
            title="PAC-A accuracy by context length",
            output=figures / "pac_A_length_heatmap.png",
            fmt=".2f",
        )

    part = df[df["subtask"] == "B_interference"]
    if not part.empty:
        for interference_type in sorted(str(x) for x in part["interference_type"].dropna().unique()):
            sub = part[part["interference_type"].astype(str) == interference_type]
            b_den = summarize(sub, ["model", "density"], ["density", "model"])
            outputs[f"fig_pac_B_density_{safe_name(interference_type)}"] = heatmap(
                b_den,
                row="model",
                col="density",
                value="score_mean",
                row_order=MODEL_ORDER,
                col_order=sorted_dropna(b_den["density"]),
                title=f"PAC-B interference density: {interference_type}",
                output=figures / f"pac_B_density_{safe_name(interference_type)}.png",
                fmt=".2f",
            )

    part = df[df["subtask"] == "C_overlap"]
    if not part.empty:
        c_model = summarize(part, ["model"], ["score_mean"], ascending=[False])
        outputs["fig_pac_C_model_ranking"] = horizontal_bar(
            c_model,
            label_col="model",
            value_col="score_mean",
            title="PAC-C entity overlap model ranking",
            output=figures / "pac_C_model_ranking.png",
            x_label="Accuracy",
        )
        c_cond = summarize(part, ["similarity", "distance"], ["similarity", "distance"])
        outputs["fig_pac_C_condition_heatmap"] = heatmap(
            c_cond,
            row="similarity",
            col="distance",
            value="score_mean",
            row_order=sorted_dropna(c_cond["similarity"]),
            col_order=sorted_dropna(c_cond["distance"]),
            title="PAC-C accuracy by entity similarity and distance",
            output=figures / "pac_C_condition_heatmap.png",
            fmt=".2f",
        )

    part = df[df["subtask"] == "D_multihop"]
    if not part.empty:
        d_hops = summarize(part, ["model", "hops"], ["hops", "model"])
        outputs["fig_pac_D_hops_heatmap"] = heatmap(
            d_hops,
            row="model",
            col="hops",
            value="score_mean",
            row_order=MODEL_ORDER,
            col_order=sorted_dropna(d_hops["hops"]),
            title="PAC-D multi-hop accuracy by hop count",
            output=figures / "pac_D_hops_heatmap.png",
            fmt=".2f",
        )
        d_dist = summarize(part, ["model", "hop_distance"], ["hop_distance", "model"])
        outputs["fig_pac_D_distance_heatmap"] = heatmap(
            d_dist,
            row="model",
            col="hop_distance",
            value="score_mean",
            row_order=MODEL_ORDER,
            col_order=sorted_dropna(d_dist["hop_distance"]),
            title="PAC-D multi-hop accuracy by fact-chain distance",
            output=figures / "pac_D_distance_heatmap.png",
            fmt=".2f",
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
    fmt: str = ".2f",
    col_labels: dict[Any, str] | None = None,
) -> Path:
    pivot = df.pivot_table(index=row, columns=col, values=value, aggfunc="mean")
    row_order = [x for x in row_order if x in pivot.index]
    col_order = [x for x in col_order if x in pivot.columns]
    pivot = pivot.reindex(index=row_order, columns=col_order)

    height = max(4.8, 0.46 * max(1, len(row_order)) + 1.7)
    width = max(7.5, 1.15 * max(1, len(col_order)) + 3.0)
    fig, ax = plt.subplots(figsize=(width, height), dpi=180)
    data = pivot.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(data)
    cmap = matplotlib.colormaps["YlGnBu"].copy()
    cmap.set_bad("#F3F4F6")
    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_title(title, fontsize=12, pad=12, weight="bold")
    ax.set_xticks(np.arange(len(col_order)))
    labels = [col_labels.get(x, x) if col_labels else x for x in col_order]
    ax.set_xticklabels([short_label(x) for x in labels], rotation=30, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(row_order)))
    ax.set_yticklabels([short_label(x) for x in row_order], fontsize=8)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(col_order), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_order), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    for i in range(len(row_order)):
        for j in range(len(col_order)):
            val = data[i, j]
            if not math.isnan(val):
                color = "white" if val >= 0.62 else "#111827"
                ax.text(j, i, format(val, fmt), ha="center", va="center", fontsize=7, color=color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label("Score / Accuracy", fontsize=8)
    save_figure(fig, output)
    return output


def horizontal_bar(df: pd.DataFrame, label_col: str, value_col: str, title: str, output: Path, x_label: str) -> Path:
    plot_df = df[[label_col, value_col]].dropna().sort_values(value_col, ascending=True)
    height = max(4.6, 0.45 * len(plot_df) + 1.4)
    fig, ax = plt.subplots(figsize=(8.5, height), dpi=180)
    y = np.arange(len(plot_df))
    vals = plot_df[value_col].astype(float).to_numpy()
    ax.barh(y, vals, color="#2563EB")
    ax.set_yticks(y)
    ax.set_yticklabels([short_label(x) for x in plot_df[label_col]], fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel(x_label, fontsize=9)
    ax.set_title(title, fontsize=12, pad=12, weight="bold")
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.8)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    for yi, val in zip(y, vals):
        ax.text(min(0.99, val + 0.015), yi, f"{val:.2f}", va="center", fontsize=8)
    save_figure(fig, output)
    return output


def write_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].round(4)
    out.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def write_readme(output: Path, raw: pd.DataFrame, outputs: dict[str, Path]) -> None:
    counts = raw.groupby(["experiment", "model"], dropna=False).size().reset_index(name="rows")
    total_by_exp = raw.groupby("experiment").size().to_dict()
    lines = [
        "# LongBench / PAC Result Report",
        "",
        "Generated from deduplicated JSONL results. Duplicate rows prefer successful records and newer top-up outputs.",
        "",
        "## Row Counts",
        "",
    ]
    for exp, count in sorted(total_by_exp.items()):
        lines.append(f"- {exp}: {count} rows")
    lines.extend(["", "## Files", ""])
    for key, path in sorted(outputs.items()):
        lines.append(f"- {key}: {path.relative_to(output)}")
    lines.extend(["", "## Model Coverage", ""])
    lines.append(counts.to_markdown(index=False))
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")


def save_figure(fig: Any, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def sorted_dropna(values: pd.Series) -> list[Any]:
    vals = [x for x in values.dropna().unique().tolist() if x != ""]
    return sorted(vals, key=lambda x: (str(type(x)), float(x) if is_number(x) else str(x)))


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def subset_from_subtask(subtask: str) -> str:
    return {
        "A_position": "A",
        "B_interference": "B",
        "C_overlap": "C",
        "D_multihop": "D",
    }.get(subtask, "")


def infer_experiment_from_path(path: Path) -> str:
    parts = [part.lower() for part in path.parts]
    if any("longbench" in part for part in parts):
        return "longbench"
    if any(part == "pac" or "pac_" in part for part in parts):
        return "pac"
    return ""


def normalize_number(value: Any) -> Any:
    number = to_float(value)
    if number is None:
        return None
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


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value).strip("_").lower()


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
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    if len(text) > 24:
        text = text[:21] + "..."
    return text


if __name__ == "__main__":
    main()
