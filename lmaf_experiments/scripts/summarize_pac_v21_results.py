from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_ORDER = [
    "qwen35_9b",
    "qwen3_8b",
    "qwen35_27b",
    "qwen35_35b_a3b",
    "qwen35_122b_a10b",
    "qwen3_14b_no_thinking",
    "qwen3_14b_thinking",
    "seed_oss_36b",
]

DISPLAY_EXCLUDED_MODELS = {
    "hunyuan_a13b",
    "tencent/Hunyuan-A13B-Instruct",
}

SUBSET_ORDER = [
    "PAC-A_position",
    "PAC-B_interference",
    "PAC-C_binding_capacity",
    "PAC-D_multihop_false_chain",
]

SUBSET_LABELS = {
    "PAC-A_position": "PAC-A position",
    "PAC-B_interference": "PAC-B interference",
    "PAC-C_binding_capacity": "PAC-C binding",
    "PAC-D_multihop_false_chain": "PAC-D v2.1 multihop",
}

DATASET_FILES = {
    "PAC-A_position": ROOT / "PAC" / "data" / "PAC-A_position" / "samples.jsonl",
    "PAC-B_interference": ROOT / "PAC" / "data" / "PAC-B_interference" / "samples.jsonl",
    "PAC-C_binding_capacity": ROOT / "PAC" / "data" / "PAC-C_binding_capacity" / "samples.jsonl",
    "PAC-D_multihop_false_chain": ROOT / "PAC" / "data" / "PAC-D_v2_1_hard" / "samples.jsonl",
}

TABLE_EXPLANATIONS = {
    "dataset_summary": "说明 PAC v2.1 各子集的数据规模、条件设置和样本文件位置，用于交代正式实验设计和每个子集承担的诊断目标。",
    "raw_result_index": "逐样本结果索引，保留答案、预测、错误类型和来源文件，便于回查热力图或曲线中任一分数背后的原始输出。",
    "summary_by_subset_model": "按 PAC 子集和模型聚合，展示准确率、覆盖率、字段准确率、诱饵捕获率等指标，是论文主表的核心来源。",
    "summary_by_condition_model": "按子集条件进一步拆分模型表现，例如位置、干扰数、绑定容量和多跳假链设置，用于画条件曲线和热力图。",
    "summary_by_model_overall": "按模型汇总所有已完成 PAC v2.1 样本，同时给出 success-only 和保守准确率，便于比较总体抗干扰能力。",
    "summary_by_subset_overall": "按 PAC-A/B/C/D 子集汇总整体难度和覆盖率，帮助判断哪个子集最能区分模型、哪个子集仍需补跑。",
    "error_types": "统计 correct、partial、decoy capture、omission、request error 等错误类型，说明模型失败来自绑定错乱还是 API 未返回。",
    "error_examples": "每类典型错误抽样，展示模型如何答错，例如字段顺序错误、捕获干扰值或只答出部分三元组。",
    "pac_A_position_accuracy_pivot": "PAC-A 的位置效应透视表，比较 10%、25%、50%、75%、90% 位置下各模型在高相似干扰中的准确率。",
    "pac_B_interference_accuracy_pivot": "PAC-B 的干扰密度透视表，展示 decoy 数增加时模型准确率变化，可用于寻找临界干扰强度和鲁棒性斜率。",
    "pac_C_binding_accuracy_pivot": "PAC-C 的实体绑定容量透视表，比较不同 K/Q 设置下模型能否保持多实体属性绑定关系。",
    "pac_D_v21_multihop_accuracy_pivot": "PAC-D v2.1 的多跳假链透视表，展示 hop 数和假链数变化时模型链路追踪与最终字段绑定稳定性。",
    "table_explanations": "本页汇总 PAC v2.1 各数据表用途说明，方便阅读者把表格指标和论文中的实验问题对应起来。",
}


def main() -> None:
    args = parse_args()
    output = (ROOT / args.output).resolve()
    tables_dir = output / "tables"
    figures_dir = output / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    rows = load_result_rows((ROOT / args.input_root).resolve())
    if not rows:
        raise SystemExit("No PAC v2.1 result rows found.")
    df = normalize_rows(dedupe_rows(rows))
    datasets = load_dataset_summary()

    tables: dict[str, Path] = {}
    tables["dataset_summary"] = write_csv(datasets, tables_dir / "dataset_summary.csv")
    tables["raw_result_index"] = write_csv(raw_result_index(df), tables_dir / "raw_result_index.csv")
    subset_model = summarize(df, ["subset", "model"], ["subset", "model"])
    tables["summary_by_subset_model"] = write_csv(subset_model, tables_dir / "summary_by_subset_model.csv")
    condition_model = summarize(df, ["subset", "condition_name", "condition_value", "model"], ["subset", "condition_sort", "model"])
    tables["summary_by_condition_model"] = write_csv(condition_model.drop(columns=["condition_sort"]), tables_dir / "summary_by_condition_model.csv")
    model_overall = summarize(df, ["model"], ["accuracy_all_conservative"], ascending=[False])
    tables["summary_by_model_overall"] = write_csv(model_overall, tables_dir / "summary_by_model_overall.csv")
    subset_overall = summarize(df, ["subset"], ["subset"])
    tables["summary_by_subset_overall"] = write_csv(subset_overall, tables_dir / "summary_by_subset_overall.csv")
    errors = summarize_errors(df)
    tables["error_types"] = write_csv(errors, tables_dir / "error_types.csv")
    tables.update(write_pivot_tables(condition_model, tables_dir))
    tables["error_examples"] = write_csv(error_examples(df), tables_dir / "error_examples.csv")
    tables["table_explanations"] = write_csv(table_explanations(), tables_dir / "table_explanations.csv")

    figures: dict[str, Path] = {}
    figures["fig_subset_accuracy"] = heatmap(
        subset_model,
        row="model",
        col="subset",
        value="accuracy_success",
        row_order=MODEL_ORDER,
        col_order=SUBSET_ORDER,
        title="PAC v2.1 success-only accuracy by subset and model",
        output=figures_dir / "pac_v21_subset_accuracy_heatmap.png",
    )
    figures["fig_subset_accuracy_all"] = heatmap(
        subset_model,
        row="model",
        col="subset",
        value="accuracy_all_conservative",
        row_order=MODEL_ORDER,
        col_order=SUBSET_ORDER,
        title="PAC v2.1 conservative accuracy by subset and model",
        output=figures_dir / "pac_v21_subset_accuracy_all_heatmap.png",
    )
    figures["fig_subset_coverage"] = heatmap(
        subset_model,
        row="model",
        col="subset",
        value="coverage",
        row_order=MODEL_ORDER,
        col_order=SUBSET_ORDER,
        title="PAC v2.1 coverage by subset and model",
        output=figures_dir / "pac_v21_subset_coverage_heatmap.png",
        fmt=".0%",
    )
    figures["fig_decoy_capture"] = heatmap(
        subset_model,
        row="model",
        col="subset",
        value="decoy_capture_rate",
        row_order=MODEL_ORDER,
        col_order=SUBSET_ORDER,
        title="PAC v2.1 decoy capture rate by subset and model",
        output=figures_dir / "pac_v21_decoy_capture_heatmap.png",
    )
    figures["fig_model_ranking"] = horizontal_bar(
        model_overall,
        label_col="model",
        value_col="accuracy_all_conservative",
        title="PAC v2.1 overall conservative accuracy",
        output=figures_dir / "pac_v21_model_ranking_conservative.png",
        x_label="Conservative accuracy",
    )
    figures["fig_error_profile"] = error_type_stacked_bar(errors, figures_dir / "pac_v21_error_type_profile.png")
    figures["fig_pac_a_position"] = line_by_condition(
        condition_model[condition_model["subset"] == "PAC-A_position"],
        title="PAC-A accuracy by target position",
        output=figures_dir / "pac_A_position_accuracy_lines.png",
        x_label="Target position (%)",
    )
    figures["fig_pac_b_interference"] = line_by_condition(
        condition_model[condition_model["subset"] == "PAC-B_interference"],
        title="PAC-B accuracy by decoy count",
        output=figures_dir / "pac_B_interference_accuracy_lines.png",
        x_label="High-similarity decoy count",
    )
    figures["fig_pac_c_binding"] = heatmap(
        condition_model[condition_model["subset"] == "PAC-C_binding_capacity"],
        row="model",
        col="condition_value",
        value="accuracy_success",
        row_order=MODEL_ORDER,
        col_order=["16/3", "16/5", "16/8", "32/3", "32/5", "32/8", "64/3", "64/5", "64/8"],
        title="PAC-C binding capacity accuracy",
        output=figures_dir / "pac_C_binding_capacity_heatmap.png",
    )
    figures["fig_pac_d_multihop"] = heatmap(
        condition_model[condition_model["subset"] == "PAC-D_multihop_false_chain"],
        row="model",
        col="condition_value",
        value="accuracy_success",
        row_order=MODEL_ORDER,
        col_order=["4/16", "4/32", "5/16", "5/32", "6/16", "6/32"],
        title="PAC-D v2.1 multihop false-chain accuracy",
        output=figures_dir / "pac_D_v21_multihop_heatmap.png",
    )
    figures["fig_field_vs_exact"] = field_vs_exact_scatter(
        subset_model,
        figures_dir / "pac_v21_field_vs_exact_scatter.png",
    )

    report_path = write_readme(output, tables, figures, df, subset_model, model_overall)
    payload_path = write_workbook_payload(output, tables, figures)
    print(f"Wrote PAC v2.1 summary to {output}")
    print(f"report: {report_path}")
    print(f"workbook_payload: {payload_path}")
    for key, path in sorted(tables.items()):
        print(f"{key}: {path}")
    for key, path in sorted(figures.items()):
        print(f"{key}: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create paper-style tables and figures for PAC v2.1 results.")
    parser.add_argument("--input-root", default="results/raw/pac_v21_queue/pac_v21_full_queue")
    parser.add_argument("--output", default="results/reports/pac_v21_all_summary")
    return parser.parse_args()


def load_result_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("*/*.jsonl")):
        subset = path.parent.name
        model = path.stem
        for row in iter_jsonl(path):
            row["_source_file"] = str(path.relative_to(ROOT))
            row.setdefault("formal_subset", subset)
            row.setdefault("model", model)
            rows.append(row)
    return rows


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
                    "formal_subset": path.parent.name,
                    "model": path.stem,
                    "error": f"JSONDecodeError: {exc}",
                    "error_type": "json_decode_error",
                }


def dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[tuple[str, str, str], tuple[tuple[int, int], dict[str, Any]]] = {}
    for idx, row in enumerate(rows):
        key = (str(row.get("formal_subset") or ""), str(row.get("model") or ""), str(row.get("sample_id") or ""))
        if not all(key):
            continue
        rank = (1 if row.get("error") in (None, "") else 0, idx)
        current = best.get(key)
        if current is None or rank >= current[0]:
            best[key] = (rank, row)
    return [item[1] for item in best.values()]


def normalize_rows(rows: list[dict[str, Any]]) -> pd.DataFrame:
    out = []
    for row in rows:
        subset = str(row.get("formal_subset") or "")
        model = str(row.get("model") or "")
        api_model = str(row.get("api_model") or "")
        if model in DISPLAY_EXCLUDED_MODELS or api_model in DISPLAY_EXCLUDED_MODELS:
            continue
        error = row.get("error")
        success = error in (None, "")
        score = to_float(row.get("score"))
        field_accuracy = to_float(row.get("field_accuracy"))
        if field_accuracy is None:
            field_accuracy = score
        error_type = row.get("error_type")
        if not error_type:
            error_type = "correct" if success and score == 1 else ("request_error" if not success else "other_error")
        condition_name, condition_value, condition_sort = infer_condition(row, subset)
        out.append(
            {
                "subset": subset,
                "model": model,
                "sample_id": str(row.get("sample_id") or ""),
                "success": bool(success),
                "score": score if success else np.nan,
                "score_all": score if success and score is not None else 0.0,
                "field_accuracy": field_accuracy if success else np.nan,
                "field_accuracy_all": field_accuracy if success and field_accuracy is not None else 0.0,
                "error_type": str(error_type),
                "error": "" if success else str(error),
                "condition_name": condition_name,
                "condition_value": condition_value,
                "condition_sort": condition_sort,
                "position": to_float(row.get("position_bin") or row.get("position_percent")),
                "decoy_count": to_float(row.get("decoy_count")),
                "binding_k": to_float(row.get("binding_k")),
                "query_count": to_float(row.get("query_count")),
                "hop_count": to_float(row.get("hop_count")),
                "false_chain_count": to_float(row.get("false_chain_count")),
                "latency_sec": to_float(row.get("latency_sec")),
                "prompt_tokens": to_float(row.get("prompt_tokens")),
                "completion_tokens": to_float(row.get("completion_tokens")),
                "answer": str(row.get("answer") or ""),
                "prediction": str(row.get("prediction") or ""),
                "source_file": str(row.get("_source_file") or ""),
            }
        )
    df = pd.DataFrame(out)
    df["model"] = pd.Categorical(df["model"], categories=MODEL_ORDER, ordered=True)
    df["subset"] = pd.Categorical(df["subset"], categories=SUBSET_ORDER, ordered=True)
    return df.sort_values(["subset", "model", "condition_sort", "sample_id"], kind="stable")


def infer_condition(row: dict[str, Any], subset: str) -> tuple[str, str, float]:
    if subset == "PAC-A_position":
        value = row.get("position_bin") or row.get("position_percent")
        return "position", int_or_text(value), float(to_float(value) or 0)
    if subset == "PAC-B_interference":
        value = row.get("decoy_count")
        return "decoy_count", int_or_text(value), float(to_float(value) or 0)
    if subset == "PAC-C_binding_capacity":
        k = int_or_text(row.get("binding_k"))
        q = int_or_text(row.get("query_count"))
        return "binding_k/query_count", f"{k}/{q}", float(to_float(row.get("binding_k")) or 0) * 100 + float(to_float(row.get("query_count")) or 0)
    if subset == "PAC-D_multihop_false_chain":
        h = int_or_text(row.get("hop_count"))
        f = int_or_text(row.get("false_chain_count"))
        return "hop_count/false_chain_count", f"{h}/{f}", float(to_float(row.get("hop_count")) or 0) * 100 + float(to_float(row.get("false_chain_count")) or 0)
    return "unknown", "", 0.0


def load_dataset_summary() -> pd.DataFrame:
    rows = []
    for subset, path in DATASET_FILES.items():
        samples = list(iter_jsonl(path)) if path.exists() else []
        if not samples:
            rows.append({"subset": subset, "sample_count": 0, "condition_count": 0, "conditions": ""})
            continue
        conditions = sorted({infer_condition(row, subset)[1] for row in samples}, key=condition_sort_key)
        rows.append(
            {
                "subset": subset,
                "label": SUBSET_LABELS.get(subset, subset),
                "sample_count": len(samples),
                "condition_count": len(conditions),
                "conditions": ", ".join(conditions),
                "data_file": str(path.relative_to(ROOT)),
            }
        )
    return pd.DataFrame(rows)


def raw_result_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["prediction_excerpt"] = out["prediction"].map(lambda x: truncate_text(x, 500))
    out["answer_excerpt"] = out["answer"].map(lambda x: truncate_text(x, 300))
    out["error_excerpt"] = out["error"].map(lambda x: truncate_text(x, 300))
    keep = [
        "subset",
        "model",
        "sample_id",
        "condition_name",
        "condition_value",
        "success",
        "score",
        "score_all",
        "field_accuracy",
        "error_type",
        "latency_sec",
        "prompt_tokens",
        "completion_tokens",
        "answer_excerpt",
        "prediction_excerpt",
        "error_excerpt",
        "source_file",
    ]
    return out[keep]


def summarize(df: pd.DataFrame, group_cols: list[str], sort_cols: list[str], ascending: list[bool] | None = None) -> pd.DataFrame:
    grouped = df.groupby(group_cols, dropna=False, observed=True)
    out = grouped.agg(
        n_total=("sample_id", "count"),
        n_eval=("success", "sum"),
        n_api_error=("success", lambda s: int((~s.astype(bool)).sum())),
        correct=("score_all", "sum"),
        field_correct=("field_accuracy_all", "sum"),
        accuracy_success=("score", "mean"),
        accuracy_all_conservative=("score_all", "mean"),
        mean_field_accuracy=("field_accuracy", "mean"),
        mean_field_accuracy_all=("field_accuracy_all", "mean"),
        decoy_capture_count=("error_type", lambda s: int((s == "decoy_value_capture").sum())),
        partial_count=("error_type", lambda s: int((s == "partial_triplet").sum())),
        omission_count=("error_type", lambda s: int((s == "omission").sum())),
        near_miss_count=("error_type", lambda s: int((s == "near_miss_value").sum())),
        mean_latency_sec=("latency_sec", "mean"),
        prompt_tokens_mean=("prompt_tokens", "mean"),
        completion_tokens_mean=("completion_tokens", "mean"),
        condition_sort=("condition_sort", "min"),
    ).reset_index()
    out["coverage"] = out["n_eval"] / out["n_total"].replace(0, np.nan)
    out["api_error_rate"] = out["n_api_error"] / out["n_total"].replace(0, np.nan)
    out["decoy_capture_rate"] = out["decoy_capture_count"] / out["n_eval"].replace(0, np.nan)
    out["partial_rate"] = out["partial_count"] / out["n_eval"].replace(0, np.nan)
    out["omission_rate"] = out["omission_count"] / out["n_eval"].replace(0, np.nan)
    out["near_miss_rate"] = out["near_miss_count"] / out["n_eval"].replace(0, np.nan)
    preferred = group_cols + [
        "n_total",
        "n_eval",
        "n_api_error",
        "coverage",
        "api_error_rate",
        "accuracy_success",
        "accuracy_all_conservative",
        "mean_field_accuracy",
        "mean_field_accuracy_all",
        "decoy_capture_rate",
        "partial_rate",
        "omission_rate",
        "near_miss_rate",
        "mean_latency_sec",
        "prompt_tokens_mean",
        "completion_tokens_mean",
        "condition_sort",
    ]
    out = out[preferred]
    if ascending is None:
        ascending = [True] * len(sort_cols)
    return out.sort_values(sort_cols, ascending=ascending, kind="stable")


def summarize_errors(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby(["subset", "model", "error_type"], dropna=False, observed=True)
        .size()
        .reset_index(name="count")
        .sort_values(["subset", "model", "error_type"], kind="stable")
    )
    return out[out["count"] > 0]


def write_pivot_tables(condition_model: pd.DataFrame, tables_dir: Path) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    for subset, filename in [
        ("PAC-A_position", "pac_A_position_accuracy_pivot.csv"),
        ("PAC-B_interference", "pac_B_interference_accuracy_pivot.csv"),
        ("PAC-C_binding_capacity", "pac_C_binding_accuracy_pivot.csv"),
        ("PAC-D_multihop_false_chain", "pac_D_v21_multihop_accuracy_pivot.csv"),
    ]:
        part = condition_model[condition_model["subset"].astype(str) == subset]
        if part.empty:
            continue
        pivot = part.pivot_table(index="model", columns="condition_value", values="accuracy_success", aggfunc="mean", observed=True)
        pivot = pivot.reindex(index=[x for x in MODEL_ORDER if x in pivot.index])
        pivot = pivot.reindex(columns=sorted(pivot.columns, key=condition_sort_key))
        outputs[filename.replace(".csv", "")] = write_csv(pivot.reset_index(), tables_dir / filename)
    return outputs


def error_examples(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    wrong = df[(df["success"]) & (df["score"].fillna(0) < 1)].copy()
    for (subset, model, error_type), part in wrong.groupby(["subset", "model", "error_type"], observed=False):
        for row in part.head(2).itertuples(index=False):
            rows.append(
                {
                    "subset": subset,
                    "model": model,
                    "error_type": error_type,
                    "sample_id": row.sample_id,
                    "condition": f"{row.condition_name}={row.condition_value}",
                    "answer": truncate_text(row.answer, 200),
                    "prediction": truncate_text(row.prediction, 300),
                }
            )
    return pd.DataFrame(rows)


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
    width = max(8.5, 1.45 * len(col_order) + 3.2)
    height = max(5.0, 0.5 * len(row_order) + 1.8)
    fig, ax = plt.subplots(figsize=(width, height), dpi=180)
    masked = np.ma.masked_invalid(data)
    cmap = matplotlib.colormaps["YlGnBu"].copy()
    cmap.set_bad("#F3F4F6")
    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_title(title, fontsize=12, weight="bold", pad=12)
    ax.set_xticks(np.arange(len(col_order)))
    ax.set_xticklabels([short_label(x) for x in col_order], rotation=25, ha="right", fontsize=8)
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
                color = "white" if val >= 0.62 else "#111827"
                ax.text(j, i, format(val, fmt), ha="center", va="center", fontsize=7, color=color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.ax.tick_params(labelsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def line_by_condition(df: pd.DataFrame, title: str, output: Path, x_label: str) -> Path:
    fig, ax = plt.subplots(figsize=(9.2, 5.4), dpi=180)
    for model in MODEL_ORDER:
        part = df[df["model"].astype(str) == model].dropna(subset=["accuracy_success"]).sort_values("condition_sort")
        if part.empty:
            continue
        x = [condition_sort_key(v) for v in part["condition_value"]]
        ax.plot(x, part["accuracy_success"], marker="o", linewidth=1.8, label=short_label(model))
    ax.set_title(title, fontsize=12, weight="bold", pad=12)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Success-only accuracy")
    ax.set_ylim(0, 1.05)
    ax.grid(color="#E5E7EB")
    ax.legend(fontsize=7, ncol=2, frameon=False)
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


def error_type_stacked_bar(errors: pd.DataFrame, output: Path) -> Path:
    if errors.empty:
        return output
    visible_types = ["correct", "partial_triplet", "decoy_value_capture", "omission", "other_error", "request_error"]
    pivot = errors.pivot_table(index="model", columns="error_type", values="count", aggfunc="sum", fill_value=0, observed=False)
    pivot = pivot.reindex(index=[x for x in MODEL_ORDER if x in pivot.index])
    cols = [c for c in visible_types if c in pivot.columns] + [c for c in pivot.columns if c not in visible_types]
    pivot = pivot[cols]
    totals = pivot.sum(axis=1).replace(0, np.nan)
    frac = pivot.div(totals, axis=0)
    fig, ax = plt.subplots(figsize=(9.2, 5.5), dpi=180)
    left = np.zeros(len(frac))
    palette = {
        "correct": "#2563EB",
        "partial_triplet": "#F59E0B",
        "decoy_value_capture": "#DC2626",
        "omission": "#6B7280",
        "request_error": "#8B5CF6",
        "other_error": "#10B981",
    }
    y = np.arange(len(frac.index))
    for col in frac.columns:
        vals = frac[col].fillna(0).to_numpy()
        ax.barh(y, vals, left=left, label=short_label(col), color=palette.get(col, "#94A3B8"))
        left += vals
    ax.set_yticks(y)
    ax.set_yticklabels([short_label(x) for x in frac.index], fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of rows")
    ax.set_title("PAC v2.1 result/error composition by model", fontsize=12, weight="bold", pad=12)
    ax.legend(fontsize=7, ncol=3, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.28))
    ax.grid(axis="x", color="#E5E7EB")
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def field_vs_exact_scatter(df: pd.DataFrame, output: Path) -> Path:
    part = df.dropna(subset=["accuracy_success", "mean_field_accuracy"]).copy()
    fig, ax = plt.subplots(figsize=(7.5, 5.2), dpi=180)
    for subset in SUBSET_ORDER:
        sub = part[part["subset"].astype(str) == subset]
        if sub.empty:
            continue
        ax.scatter(sub["mean_field_accuracy"], sub["accuracy_success"], s=44, label=short_label(subset), alpha=0.85)
    ax.plot([0, 1], [0, 1], color="#9CA3AF", linewidth=1, linestyle="--")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Mean field accuracy")
    ax.set_ylabel("Exact accuracy")
    ax.set_title("Field-level correctness vs exact triplet correctness", fontsize=12, weight="bold", pad=12)
    ax.grid(color="#E5E7EB")
    ax.legend(fontsize=7, frameon=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def write_readme(
    output: Path,
    tables: dict[str, Path],
    figures: dict[str, Path],
    df: pd.DataFrame,
    subset_model: pd.DataFrame,
    model_overall: pd.DataFrame,
) -> Path:
    total = len(df)
    eval_count = int(df["success"].sum())
    coverage = eval_count / total if total else float("nan")
    top = model_overall.dropna(subset=["accuracy_all_conservative"]).head(5)
    lines = [
        "# PAC v2.1 Result Summary",
        "",
        f"Generated at: `{fmt_time(time.time())}`",
        "",
        "## Scope",
        "",
        "This report summarizes the currently available PAC v2.1 queue results. Scores are separated into success-only accuracy and conservative accuracy, where API errors count as zero.",
        "",
        "## Run Coverage",
        "",
        f"- Total planned/evaluated rows in current raw files: `{total}`",
        f"- Successful API/evaluable rows: `{eval_count}`",
        f"- Coverage: `{coverage:.1%}`",
        "",
        "## Top Models By Conservative Accuracy",
        "",
        dataframe_to_markdown(top[["model", "n_total", "n_eval", "coverage", "accuracy_success", "accuracy_all_conservative"]]),
        "",
        "## Key Interpretation",
        "",
        "- PAC-A and PAC-B show clear degradation under position and high-similarity interference pressure.",
        "- PAC-C is the cleanest current binding-capacity signal.",
        "- PAC-D v2.1 is effective for exposing multihop field-binding failures, but the current sample size is small.",
        "- Low-coverage models should be interpreted cautiously until failed API rows are topped up.",
        "",
        "## Tables",
        "",
    ]
    for key, path in sorted(tables.items()):
        lines.append(f"- {key}: `{path.relative_to(output)}`")
    lines.extend(["", "## Figures", ""])
    for key, path in sorted(figures.items()):
        lines.append(f"- {key}: `{path.relative_to(output)}`")
    path = output / "README.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_workbook_payload(output: Path, tables: dict[str, Path], figures: dict[str, Path]) -> Path:
    sheet_specs = []
    for name, key, max_rows in [
        ("Dataset", "dataset_summary", 100),
        ("ModelOverall", "summary_by_model_overall", 100),
        ("SubsetModel", "summary_by_subset_model", 250),
        ("ConditionModel", "summary_by_condition_model", 500),
        ("Errors", "error_types", 300),
        ("A_Position", "pac_A_position_accuracy_pivot", 100),
        ("B_Interference", "pac_B_interference_accuracy_pivot", 100),
        ("C_Binding", "pac_C_binding_accuracy_pivot", 100),
        ("D_Multihop", "pac_D_v21_multihop_accuracy_pivot", 100),
        ("TableNotes", "table_explanations", 100),
        ("ErrorExamples", "error_examples", 200),
        ("RawIndex", "raw_result_index", 2000),
    ]:
        path = tables.get(key)
        if path and path.exists():
            df = pd.read_csv(path)
            sheet_specs.append({"name": name, "rows": dataframe_to_rows(df.head(max_rows))})
    payload = {
        "title": "PAC v2.1 Result Summary",
        "generated_at": fmt_time(time.time()),
        "sheets": sheet_specs,
        "figures": {key: str(path.relative_to(output)) for key, path in figures.items() if path.suffix.lower() == ".png"},
    }
    path = output / "workbook_payload.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].round(6)
    out.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def table_explanations() -> pd.DataFrame:
    return pd.DataFrame(
        [{"table": table, "explanation": explanation} for table, explanation in TABLE_EXPLANATIONS.items()]
    )


def dataframe_to_rows(df: pd.DataFrame) -> list[list[Any]]:
    rows = [df.columns.tolist()]
    for record in df.where(pd.notna(df), None).to_dict(orient="records"):
        rows.append([json_safe(record.get(col)) for col in df.columns])
    return rows


def dataframe_to_markdown(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "_No rows._"
    view = df.head(max_rows).copy()
    lines = [
        "| " + " | ".join(str(c) for c in view.columns) + " |",
        "| " + " | ".join(["---"] * len(view.columns)) + " |",
    ]
    for record in view.where(pd.notna(view), "").to_dict(orient="records"):
        cells = []
        for col in view.columns:
            value = record.get(col)
            if isinstance(value, float):
                cells.append(f"{value:.4f}")
            else:
                cells.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if pd.isna(value):
        return None
    return value


def to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def int_or_text(value: Any) -> str:
    number = to_float(value)
    if number is None:
        return str(value or "")
    if abs(number - int(number)) < 1e-9:
        return str(int(number))
    return str(number)


def condition_sort_key(value: Any) -> float:
    text = str(value)
    if "/" in text:
        left, _, right = text.partition("/")
        return (to_float(left) or 0) * 100 + (to_float(right) or 0)
    return to_float(text) or 0


def truncate_text(value: Any, limit: int) -> str:
    text = str(value or "").replace("\r", " ").replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def short_label(value: Any) -> str:
    text = str(value)
    replacements = {
        "PAC-A_position": "PAC-A",
        "PAC-B_interference": "PAC-B",
        "PAC-C_binding_capacity": "PAC-C",
        "PAC-D_multihop_false_chain": "PAC-D v2.1",
        "qwen35_": "Q3.5-",
        "qwen3_": "Q3-",
        "_no_thinking": "-noT",
        "_thinking": "-T",
        "_a": "-A",
        "hunyuan_a13b": "Hunyuan-A13B",
        "seed_oss_36b": "Seed-OSS-36B",
        "decoy_value_capture": "decoy capture",
        "partial_triplet": "partial",
        "request_error": "API error",
        "other_error": "other",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text[:28] + "..." if len(text) > 31 else text


def fmt_time(value: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(value))


if __name__ == "__main__":
    main()
