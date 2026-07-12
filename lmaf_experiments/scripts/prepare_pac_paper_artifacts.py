from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "results" / "reports" / "pac_paper_ready"
FIGURES = REPORT / "figures"
TABLES = REPORT / "tables"

SOURCE_TABLES = ROOT / "results" / "reports" / "pac_v21_full_no_hunyuan_queue" / "pac_v21_full_queue"

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

MODEL_LABELS = {
    "qwen35_9b": "Qwen3.5-9B",
    "qwen3_8b": "Qwen3-8B",
    "qwen35_27b": "Qwen3.5-27B",
    "qwen35_35b_a3b": "Qwen3.5-35B-A3B",
    "qwen35_122b_a10b": "Qwen3.5-122B-A10B",
    "qwen3_14b_no_thinking": "Qwen3-14B-noT",
    "qwen3_14b_thinking": "Qwen3-14B-T",
    "seed_oss_36b": "Seed-OSS-36B",
}

ARCH_LABELS = {
    "qwen35_9b": "Dense 9B",
    "qwen3_8b": "Dense 8B",
    "qwen35_27b": "Dense 27B",
    "qwen35_35b_a3b": "MoE 35B/3B",
    "qwen35_122b_a10b": "MoE 122B/10B",
    "qwen3_14b_no_thinking": "Dense 14B",
    "qwen3_14b_thinking": "Dense 14B",
    "seed_oss_36b": "Dense 36B",
}

SUBSET_ORDER = [
    "PAC-A_position",
    "PAC-B_interference",
    "PAC-C_binding_capacity",
    "PAC-D_multihop_false_chain",
]

SUBSET_LABELS = {
    "PAC-A_position": "PAC-A\n位置效应",
    "PAC-B_interference": "PAC-B\n干扰密度",
    "PAC-C_binding_capacity": "PAC-C\n绑定容量",
    "PAC-D_multihop_false_chain": "PAC-D v2.1\n多跳假链",
}

PALETTE = [
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#E45756",
    "#72B7B2",
    "#B279A2",
    "#9D755D",
    "#BAB0AC",
]


def main() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    subset = read_csv("summary_by_subset_model.csv")
    condition = read_csv("summary_by_condition_model.csv")
    subset = subset[subset["model"].isin(MODEL_ORDER)].copy()
    condition = condition[condition["model"].isin(MODEL_ORDER)].copy()

    summary_rows = build_summary_rows(subset)
    chart_specs = build_figures(subset, condition)
    payload = {
        "table_workbook": build_table_payload(summary_rows),
        "chart_workbook": build_chart_payload(chart_specs),
        "output_dir": str(REPORT),
        "desktop_dir": str(Path.home() / "Desktop"),
    }
    (REPORT / "pac_paper_payload.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(summary_rows).to_csv(TABLES / "pac_paper_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(chart_specs).to_csv(TABLES / "pac_chart_explanations.csv", index=False, encoding="utf-8-sig")
    print(f"Wrote {REPORT / 'pac_paper_payload.json'}")


def read_csv(name: str) -> pd.DataFrame:
    path = SOURCE_TABLES / name
    if not path.exists():
        raise SystemExit(f"Missing PAC summary table: {path}")
    return pd.read_csv(path, encoding="utf-8-sig")


def build_summary_rows(subset: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        mdf = subset[subset["model"] == model]
        subset_scores = {
            row["subset"]: float(row["accuracy"]) * 100
            for _, row in mdf.iterrows()
            if pd.notna(row.get("accuracy"))
        }
        field_scores = [float(v) for v in mdf["mean_field_accuracy"].dropna()]
        decoy_scores = [float(v) for v in mdf["decoy_capture_rate"].dropna()]
        accuracies = [subset_scores.get(name) for name in SUBSET_ORDER if subset_scores.get(name) is not None]
        rows.append(
            {
                "model": MODEL_LABELS[model],
                "model_key": model,
                "arch": ARCH_LABELS[model],
                "pac_a": round_or_dash(subset_scores.get("PAC-A_position")),
                "pac_b": round_or_dash(subset_scores.get("PAC-B_interference")),
                "pac_c": round_or_dash(subset_scores.get("PAC-C_binding_capacity")),
                "pac_d": round_or_dash(subset_scores.get("PAC-D_multihop_false_chain")),
                "overall": round_or_dash(np.mean(accuracies) if accuracies else None),
                "field": round_or_dash(np.mean(field_scores) * 100 if field_scores else None),
                "decoy": round_or_dash(np.mean(decoy_scores) * 100 if decoy_scores else None),
            }
        )
    return rows


def build_table_payload(summary_rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [
        [
            item["model"],
            item["arch"],
            item["pac_a"],
            item["pac_b"],
            item["pac_c"],
            item["pac_d"],
            item["overall"],
            item["field"],
            item["decoy"],
        ]
        for item in summary_rows
    ]
    note = (
        "从 PAC-Test v2.1 自设数据集结果来看，高相似干扰、实体绑定和多跳假链显著拉开了模型差异。"
        "PAC-A/B 体现位置与干扰压力下的事实保持能力，PAC-C/D 分别考察多实体属性绑定和链式推理稳定性。"
        "整体上，Qwen3.5-27B 与 Qwen3.5-122B-A10B 在多数子实验中保持较高准确率，Qwen3-8B 与 Qwen3-14B-noT 在强干扰和绑定任务中下降明显。"
        "该结果说明长上下文能力不只是窗口长度问题，更关键的是在复杂上下文中维持目标事实绑定、过滤干扰项和追踪多跳链路。"
    )
    return {
        "sheet_name": "PAC自设数据集",
        "title": "PAC-Test v2.1 自设数据集评测结果汇总",
        "group_headers": ["模型信息", "", "PAC-Test v2.1 子集准确率", "", "", "", "综合指标", "", ""],
        "headers": [
            "模型",
            "架构",
            "PAC-A\n位置效应",
            "PAC-B\n干扰密度",
            "PAC-C\n绑定容量",
            "PAC-D v2.1\n多跳假链",
            "综合\n均分",
            "字段\n保持",
            "干扰\n捕获率↓",
        ],
        "rows": rows,
        "note": note,
    }


def build_figures(subset: pd.DataFrame, condition: pd.DataFrame) -> list[dict[str, str]]:
    specs = []
    specs.append(make_subset_heatmap(subset))
    specs.append(make_model_ranking(subset))
    specs.append(make_pac_a_lines(condition))
    specs.append(make_pac_b_lines(condition))
    specs.append(make_pac_c_heatmap(condition))
    specs.append(make_pac_d_heatmap(condition))
    specs.append(make_error_profile(subset))
    return specs


def build_chart_payload(chart_specs: list[dict[str, str]]) -> dict[str, Any]:
    conclusion = (
        "PAC-Test v2.1 的核心结果表明，公开基准中不明显的模型差异会在高相似干扰、实体绑定和多跳假链条件下被放大。"
        "Qwen3.5-27B 与 Qwen3.5-122B-A10B 在实体绑定和链式追踪任务上更稳定，说明较强模型的优势主要体现在抗干扰阈值和关系保持能力上。"
        "相对较小的模型在 PAC-A/B 中容易出现干扰捕获或部分绑定错误，说明长上下文失效并非单纯由长度导致，而是目标事实与相似干扰之间的绑定竞争导致。"
        "因此，PAC-Test 可以作为公开基准天花板效应之后的核心归因实验，用于解释复杂中文长上下文场景中的记忆衰减机制。"
    )
    return {
        "sheet_name": "PAC图表说明",
        "title": "PAC-Test v2.1 实验结果图示说明与现象分析",
        "headers": ["图表", "横轴含义", "纵轴含义", "测试指标与专业术语解释", "实验现象", "初步结论"],
        "rows": chart_specs,
        "conclusion_sheet": "总体结论",
        "conclusion_title": "PAC-Test v2.1 总体结论",
        "conclusion": conclusion,
    }


def make_subset_heatmap(subset: pd.DataFrame) -> dict[str, str]:
    pivot = pivot_subset(subset, "accuracy")
    path = FIGURES / "pac_subset_accuracy_heatmap.png"
    plot_heatmap(pivot, path, "PAC v2.1 subset accuracy", "PAC subset", "Model")
    return {
        "image": str(path),
        "x": "PAC 子实验：PAC-A、PAC-B、PAC-C、PAC-D v2.1，分别对应位置效应、干扰密度、实体绑定容量和多跳假链追踪。",
        "y": "模型名称，按实验中使用的模型顺序排列。",
        "metric": "颜色表示子实验准确率，数值越高代表模型在该类复杂长上下文条件下越稳定。",
        "phenomenon": "PAC-C 和 PAC-D 更能拉开强模型与弱模型差距；Qwen3.5-27B、Qwen3.5-122B-A10B 在多数子实验中保持较高水平，Qwen3-8B 在复杂绑定任务中明显较低。",
        "conclusion": "PAC-Test 相比普通检索任务具有更强区分度，能够揭示复杂上下文中的绑定保持与干扰过滤能力差异。",
    }


def make_model_ranking(subset: pd.DataFrame) -> dict[str, str]:
    values = pivot_subset(subset, "accuracy").mean(axis=1).sort_values()
    path = FIGURES / "pac_model_overall_ranking.png"
    fig, ax = plt.subplots(figsize=(7.6, 4.6), dpi=180)
    colors = ["#9ECAE1" if value < values.median() else "#3182BD" for value in values]
    ax.barh([MODEL_LABELS.get(i, i) for i in values.index], values.values, color=colors)
    ax.set_xlabel("Mean subset accuracy")
    ax.set_title("PAC v2.1 overall ranking")
    ax.set_xlim(0, max(1.0, values.max() * 1.15))
    for idx, value in enumerate(values.values):
        ax.text(value + 0.015, idx, f"{value:.2f}", va="center", fontsize=8)
    polish(ax)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return {
        "image": str(path),
        "x": "PAC-A/B/C/D 四个子实验准确率的平均值。",
        "y": "模型名称，按综合表现从低到高排序。",
        "metric": "综合均分用于概括模型在高干扰、绑定和多跳条件下的总体稳定性。",
        "phenomenon": "Qwen3.5-27B 和 Qwen3.5-122B-A10B 综合表现居前；Qwen3-8B 和 Qwen3-14B-noT 明显偏低，说明强干扰条件下模型差异被放大。",
        "conclusion": "模型规模或架构优势主要体现在复杂上下文中的关系保持能力，而不是简单检索任务上的领先。",
    }


def make_pac_a_lines(condition: pd.DataFrame) -> dict[str, str]:
    df = condition[condition["subset"] == "PAC-A_position"].copy()
    df["condition_value"] = pd.to_numeric(df["condition_value"], errors="coerce")
    path = FIGURES / "pac_a_position_effect.png"
    plot_lines(df, path, "PAC-A position effect", "Target position (%)")
    return {
        "image": str(path),
        "x": "目标事实在 32K 上下文中的相对位置：10%、25%、50%、75%、90%。",
        "y": "不同模型在对应位置条件下的准确率。",
        "metric": "PAC-A 在固定高相似干扰强度下改变目标事实位置，用于观察位置衰减和中间位置稳定性。",
        "phenomenon": "高相似干扰下，不同模型对位置变化的敏感性明显不同；部分模型在中后部位置出现下降或波动。",
        "conclusion": "位置效应在低干扰公开检索任务中不明显，但在高相似干扰下会被放大，支持将位置与干扰因素联合分析。",
    }


def make_pac_b_lines(condition: pd.DataFrame) -> dict[str, str]:
    df = condition[condition["subset"] == "PAC-B_interference"].copy()
    df["condition_value"] = pd.to_numeric(df["condition_value"], errors="coerce")
    path = FIGURES / "pac_b_interference_density.png"
    plot_lines(df, path, "PAC-B interference density", "High-similarity decoy count")
    return {
        "image": str(path),
        "x": "高相似干扰项数量，包括 0、16、32、64、128、192。",
        "y": "对应干扰强度下的模型准确率。",
        "metric": "PAC-B 固定目标位置，逐步增加干扰密度，用于估计模型的抗干扰阈值和下降斜率。",
        "phenomenon": "随着干扰密度升高，弱模型更容易出现干扰捕获或部分绑定错误；强模型下降更缓慢。",
        "conclusion": "高相似干扰是暴露长上下文记忆衰减和模型差异的关键压力源。",
    }


def make_pac_c_heatmap(condition: pd.DataFrame) -> dict[str, str]:
    df = condition[condition["subset"] == "PAC-C_binding_capacity"].copy()
    path = FIGURES / "pac_c_binding_capacity.png"
    pivot = pivot_condition(df)
    order = ["16/3", "16/5", "16/8", "32/3", "32/5", "32/8", "64/3", "64/5", "64/8"]
    plot_heatmap(pivot.reindex(index=MODEL_ORDER, columns=order), path, "PAC-C binding capacity", "K / Q", "Model")
    return {
        "image": str(path),
        "x": "K/Q 条件：K 表示上下文中的实体记录数，Q 表示一次查询的实体数量。",
        "y": "模型名称。",
        "metric": "准确率衡量模型能否同时保持多个实体-属性-值绑定关系。",
        "phenomenon": "Qwen3.5-27B 与 Qwen3.5-122B-A10B 在多实体绑定任务中表现稳定；小模型在 K 和 Q 增加时更容易出现绑定混淆。",
        "conclusion": "实体绑定容量是区分模型长上下文记忆稳定性的核心指标之一。",
    }


def make_pac_d_heatmap(condition: pd.DataFrame) -> dict[str, str]:
    df = condition[condition["subset"] == "PAC-D_multihop_false_chain"].copy()
    path = FIGURES / "pac_d_multihop_false_chain.png"
    pivot = pivot_condition(df)
    order = ["4/16", "4/32", "5/16", "5/32", "6/16", "6/32"]
    plot_heatmap(pivot.reindex(index=MODEL_ORDER, columns=order), path, "PAC-D v2.1 multihop false-chain", "Hops / false chains", "Model")
    return {
        "image": str(path),
        "x": "hop 数与假链数量组合，例如 4/16 表示 4 跳真实链和 16 条假链干扰。",
        "y": "模型名称。",
        "metric": "准确率衡量模型能否沿真实链完成多跳追踪并拒绝假链。",
        "phenomenon": "强模型在 PAC-D v2.1 中明显更稳定；部分模型虽然能抽取局部字段，但难以同时维持链路、验证信息和最终答案绑定。",
        "conclusion": "多跳假链任务显示，长上下文失效不仅是检索失败，更常表现为链路追踪和中间状态绑定断裂。",
    }


def make_error_profile(subset: pd.DataFrame) -> dict[str, str]:
    model_df = subset.groupby("model", as_index=False)[["decoy_capture_rate", "partial_rate"]].mean()
    model_df["correct_like"] = subset.groupby("model")["accuracy"].mean().reindex(model_df["model"]).values
    model_df = model_df.set_index("model").reindex(MODEL_ORDER)
    path = FIGURES / "pac_error_profile.png"
    fig, ax = plt.subplots(figsize=(8.4, 4.7), dpi=180)
    labels = [MODEL_LABELS.get(i, i) for i in model_df.index]
    x = np.arange(len(labels))
    width = 0.25
    ax.bar(x - width, model_df["correct_like"], width, label="Accuracy", color="#4C78A8")
    ax.bar(x, model_df["partial_rate"], width, label="Partial binding", color="#F58518")
    ax.bar(x + width, model_df["decoy_capture_rate"], width, label="Decoy capture", color="#E45756")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Rate")
    ax.set_title("PAC v2.1 error tendency")
    ax.legend(ncols=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    polish(ax)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return {
        "image": str(path),
        "x": "模型名称。",
        "y": "准确率、部分绑定错误率和干扰捕获率。",
        "metric": "partial binding 表示只保持部分字段或部分关系，decoy capture 表示模型被高相似干扰项吸引。",
        "phenomenon": "弱模型的干扰捕获和部分绑定错误更突出，说明错误主要来自相似实体与属性之间的绑定竞争。",
        "conclusion": "PAC-Test 能进一步定位错误来源，为解释长上下文记忆衰减提供比单一准确率更细的归因信号。",
    }


def plot_lines(df: pd.DataFrame, path: Path, title: str, x_label: str) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 4.8), dpi=180)
    for idx, model in enumerate(MODEL_ORDER):
        mdf = df[df["model"] == model].sort_values("condition_value")
        if mdf.empty:
            continue
        ax.plot(
            mdf["condition_value"],
            mdf["accuracy"],
            marker="o",
            linewidth=1.8,
            markersize=4,
            color=PALETTE[idx],
            label=MODEL_LABELS[model],
        )
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(ncols=2, fontsize=7, frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    polish(ax)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(pivot: pd.DataFrame, path: Path, title: str, x_label: str, y_label: str) -> None:
    pivot = pivot.copy()
    labels_y = [MODEL_LABELS.get(str(idx), str(idx)) for idx in pivot.index]
    labels_x = [str(col).replace("PAC-A_position", "A").replace("PAC-B_interference", "B").replace("PAC-C_binding_capacity", "C").replace("PAC-D_multihop_false_chain", "D") for col in pivot.columns]
    data = pivot.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=180)
    im = ax.imshow(data, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(labels_x)))
    ax.set_xticklabels(labels_x, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(labels_y)))
    ax.set_yticklabels(labels_y)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7, color="#111111" if value < 0.65 else "white")
    cbar = fig.colorbar(im, ax=ax, fraction=0.032, pad=0.02)
    cbar.ax.set_ylabel("Accuracy", rotation=270, labelpad=12)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def pivot_subset(subset: pd.DataFrame, value: str) -> pd.DataFrame:
    return subset.pivot(index="model", columns="subset", values=value).reindex(index=MODEL_ORDER, columns=SUBSET_ORDER)


def pivot_condition(df: pd.DataFrame) -> pd.DataFrame:
    return df.pivot(index="model", columns="condition_value", values="accuracy")


def polish(ax: plt.Axes) -> None:
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def round_or_dash(value: Any) -> float | str:
    if value is None or pd.isna(value):
        return "—"
    return round(float(value), 1)


if __name__ == "__main__":
    main()
