from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "reports" / "paper_ready_tables"

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


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "public": build_public_table(),
        "pac": build_pac_table(),
    }
    path = OUTPUT / "paper_ready_tables_payload.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {path}")


def build_public_table() -> dict[str, Any]:
    base = ROOT / "results" / "reports" / "public_benchmarks_summary_no_hunyuan" / "tables"
    longbench = read_csv(base / "longbench_by_model.csv")
    niah = read_csv(base / "niah_by_subtask_model.csv")
    ruler = read_csv(base / "ruler_by_task_model.csv")

    lb_map = dict(zip(longbench["model"], longbench["score_mean"]))
    niah_map = pivot_score(niah, "subtask")
    ruler_map = pivot_score(ruler, "task")

    headers = [
        "模型",
        "架构",
        "LongBench\n均分",
        "NIAH\nSingle",
        "NIAH\nMulti",
        "NIAH\nSequential",
        "RULER\nNIAH",
        "RULER\nHotpotQA",
        "RULER\nSQuAD",
        "RULER\nVarTrack",
    ]
    group_headers = [
        "模型信息",
        "",
        "LongBench",
        "NIAH",
        "",
        "",
        "RULER",
        "",
        "",
        "",
    ]
    rows = []
    for model in MODEL_ORDER:
        rows.append(
            [
                MODEL_LABELS[model],
                ARCH_LABELS[model],
                pct(lb_map.get(model)),
                pct(niah_map.get((model, "single"))),
                pct(niah_map.get((model, "multi"))),
                pct(niah_map.get((model, "sequential"))),
                pct(ruler_map.get((model, "niah"))),
                pct(ruler_map.get((model, "qa_hotpotqa"))),
                pct(ruler_map.get((model, "qa_squad"))),
                pct(ruler_map.get((model, "variable_tracking"))),
            ]
        )
    note = (
        "该表汇总去除 Hunyuan-A13B 后的公开基准结果，数值为有效样本准确率百分制。LongBench 反映真实长文本理解，"
        "NIAH/RULER 反映检索与有效上下文能力；公开任务整体偏高，主要作为基础能力和天花板效应证据。"
    )
    return {
        "title": "公开数据集评测结果汇总",
        "group_headers": group_headers,
        "headers": headers,
        "rows": rows,
        "note": note,
    }


def build_pac_table() -> dict[str, Any]:
    base = ROOT / "results" / "reports" / "pac_v21_all_summary_no_hunyuan" / "tables"
    subset = read_csv(base / "summary_by_subset_model.csv")
    overall = read_csv(base / "summary_by_model_overall.csv")
    subset_map = pivot_score(subset, "subset", value_col="accuracy_success")
    overall_map = dict(zip(overall["model"], overall["accuracy_all_conservative"]))
    coverage_map = dict(zip(overall["model"], overall["coverage"]))

    headers = [
        "模型",
        "架构",
        "PAC-A\n位置效应",
        "PAC-B\n干扰密度",
        "PAC-C\n绑定容量",
        "PAC-D v2.1\n多跳假链",
        "综合\n保守分",
        "有效\n覆盖率",
    ]
    group_headers = [
        "模型信息",
        "",
        "PAC-Test v2.1 子集准确率",
        "",
        "",
        "",
        "综合指标",
        "",
    ]
    rows = []
    for model in MODEL_ORDER:
        rows.append(
            [
                MODEL_LABELS[model],
                ARCH_LABELS[model],
                pct(subset_map.get((model, "PAC-A_position"))),
                pct(subset_map.get((model, "PAC-B_interference"))),
                pct(subset_map.get((model, "PAC-C_binding_capacity"))),
                pct(subset_map.get((model, "PAC-D_multihop_false_chain"))),
                pct(overall_map.get(model)),
                pct(coverage_map.get(model)),
            ]
        )
    note = (
        "该表汇总 PAC-Test v2.1 自设数据集结果，数值为百分制。PAC-A/B/C/D 分别考察位置效应、干扰密度、实体绑定容量和多跳假链追踪。"
        "相比公开基准，该表更适合支撑高相似干扰下的记忆衰减与绑定失效分析。"
    )
    return {
        "title": "PAC-Test v2.1 自设数据集结果汇总",
        "group_headers": group_headers,
        "headers": headers,
        "rows": rows,
        "note": note,
    }


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def pivot_score(df: pd.DataFrame, key_col: str, value_col: str = "score_mean") -> dict[tuple[str, str], float]:
    return {
        (str(row["model"]), str(row[key_col])): float(row[value_col])
        for _, row in df.iterrows()
        if pd.notna(row.get(value_col))
    }


def pct(value: Any) -> float | str:
    if value is None or pd.isna(value):
        return "—"
    return round(float(value) * 100, 1)


if __name__ == "__main__":
    main()
