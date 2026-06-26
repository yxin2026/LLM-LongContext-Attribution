#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模型横向对比脚本
读取 results/ 目录下的多个 _summary.json，生成对比表格和图表。

用法：
    python compare_models.py
    python compare_models.py --results results/qwen*.json results/deepseek*.json
"""

import argparse
import json
import sys
from pathlib import Path

def load_summary(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def extract_model_name(path: Path) -> str:
    # 文件名格式: {model}_{timestamp}_summary.json
    stem = path.stem  # e.g. qwen2.5_7b_20240101_120000_summary
    parts = stem.rsplit("_", 2)  # 去掉最后两段时间戳和 "summary"
    return parts[0].replace("_", ":") if len(parts) >= 2 else stem

def main():
    parser = argparse.ArgumentParser(description="PAC-Test 多模型对比")
    parser.add_argument("--results-dir", default="results",
                        help="结果目录（默认 results/）")
    parser.add_argument("--results", nargs="*",
                        help="指定 summary.json 文件列表")
    args = parser.parse_args()

    results_dir = Path(__file__).parent / args.results_dir

    if args.results:
        files = [Path(p) for p in args.results if p.endswith("_summary.json")]
    else:
        files = sorted(results_dir.glob("*_summary.json"))

    if not files:
        print(f"在 {results_dir} 中未找到 *_summary.json，请先运行 evaluate.py")
        sys.exit(1)

    subsets = ["A", "B", "C", "D"]
    subset_labels = {"A": "位置效应", "B": "干扰稀释",
                     "C": "信息覆盖", "D": "多跳衰减"}

    data = {}  # model_name -> {subset -> em}
    for f in files:
        try:
            summary = load_summary(f)
            model   = extract_model_name(f)
            data[model] = {s: summary[s]["em"] for s in subsets if s in summary}
        except Exception as e:
            print(f"跳过 {f.name}: {e}")

    if not data:
        print("没有有效数据")
        sys.exit(1)

    # --- 控制台表格 ---
    header_cols = ["模型"] + [f"Subset {s}" for s in subsets] + ["平均"]
    col_w = 16
    sep = "+" + "+".join(["-" * col_w] * len(header_cols)) + "+"
    def row(cells):
        return "|" + "|".join(str(c).center(col_w) for c in cells) + "|"

    print("\n" + "=" * (col_w * len(header_cols) + len(header_cols) + 1))
    print("  PAC-Test 多模型对比（Exact Match）")
    print("=" * (col_w * len(header_cols) + len(header_cols) + 1))
    print(sep)
    print(row(header_cols))
    print(sep)
    for model, scores in data.items():
        vals = [f"{scores[s]:.1%}" if s in scores else "—" for s in subsets]
        avgs = [scores[s] for s in subsets if s in scores]
        avg  = f"{sum(avgs)/len(avgs):.1%}" if avgs else "—"
        print(row([model[:col_w-1]] + vals + [avg]))
    print(sep)

    # --- 可视化 ---
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        matplotlib.rcParams['axes.unicode_minus'] = False
        import numpy as np

        models = list(data.keys())
        n_models = len(models)
        x = np.arange(len(subsets))
        width = 0.8 / n_models

        fig, ax = plt.subplots(figsize=(12, 6))
        for i, model in enumerate(models):
            vals = [data[model].get(s, 0) for s in subsets]
            offset = (i - n_models / 2 + 0.5) * width
            bars = ax.bar(x + offset, vals, width, label=model)

        ax.set_xticks(x)
        ax.set_xticklabels([f"Subset {s}\n{subset_labels[s]}" for s in subsets])
        ax.set_ylabel("Exact Match")
        ax.set_ylim(0, 1.1)
        ax.set_title("PAC-Test 多模型对比")
        ax.legend(loc="upper right")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

        out_path = results_dir / "comparison_chart.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        print(f"\n对比图表已保存: {out_path}")
        plt.close()
    except ImportError:
        print("matplotlib 未安装，跳过可视化")

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
