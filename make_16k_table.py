#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PAC-Test「固定 16k」横向对比表生成器。

对照 实验设计/实验设计.xlsx 的 PAC-Test 表：行=模型，列=各子集在 **固定 16k token**
下的代表性切片，值=准确率。本脚本从 results/ 顶层的 v3 结果（*_details.jsonl）
自动抽出 total_length==16000 的样本，按下面 SLICES 定义的 12 个列切片算 EM/Contains，
输出对齐的控制台表 + 一份 Markdown（可直接贴回 xlsx/文档）。

只看 v3：results/*.jsonl 的 glob 不递归，已归档的 v1 结果在 results/old_v1_dataset/
里不会被扫到；且 v1 的 16k 档是 16384≠16000，即便混入也不匹配。

用法：
    python make_16k_table.py                       # 自动发现所有模型，EM
    python make_16k_table.py --metric contains
    python make_16k_table.py --models qwen2.5:7b deepseek-r1:7b
"""

import argparse
import glob
import json
import re
import sys
from pathlib import Path
from datetime import date

EVAL_DIR  = Path(__file__).parent / "results"
FIXED_LEN = 16000        # tokens

# 「固定 16k」对比表的列切片：(组, 列名, subset, 选取函数)
# C 子集在 sim×dist 两维上只取 3 个代表切片（与设计表一致）：
#   Sim-High = 同名同域（干扰最强），Sim-Med = 同名异域（中等），
#   Dist-Far = 距离最远（跨所有相似度平均）。如需改口径，改这里即可。
SLICES = [
    ("PAC-A 位置效应", "Pos-10%",  "A", lambda r: r.get("position_ratio") == 0.1),
    ("PAC-A 位置效应", "Pos-50%",  "A", lambda r: r.get("position_ratio") == 0.5),
    ("PAC-A 位置效应", "Pos-90%",  "A", lambda r: r.get("position_ratio") == 0.9),
    ("PAC-B 干扰稀释", "Noise-0%",  "B", lambda r: r.get("noise_density") == 0.0),
    ("PAC-B 干扰稀释", "Noise-50%", "B", lambda r: r.get("noise_density") == 0.5),
    ("PAC-B 干扰稀释", "Noise-90%", "B", lambda r: r.get("noise_density") == 0.9),
    ("PAC-C 信息覆盖", "Sim-High",  "C", lambda r: r.get("similarity_level") == "same_name_same_domain"),
    ("PAC-C 信息覆盖", "Sim-Med",   "C", lambda r: r.get("similarity_level") == "same_name_diff_domain"),
    ("PAC-C 信息覆盖", "Dist-Far",  "C", lambda r: r.get("distance_level") == "far"),
    ("PAC-D 多跳衰减", "2-hop",     "D", lambda r: r.get("num_hops") == 2),
    ("PAC-D 多跳衰减", "3-hop",     "D", lambda r: r.get("num_hops") == 3),
    ("PAC-D 多跳衰减", "4-hop",     "D", lambda r: r.get("num_hops") == 4),
]


def discover_models():
    models = []
    for f in glob.glob(str(EVAL_DIR / "*_details.jsonl")):
        m = re.match(r"(.+?)_\d{8}_\d{6}_details\.jsonl$", Path(f).name)
        if m:
            models.append(m.group(1))
    return sorted(set(models))


def load_latest_details(safe_model):
    files = sorted(glob.glob(str(EVAL_DIR / f"{safe_model}_*_details.jsonl")))
    if not files:
        return [], None
    rows = []
    with open(files[-1], encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows, Path(files[-1]).name


def slice_acc(rows16k, subset, filt, metric):
    """返回 (acc_percent_or_None, n)。"""
    sel = [r for r in rows16k if r.get("subset") == subset and filt(r)]
    if not sel:
        return None, 0
    hit = sum(1 for r in sel if r.get(metric))
    return 100.0 * hit / len(sel), len(sel)


COL_DEF_NOTE = ("列定义：A=position_ratio(0.1/0.5/0.9)；B=noise_density(0/0.5/0.9)；"
                "C=Sim-High(同名同域)/Sim-Med(同名异域)/Dist-Far(距离远，跨相似度)；"
                "D=num_hops(2/3/4)。")


def write_md(table, metric, safe_models, sources, col_names):
    metric_name = "Exact Match" if metric == "em" else "Contains Match"
    out = EVAL_DIR / f"table_16k_{metric}.md"
    lines = [f"# PAC-Test 固定 16k 横向对比表（{metric_name}）\n",
             f"> 自动生成于 {date.today().isoformat()}　|　切片取自 total_length==16000 token 的样本"
             "　|　`—` = 该切片无数据/未评测\n",
             "| 模型 | " + " | ".join(col_names) + " |",
             "|" + "---|" * (len(col_names) + 1)]
    for model, safe in safe_models:
        if not sources[model]:
            continue
        cells = ["—" if table[model][c][0] is None else f"{table[model][c][0]:.0f}%"
                 for c in col_names]
        lines.append(f"| {model} | " + " | ".join(cells) + " |")
    lines += ["", COL_DEF_NOTE]
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=None,
                    help="Ollama 模型 id（默认自动发现 results/ 里的全部）")
    ap.add_argument("--metric", default="both", choices=["em", "contains", "both"],
                    help="em / contains / both（默认两者都出，各写一份 md）")
    args = ap.parse_args()
    metrics = ["em", "contains"] if args.metric == "both" else [args.metric]

    if args.models:
        safe_models = [(m, re.sub(r"[:/\\]", "_", m)) for m in args.models]
    else:
        safe_models = [(s.replace("_", ":", 1), s) for s in discover_models()]
    if not safe_models:
        print("results/ 里没有任何 *_details.jsonl，先跑 evaluate.py。")
        return

    col_names = [c for _, c, _, _ in SLICES]
    groups = []
    for g, _, _, _ in SLICES:
        if g not in groups:
            groups.append(g)

    # 加载一次（与指标无关）：model -> (rows16k, src, 已评测subset集合)
    loaded = {}
    for model, safe in safe_models:
        rows, src = load_latest_details(safe)
        loaded[model] = ([r for r in rows if r.get("total_length") == FIXED_LEN],
                         src, {r.get("subset") for r in rows})
    sources = {model: loaded[model][1] for model, _ in safe_models}

    # 每个指标各算一张 table[model][col] = (acc, n)
    tables = {}
    for metric in metrics:
        tables[metric] = {}
        for model, safe in safe_models:
            rows16k = loaded[model][0]
            tables[metric][model] = {col: slice_acc(rows16k, subset, filt, metric)
                                     for (_, col, subset, filt) in SLICES}

    # ---------- 控制台：每格显示各指标值 ----------
    mlabels = "/".join("EM" if m == "em" else "Contains" for m in metrics)
    print("=" * 78)
    print(f"PAC-Test 固定 16k 横向对比  |  每格 = {mlabels}")
    print("=" * 78)
    for model, safe in safe_models:
        rows16k, src, covered = loaded[model]
        print(f"\n■ {model}   (来源: {src or '无结果'})")
        if not src:
            continue
        for g in groups:
            parts = []
            for (_, col, subset, filt) in SLICES:
                if _g_of(col) != g:
                    continue
                vals = [tables[m][model][col] for m in metrics]
                n = vals[0][1]
                accs = "/".join("—" if a is None else f"{a:.0f}%" for a, _ in vals)
                parts.append(f"{col}={accs}" + ("" if n == 0 else f"(n={n})"))
            print(f"   {g}:  " + "   ".join(parts))
        missing = [s for s in "ABCD" if s not in covered]
        if missing:
            print(f"   未评测子集: {', '.join(missing)}")

    # ---------- 每个指标各写一份 Markdown ----------
    print()
    for metric in metrics:
        out = write_md(tables[metric], metric, safe_models, sources, col_names)
        print(f"Markdown 已写出: {out}")


# 小工具：列名 -> 组名
_COL2GROUP = {c: g for (g, c, _, _) in SLICES}
def _g_of(col):
    return _COL2GROUP[col]


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
