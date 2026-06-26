#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PAC-Test heatmaps -- gkamradt "Needle In A Haystack" visual style, applied to
the four PAC subsets (A/B/C/D).

Each subset becomes a green->red accuracy heatmap whose X axis is the context
length (in tokens) -- exactly like the reference NIAH plot -- and whose Y axis
is the *one* stress variable that subset sweeps:

    A  位置效应   Y = needle depth (0..100 %)            facet: none
    B  干扰稀释   Y = distractor / noise density          facet: dilution type (x3)
    C  信息覆盖   Y = entity-similarity level (4)         facet: distance (near/med/far)
    D  多跳衰减   Y = reasoning hops (2/3/4)              facet: distance

Subset A reproduces the reference figure 1:1 (downward depth arrow, Top/Bottom
of Document labels). The other three reuse the same colormap / cell style and
add small-multiple facets for their 3rd controlled variable.

Reads the latest results/<model>_*_details.jsonl (falls back to the
results/<model>_subset{X}_checkpoint.jsonl files for in-progress runs).

Usage:
    python plot_pac.py                                   # all models found, EM
    python plot_pac.py --models qwen2.5:7b --metric em
    python plot_pac.py --models qwen2.5:7b deepseek-r1:7b --metric contains
    python plot_pac.py --subsets A B                     # only some subsets
"""

import argparse
import glob
import json
import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

BASE_DIR        = Path(__file__).parent
EVAL_RESULTS_DIR = BASE_DIR / "results"
OUTPUT_DIR      = BASE_DIR / "pac_plots"

# Exact gkamradt colormap: red -> amber -> green
PAC_CMAP = LinearSegmentedColormap.from_list("niah", ["#F0496E", "#EBB839", "#0CD79F"])

CHARS_PER_TOKEN = 1.71   # 4096 chars = 2390 tokens on Qwen2.5-7b


def chars_to_tokens(n_chars: int) -> int:
    return int(round(n_chars / CHARS_PER_TOKEN))


def token_label(n_tokens: int) -> str:
    if n_tokens >= 1000:
        k = n_tokens / 1000
        return f"{k:.0f}K" if k == int(k) else f"{k:.1f}K"
    return str(n_tokens)


# =========================================================
# Subset configuration
# =========================================================
# Each config tells the loader/plotter:
#   y_fields  : candidate detail keys holding the Y (stress) variable
#   y_label   : big vertical axis label
#   y_kind    : "depth" -> 0..1 ratio shown as %, with Top/Bottom arrow
#               "ratio" -> 0..1 ratio shown as %
#               "cat"   -> categorical, use y_order + y_names
#               "int"   -> integer (hops)
#   facet     : detail key to split into side-by-side heatmaps (or None)
#   facet_order / facet_names : ordering + display names for facets
#   y_order / y_names         : ordering + display names for categorical Y

SUBSET_CFG = {
    "A": dict(
        cn="位置效应", en="Position Effect",
        y_fields=["position_ratio"], y_label="Placed Fact Document Depth",
        y_kind="depth", facet=None,
        goal=("Goal: 验证 Lost-in-the-Middle / 位置编码衰减 —— 关键事实在文档不同深度处的可检索性\n"
              "Needle: 单条领域事实（实体+属性），位置可变   |   Question: {实体}的{属性}是什么？   |   其余文本保持总长恒定"),
    ),
    "B": dict(
        cn="干扰稀释", en="Dilution Effect",
        y_fields=["noise_density", "dilution_ratio"], y_label="Distractor Density",
        y_kind="ratio",
        facet="dilution_type",
        facet_order=["in_domain_related", "out_domain_unrelated", "random_noise"],
        facet_names={"in_domain_related": "领域内相关", "out_domain_unrelated": "领域外无关",
                     "random_noise": "随机噪声"},
        goal=("Goal: 验证注意力稀释假说 —— 关键事实固定于开头，干扰内容按密度递增\n"
              "Y: 干扰密度   |   分面: 干扰类型   |   Question: {实体}的{属性}是什么？"),
    ),
    "C": dict(
        cn="信息覆盖", en="Overwriting Effect",
        y_fields=["similarity_level"], y_label="Entity Similarity (interference ↑)",
        y_kind="cat",
        y_order=["completely_different", "diff_name_same_domain",
                 "same_name_diff_domain", "same_name_same_domain"],
        y_names={"completely_different": "完全不同(对照)", "diff_name_same_domain": "异名·同域",
                 "same_name_diff_domain": "同名·异域", "same_name_same_domain": "同名·同域"},
        facet="distance_level",
        facet_order=["near", "medium", "far"],
        facet_names={"near": "近", "medium": "中", "far": "远"},
        goal=("Goal: 验证相似实体导致的记忆覆盖 —— 目标实体 A 与易混淆实体 B 共现\n"
              "Y: A/B 相似度（越往下干扰越强）   |   分面: A-B 距离   |   Question: 区分两实体并取目标属性"),
    ),
    "D": dict(
        cn="多跳衰减", en="Multi-hop Decay",
        y_fields=["num_hops"], y_label="Reasoning Hops",
        y_kind="int",
        facet="distance_level",
        facet_order=["adjacent", "near", "medium", "far"],
        facet_names={"adjacent": "邻接", "near": "近", "medium": "中", "far": "远"},
        goal=("Goal: 测试链式信息在长上下文中的传递衰减 —— 需整合 2-4 个分散事实\n"
              "Y: 推理跳数   |   分面: 跳间距   |   Question: 沿事实链得出最终答案"),
    ),
}


# =========================================================
# Data loading
# =========================================================

def _detail_files(safe_model: str):
    """Return the rows of the newest *_details.jsonl, else merge checkpoints."""
    files = sorted(glob.glob(str(EVAL_RESULTS_DIR / f"{safe_model}_*_details.jsonl")))
    rows = []
    if files:
        with open(files[-1], encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows, Path(files[-1]).name

    # fallback: in-progress checkpoints
    ckpts = sorted(glob.glob(str(EVAL_RESULTS_DIR / f"{safe_model}_subset*_checkpoint.jsonl")))
    for c in ckpts:
        with open(c, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    src = "checkpoints" if ckpts else None
    return rows, src


def _y_value(row, cfg):
    for k in cfg["y_fields"]:
        if k in row and row[k] is not None:
            return row[k]
    return None


def build_pivots(rows, subset, cfg, metric):
    """Return {facet_value: (pivot_df, y_display_labels)} for one subset.

    pivot_df: index = ordered Y, columns = sorted context-length(tokens), value = mean(metric).
    facet_value is None when the subset has no facet.
    """
    rows = [r for r in rows if r.get("subset") == subset]
    if not rows:
        return {}

    # tag each row with tokens, y, facet
    recs = []
    for r in rows:
        clen = r.get("total_length") or r.get("context_length_chars") or r.get("context_length")
        if not clen:
            continue
        yv = _y_value(r, cfg)
        if yv is None:
            continue
        # v3 results carry total_length_unit="tokens" -> use as-is;
        # v1/v2 results have no unit (chars) -> convert chars->tokens.
        unit = r.get("total_length_unit", "chars")
        tok = int(clen) if unit == "tokens" else chars_to_tokens(int(clen))
        fac = r.get(cfg["facet"]) if cfg["facet"] else None
        mv = r.get(metric)
        if mv is None:          # row predates this metric (e.g. v1/v2 has no score_norm)
            continue
        recs.append({
            "tok": tok,
            "y": yv,
            "facet": fac,
            "score": float(mv),
        })
    if not recs:
        return {}
    df = pd.DataFrame(recs)

    # ---- determine Y ordering + display labels ----
    def y_order_and_labels(yvals):
        kind = cfg["y_kind"]
        if kind in ("depth", "ratio"):
            order = sorted(set(yvals), key=float)
            labels = [f"{float(v) * 100:.0f}%" for v in order]
        elif kind == "int":
            order = sorted(set(yvals), key=lambda v: int(v))
            labels = [f"{int(v)}-hop" for v in order]
        else:  # categorical
            present = set(yvals)
            order = [v for v in cfg["y_order"] if v in present]
            order += [v for v in present if v not in order]
            labels = [cfg.get("y_names", {}).get(v, str(v)) for v in order]
        return order, labels

    # ---- facet ordering ----
    if cfg["facet"]:
        present = set(df["facet"].dropna())
        facets = [v for v in cfg.get("facet_order", []) if v in present]
        facets += [v for v in present if v not in facets]
    else:
        facets = [None]

    out = {}
    for fac in facets:
        sub = df if fac is None else df[df["facet"] == fac]
        if sub.empty:
            continue
        order, labels = y_order_and_labels(sub["y"].tolist())
        cols = sorted(sub["tok"].unique())
        grid = np.full((len(order), len(cols)), np.nan)
        g = sub.groupby(["y", "tok"])["score"].mean()
        for i, yv in enumerate(order):
            for j, c in enumerate(cols):
                if (yv, c) in g.index:
                    grid[i, j] = g.loc[(yv, c)]
        pivot = pd.DataFrame(grid, index=labels, columns=cols)
        out[fac] = pivot
    return out


# =========================================================
# Heatmap drawing (gkamradt style)
# =========================================================

def _draw_cells(ax, pivot):
    masked = np.ma.masked_invalid(pivot.values)
    im = ax.imshow(masked, cmap=PAC_CMAP, vmin=0, vmax=1, aspect="auto")
    ax.set_facecolor("#EEEEEE")
    for i in range(pivot.shape[0] + 1):
        ax.axhline(i - 0.5, color="gray", linewidth=0.5, zorder=2)
    for j in range(pivot.shape[1] + 1):
        ax.axvline(j - 0.5, color="gray", linewidth=0.5, zorder=2)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v:.0%}", ha="center", va="center",
                    fontsize=9, fontweight="bold",
                    color="white" if v < 0.38 else "black")
    tok_l = [token_label(c) for c in pivot.columns]
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(tok_l, rotation=45, ha="right", fontsize=10)
    ax.xaxis.set_tick_params(length=0)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(list(pivot.index), fontsize=10)
    ax.yaxis.set_tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    return im


def plot_subset(rows, subset, cfg, model, metric, today, out_path):
    pivots = build_pivots(rows, subset, cfg, metric)
    if not pivots:
        print(f"  [{subset}] no data, skipped")
        return False

    facets = list(pivots.keys())
    n = len(facets)
    metric_name = {
        "em": "Exact Match",
        "contains": "Contains Match",
        "score_norm": "Judge Score (1–10 → norm)",
    }.get(metric, metric)

    fig = plt.figure(figsize=(7.0 * n + 3.2, 9))

    # heatmap band geometry (leave room: left big-label, right colorbar, bottom goal)
    left0, band_w = 0.155, 0.70
    bottom, height = 0.225, 0.60
    gap = 0.035
    each_w = (band_w - gap * (n - 1)) / n

    last_im = None
    axes = []
    for idx, fac in enumerate(facets):
        l = left0 + idx * (each_w + gap)
        ax = fig.add_axes([l, bottom, each_w, height])
        last_im = _draw_cells(ax, pivots[fac])
        ax.set_xlabel("Token Limit", fontsize=10.5, labelpad=6)
        if idx > 0:
            ax.set_yticklabels([])
        if cfg["facet"]:
            fname = cfg.get("facet_names", {}).get(fac, str(fac))
            ax.set_title(f"{cfg['facet']} = {fname}", fontsize=11, fontweight="bold", pad=8)
        axes.append(ax)

    # ---- titles ----
    fig.text(0.50, 0.975,
             f'Pressure Testing {model} via PAC-{subset} "{cfg["cn"]}"',
             ha="center", va="top", fontsize=14, fontweight="bold")
    fig.text(0.50, 0.938,
             f'{cfg["en"]}  |  Accuracy = {metric_name}  across Context Length × '
             f'{cfg["y_label"]}',
             ha="center", va="top", fontsize=10.5, color="#555555")

    # ---- left big vertical Y label (over the first heatmap band) ----
    p0 = axes[0].get_position()
    mid = (p0.y0 + p0.y1) / 2
    fig.text(p0.x0 - 0.085, mid, cfg["y_label"], ha="center", va="center",
             fontsize=11.5, fontweight="bold", rotation=-90, color="#333333")

    # depth subsets get the Top/Bottom-of-document arrow like the reference
    if cfg["y_kind"] == "depth":
        fig.text(p0.x0 - 0.012, p0.y1 + 0.006, "Top Of\nDocument", ha="right", va="top",
                 fontsize=8.5, color="#666666", style="italic", multialignment="right")
        fig.text(p0.x0 - 0.012, p0.y0 - 0.006, "Bottom Of\nDocument", ha="right", va="bottom",
                 fontsize=8.5, color="#666666", style="italic", multialignment="right")
        ax_x = p0.x0 - 0.085
        fig.add_artist(FancyArrowPatch(
            posA=(ax_x, p0.y0 + 0.05), posB=(ax_x, p0.y0 + 0.008),
            transform=fig.transFigure, arrowstyle="->,head_width=4,head_length=6",
            color="#555555", linewidth=1.5))

    # ---- right colorbar + accuracy labels ----
    pL = axes[-1].get_position()
    cbar_ax = fig.add_axes([pL.x1 + 0.018, bottom, 0.013, height])
    cbar = fig.colorbar(last_im, cax=cbar_ax)
    cbar.set_ticks([])
    cbar.outline.set_linewidth(0.5)
    acc_x = pL.x1 + 0.045
    fig.text(acc_x, pL.y1, "100%\nAccuracy", ha="left", va="top",
             fontsize=8.5, fontweight="bold", color="#0CD79F")
    fig.text(acc_x, (pL.y0 + pL.y1) / 2, " 50%\nAccuracy", ha="left", va="center",
             fontsize=8.5, fontweight="bold", color="#EBB839")
    fig.text(acc_x, pL.y0, "  0%\nAccuracy", ha="left", va="bottom",
             fontsize=8.5, fontweight="bold", color="#F0496E")

    # ---- bottom goal box ----
    gbox = fig.add_axes([0.06, 0.012, 0.88, 0.082])
    gbox.set_xlim(0, 1); gbox.set_ylim(0, 1); gbox.set_axis_off()
    gbox.add_patch(FancyBboxPatch((0.0, 0.0), 1.0, 1.0, boxstyle="round,pad=0.04",
                                  linewidth=1.5, edgecolor="#BBBBBB", facecolor="#F8F8F8",
                                  transform=gbox.transAxes, clip_on=False))
    gbox.text(0.5, 0.5, cfg["goal"], ha="center", va="center", fontsize=9.5,
              transform=gbox.transAxes, color="#333333")

    fig.text(0.985, 0.985, f"Model: {model}    Date: {today}",
             ha="right", va="top", fontsize=8, color="#999999")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [{subset}] saved: {out_path.name}")
    return True


# =========================================================
# Main
# =========================================================

def discover_models():
    models = set()
    for f in glob.glob(str(EVAL_RESULTS_DIR / "*_details.jsonl")):
        m = re.match(r"(.+?)_\d{8}_\d{6}_details\.jsonl$", Path(f).name)
        if m:
            models.add(m.group(1))
    for f in glob.glob(str(EVAL_RESULTS_DIR / "*_subset*_checkpoint.jsonl")):
        m = re.match(r"(.+?)_subset[A-D]_checkpoint\.jsonl$", Path(f).name)
        if m:
            models.add(m.group(1))
    return sorted(models)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=None,
                    help="Ollama model ids, e.g. qwen2.5:7b (default: auto-discover in results/)")
    ap.add_argument("--metric", default="both",
                    choices=["em", "contains", "score_norm", "both", "all"],
                    help="em / contains / score_norm / both(em+contains) / all(三者全出)（文件名各带后缀）")
    ap.add_argument("--subsets", nargs="+", default=["A", "B", "C", "D"])
    args = ap.parse_args()
    metrics = {
        "both": ["em", "contains"],
        "all": ["em", "contains", "score_norm"],
    }.get(args.metric, [args.metric])

    today = date.today().isoformat()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.models:
        # accept either raw ids or already-safe names
        safe_models = [(m, re.sub(r"[:/\\]", "_", m)) for m in args.models]
    else:
        safe_models = [(s.replace("_", ":", 1), s) for s in discover_models()]
        if not safe_models:
            print("No results found in results/. Run evaluate.py first.")
            return

    for model, safe in safe_models:
        rows, src = _detail_files(safe)
        if not rows:
            print(f"\n--- {model}: no results, skipped ---")
            continue
        print(f"\n--- {model}  (source: {src}, {len(rows)} rows) ---")
        for subset in args.subsets:
            if subset not in SUBSET_CFG:
                continue
            for metric in metrics:
                out = OUTPUT_DIR / f"{safe}_PAC-{subset}_{metric}.png"
                plot_subset(rows, subset, SUBSET_CFG[subset], model, metric, today, out)


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
