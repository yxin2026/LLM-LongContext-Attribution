#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NIAH heatmap -- exact visual style of gkamradt/LLMTest_NeedleInAHaystack.

Layout per plot:
  Left   : "Top Of Document" / big vertical "Placed Fact Document Depth v"
           / "Bottom Of Document"
  Right  : colorbar + "100% / 50% / 0% Accuracy Of Retrieval" color labels
  Bottom : rounded-rectangle goal description box
  Title  : 'Pressure Testing {Model} {N}K via "Needle In A HayStack"'
  Sub    : 'Asking {Model} To Do Fact Retrieval Across Context Lengths & Document Depth'

Usage:
    python plot_niah.py
    python plot_niah.py --models qwen2.5:7b deepseek-r1:7b
    python plot_niah.py --metric em
"""

import json
import re
import argparse
import glob
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

BASE_DIR         = Path(__file__).parent
NIAH_RESULTS_DIR = BASE_DIR / "niah_results"
EVAL_RESULTS_DIR = BASE_DIR / "results"
OUTPUT_DIR       = BASE_DIR / "niah_plots"

# Exact gkamradt colormap
NIAH_CMAP = LinearSegmentedColormap.from_list(
    "niah", ["#F0496E", "#EBB839", "#0CD79F"]
)

CHARS_PER_TOKEN = 1.71   # 4096 chars = 2390 tokens on Qwen2.5-7b


def chars_to_tokens(n_chars: int) -> int:
    return int(round(n_chars / CHARS_PER_TOKEN))


def token_label(n_tokens: int) -> str:
    if n_tokens >= 1000:
        k = n_tokens / 1000
        return f"{k:.0f}K" if k == int(k) else f"{k:.1f}K"
    return str(n_tokens)


SYNTHETIC_NEEDLE   = "量子隼-7型实验飞船于2089年完成了首次超光速测试，飞行距离达到47.3光年。"
SYNTHETIC_QUESTION = "量子隼-7型实验飞船的首次超光速测试飞行距离是多少？"
SYNTHETIC_ANSWER   = "47.3光年"


# =========================================================
# Data loaders
# =========================================================

def load_synthetic_niah(model_name: str) -> pd.DataFrame:
    """Load results from niah_test.py.
    Prefers new token-based naming (_tok{N}_dep{M}); falls back to old
    char-based naming (_len{N}_depth{M}) converting chars -> tokens.
    """
    safe = re.sub(r"[:/\\]", "_", model_name)
    rows = []

    # New format (token-based lengths)
    for fpath in glob.glob(str(NIAH_RESULTS_DIR / f"{safe}_tok*_dep*_results.json")):
        with open(fpath, encoding="utf-8") as f:
            d = json.load(f)
        rows.append({
            "Document Depth": float(d["depth_percent"]),
            "Context Length":  int(d["context_length_tokens"]),
            "Score":           d["score"] / 10.0,
        })

    # Old format fallback (char-based lengths)
    if not rows:
        for fpath in glob.glob(str(NIAH_RESULTS_DIR / f"{safe}_len*_depth*_results.json")):
            with open(fpath, encoding="utf-8") as f:
                d = json.load(f)
            rows.append({
                "Document Depth": float(d["depth_percent"]),
                "Context Length":  chars_to_tokens(int(d["context_length"])),
                "Score":           d["score"] / 10.0,
            })

    return pd.DataFrame(rows)


def load_knowledge_niah(model_name: str) -> pd.DataFrame:
    """Load results from niah_knowledge_test.py (new token-based naming _tok{N}_dep{M})."""
    safe    = re.sub(r"[:/\\]", "_", model_name)
    pattern = str(BASE_DIR / "niah_knowledge_results" / f"{safe}_*_tok*_dep*.json")
    rows = []
    for fpath in glob.glob(pattern):
        with open(fpath, encoding="utf-8") as f:
            d = json.load(f)
        rows.append({
            "Document Depth": float(d["depth_percent"]),
            "Context Length":  int(d["context_length_tokens"]),
            "Score":           d["score"] / 10.0,
        })
    return pd.DataFrame(rows)


def load_subset_a(model_name: str, metric: str = "contains") -> pd.DataFrame:
    """Load Subset A results from PAC-Test evaluate.py (_details.jsonl).
    Falls back to niah_knowledge_results if no evaluate.py results exist.
    """
    # Try niah_knowledge_results first (new token-based, richer grid)
    df = load_knowledge_niah(model_name)
    if not df.empty:
        return df

    # Fallback: evaluate.py details.jsonl (4 char-based lengths)
    safe    = re.sub(r"[:/\\]", "_", model_name)
    pattern = str(EVAL_RESULTS_DIR / f"{safe}_*_details.jsonl")
    files   = sorted(glob.glob(pattern))
    if not files:
        return pd.DataFrame()
    rows = []
    with open(files[-1], encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("subset") != "A":
                continue
            length_chars = int(d.get("total_length",
                             d.get("context_length_chars",
                             d.get("context_length", 0))))
            rows.append({
                "Document Depth": round(float(d["position_ratio"]) * 100, 1),
                "Context Length":  chars_to_tokens(length_chars),
                "Score":           float(d.get(metric, False)),
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def to_pivot(df: pd.DataFrame):
    if df.empty:
        return None
    pivot = (
        df.groupby(["Document Depth", "Context Length"])["Score"]
        .mean()
        .reset_index()
        .pivot(index="Document Depth", columns="Context Length", values="Score")
    )
    # 0 % depth (top of document) at first row => top of heatmap
    pivot = pivot.sort_index(ascending=True)
    pivot = pivot[sorted(pivot.columns)]
    return pivot


# =========================================================
# Core plot  --  exact NIAH visual style
# =========================================================

def plot_niah_heatmap(
    pivot,
    model_name: str,
    title_line1: str,
    title_line2: str,
    goal_text: str,
    out_path,
    today: str = None,
):
    if today is None:
        today = date.today().isoformat()

    fig = plt.figure(figsize=(17.5, 9))

    # --- Heatmap axes: explicit position leaves room on all sides ---
    # [left, bottom, width, height]  all in figure-fraction [0, 1]
    AX_L, AX_B, AX_W, AX_H = 0.13, 0.17, 0.62, 0.70
    ax = fig.add_axes([AX_L, AX_B, AX_W, AX_H])

    masked = np.ma.masked_invalid(pivot.values)
    im = ax.imshow(masked, cmap=NIAH_CMAP, vmin=0, vmax=1, aspect="auto")
    ax.set_facecolor("#EEEEEE")   # missing cells show as light gray

    # Cell borders
    for i in range(pivot.shape[0] + 1):
        ax.axhline(i - 0.5, color="gray", linewidth=0.5, zorder=2)
    for j in range(pivot.shape[1] + 1):
        ax.axvline(j - 0.5, color="gray", linewidth=0.5, zorder=2)

    # Percentage annotations inside cells (skip NaN)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v:.0%}",
                    ha="center", va="center",
                    fontsize=9, fontweight="bold",
                    color="white" if v < 0.38 else "black")

    # X-axis: token limits
    tok_labels = [token_label(c) for c in pivot.columns]
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(tok_labels, rotation=45, ha="right", fontsize=10)
    ax.xaxis.set_tick_params(length=0)
    ax.set_xlabel("Token Limit", fontsize=11, labelpad=8)

    # Y-axis: depth percentages (0 % = top of document = top row)
    dep_labels = [f"{float(d):.0f}%" for d in pivot.index]
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(dep_labels, fontsize=10)
    ax.yaxis.set_tick_params(length=0)
    ax.set_ylabel("")

    for sp in ax.spines.values():
        sp.set_visible(False)

    # ---- Title & subtitle ----
    fig.text(0.50, 0.975, title_line1,
             ha="center", va="top", fontsize=13.5, fontweight="bold")
    fig.text(0.50, 0.935, title_line2,
             ha="center", va="top", fontsize=10.5, color="#555555")

    # Axes bounding box in figure-fraction coords
    p   = ax.get_position()   # x0, y0, x1, y1
    mid = (p.y0 + p.y1) / 2

    # =========================================================
    # LEFT: "Top/Bottom Of Document"  +  big vertical depth label
    # =========================================================
    # Small italic labels flush with y-axis on either end
    fig.text(p.x0 - 0.01, p.y1 + 0.005,
             "Top Of\nDocument",
             ha="right", va="top",
             fontsize=8.5, color="#666666", style="italic",
             multialignment="right")
    fig.text(p.x0 - 0.01, p.y0 - 0.005,
             "Bottom Of\nDocument",
             ha="right", va="bottom",
             fontsize=8.5, color="#666666", style="italic",
             multialignment="right")

    # Big vertical label: rotation=-90 so text reads top-to-bottom
    fig.text(p.x0 - 0.075, mid,
             "Placed Fact Document Depth",
             ha="center", va="center",
             fontsize=11.5, fontweight="bold",
             rotation=-90, color="#333333")

    # Downward arrow below the vertical label (drawn in figure-fraction coords)
    arrow_x = p.x0 - 0.075
    fig.add_artist(FancyArrowPatch(
        posA=(arrow_x, p.y0 + 0.045),
        posB=(arrow_x, p.y0 + 0.005),
        transform=fig.transFigure,
        arrowstyle="->,head_width=4,head_length=6",
        color="#555555",
        linewidth=1.5,
    ))

    # =========================================================
    # RIGHT: colorbar  +  "% Accuracy Of Retrieval" labels
    # =========================================================
    cbar_x  = p.x1 + 0.018
    cbar_ax = fig.add_axes([cbar_x, p.y0, 0.014, p.height])
    cbar    = fig.colorbar(im, cax=cbar_ax)
    cbar.set_ticks([])
    cbar.outline.set_linewidth(0.5)

    acc_x = p.x1 + 0.046
    fig.text(acc_x, p.y1,  "100%\nAccuracy\nOf Retrieval",
             ha="left", va="top",    fontsize=8.5, fontweight="bold",
             color="#0CD79F", multialignment="left")
    fig.text(acc_x, mid,   " 50%\nAccuracy\nOf Retrieval",
             ha="left", va="center", fontsize=8.5, fontweight="bold",
             color="#EBB839", multialignment="left")
    fig.text(acc_x, p.y0,  "  0%\nAccuracy\nOf Retrieval",
             ha="left", va="bottom", fontsize=8.5, fontweight="bold",
             color="#F0496E", multialignment="left")

    # =========================================================
    # BOTTOM: rounded-rectangle goal description box
    # =========================================================
    gbox = fig.add_axes([0.06, 0.005, 0.88, 0.105])
    gbox.set_xlim(0, 1)
    gbox.set_ylim(0, 1)
    gbox.set_axis_off()

    gbox.add_patch(FancyBboxPatch(
        (0.0, 0.0), 1.0, 1.0,
        boxstyle="round,pad=0.04",
        linewidth=1.5,
        edgecolor="#BBBBBB",
        facecolor="#F8F8F8",
        transform=gbox.transAxes,
        clip_on=False,
    ))
    gbox.text(0.5, 0.5, goal_text,
              ha="center", va="center", fontsize=9.5,
              transform=gbox.transAxes, color="#333333")

    # =========================================================
    # TOP-RIGHT: Model + Date (small)
    # =========================================================
    fig.text(0.985, 0.985,
             f"Model: {model_name}    Date: {today}",
             ha="right", va="top", fontsize=8, color="#999999")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# =========================================================
# Main
# =========================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+",
                        default=["qwen2.5:7b", "deepseek-r1:7b", "qwen3.5:9b"])
    parser.add_argument("--metric", default="contains",
                        choices=["contains", "em"])
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()

    for model in args.models:
        safe = re.sub(r"[:/\\]", "_", model)
        print(f"\n--- {model} ---")

        # ---- 1. Synthetic NIAH ----
        df_syn = load_synthetic_niah(model)
        if not df_syn.empty:
            pivot = to_pivot(df_syn)
            if pivot is not None:
                ctx_max   = int(df_syn["Context Length"].max())
                ctx_label = token_label(ctx_max)
                plot_niah_heatmap(
                    pivot,
                    model_name  = model,
                    title_line1 = f'Pressure Testing {model} {ctx_label} Context'
                                  f' via "Needle In A HayStack"',
                    title_line2 = f'Asking {model} To Do Fact Retrieval Across'
                                  ' Context Lengths & Document Depth',
                    goal_text   = (
                        f"Goal: Test {model} Ability To Retrieve A Simple Fact"
                        " From A Large Block Of Text\n"
                        f"Needle: {SYNTHETIC_NEEDLE}\n"
                        f"Retrieval Question: {SYNTHETIC_QUESTION}"
                        f"    |    Expected Answer: {SYNTHETIC_ANSWER}"
                        "    |    Haystack: 通用中文填充句（45条循环复制）"
                    ),
                    out_path    = OUTPUT_DIR / f"{safe}_synthetic_niah.png",
                    today       = today,
                )
        else:
            print(f"  No synthetic NIAH results for {model}")

        # ---- 2. Knowledge-aware NIAH (Subset A) ----
        df_know = load_subset_a(model, metric=args.metric)
        if not df_know.empty:
            pivot = to_pivot(df_know)
            if pivot is not None:
                ctx_max     = int(df_know["Context Length"].max())
                ctx_label   = token_label(ctx_max)
                metric_name = "Contains Match" if args.metric == "contains" else "Exact Match"
                plot_niah_heatmap(
                    pivot,
                    model_name  = model,
                    title_line1 = f'Pressure Testing {model} {ctx_label} Context'
                                  f' via "Needle In A HayStack"',
                    title_line2 = f'Asking {model} To Retrieve Real Domain Knowledge'
                                  f' | {metric_name}',
                    goal_text   = (
                        f"Goal: Test {model} Ability To Retrieve Real Domain Facts"
                        " Under Semantic Interference\n"
                        "Needle: 来自 facts_library.json 的真实领域知识（金融 / CS / 医学 / 法律 / 教育）"
                        "    |    Retrieval Question: {实体}的{属性}是什么？\n"
                        f"Haystack: Subset A 同领域填充句（各域 ~200 条）"
                        f"    |    Metric: {metric_name}"
                    ),
                    out_path    = OUTPUT_DIR / f"{safe}_knowledge_niah_{args.metric}.png",
                    today       = today,
                )
        else:
            print(f"  No Subset A results for {model}")

    # ---- 3. Multi-model comparison (Subset A) ----
    models_ok, pivots_ok = [], []
    for model in args.models:
        df = load_subset_a(model, metric=args.metric)
        p  = to_pivot(df)
        if p is not None:
            models_ok.append(model)
            pivots_ok.append(p)

    if len(pivots_ok) >= 2:
        n   = len(pivots_ok)
        fig, axes = plt.subplots(1, n, figsize=(15 * n, 8), sharey=True)
        if n == 1:
            axes = [axes]

        for ax, pivot, model in zip(axes, pivots_ok, models_ok):
            masked = np.ma.masked_invalid(pivot.values)
            im = ax.imshow(masked, cmap=NIAH_CMAP, vmin=0, vmax=1, aspect="auto")
            ax.set_facecolor("#EEEEEE")
            for i in range(pivot.shape[0] + 1):
                ax.axhline(i - 0.5, color="gray", linewidth=0.5)
            for j in range(pivot.shape[1] + 1):
                ax.axvline(j - 0.5, color="gray", linewidth=0.5)
            for i in range(pivot.shape[0]):
                for j in range(pivot.shape[1]):
                    v = pivot.values[i, j]
                    if np.isnan(v):
                        continue
                    ax.text(j, i, f"{v:.0%}",
                            ha="center", va="center",
                            fontsize=8, fontweight="bold",
                            color="white" if v < 0.38 else "black")
            tok_l = [token_label(c) for c in pivot.columns]
            ax.set_xticks(range(pivot.shape[1]))
            ax.set_xticklabels(tok_l, rotation=45, ha="right", fontsize=9)
            ax.xaxis.set_tick_params(length=0)
            ax.set_xlabel("Token Limit", fontsize=11)
            dep_l = [f"{float(d):.0f}%" for d in pivot.index]
            ax.set_yticks(range(pivot.shape[0]))
            ax.set_yticklabels(dep_l if ax is axes[0] else [], fontsize=9)
            ax.yaxis.set_tick_params(length=0)
            ax.set_title(model, fontsize=13, fontweight="bold", pad=12)
            for sp in ax.spines.values():
                sp.set_visible(False)

        metric_label = "Contains" if args.metric == "contains" else "EM"
        fig.suptitle(
            f'Pressure Testing All Models -- Real Knowledge'
            f' "Needle In A HayStack" | {metric_label}',
            fontsize=15, fontweight="bold", y=1.01,
        )
        plt.tight_layout()
        out = OUTPUT_DIR / f"all_models_knowledge_niah_{args.metric}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\n  Saved comparison: {out}")


if __name__ == "__main__":
    main()
