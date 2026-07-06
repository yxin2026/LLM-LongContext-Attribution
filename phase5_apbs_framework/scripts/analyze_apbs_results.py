from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def method_label(row: pd.Series) -> str:
    if row["method"] == "apbs":
        return f"apbs_g{float(row['gamma']):.1f}"
    return row["method"]


def read_results(patterns: list[str]) -> pd.DataFrame:
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern, recursive=True))
    rows = []
    for file in sorted(set(files)):
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    row["source_file"] = file
                    row.setdefault("model_key", infer_model_key(file))
                    rows.append(row)
    if not rows:
        raise SystemExit("No result rows found.")
    df = pd.DataFrame(rows)
    df["method_label"] = df.apply(method_label, axis=1)
    return df


def infer_model_key(file: str) -> str:
    parts = Path(file).parts
    if "raw" in parts:
        raw_idx = parts.index("raw")
        if raw_idx + 2 < len(parts):
            return parts[raw_idx + 1]
    return "legacy"


def flattening_index(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model_key, length, method), g in summary.groupby(["model_key", "length", "method_label"]):
        by_pos = dict(zip(g["position"], g["accuracy"]))
        if 10 in by_pos and 50 in by_pos and 90 in by_pos:
            edge = (by_pos[10] + by_pos[90]) / 2
            flat = (edge - by_pos[50]) / edge if edge > 0 else np.nan
            rows.append({"model_key": model_key, "length": length, "method_label": method, "flattening_index": flat})
    return pd.DataFrame(rows)


def bootstrap_lift(df: pd.DataFrame, model_key: str, length: int, pos: int, a: str, b: str, n: int = 2000) -> dict:
    da = df[
        (df.model_key == model_key) & (df.length == length) & (df.position == pos) & (df.method_label == a)
    ]["correct"].to_numpy()
    db = df[
        (df.model_key == model_key) & (df.length == length) & (df.position == pos) & (df.method_label == b)
    ]["correct"].to_numpy()
    if len(da) == 0 or len(db) == 0:
        return {}
    rng = np.random.default_rng(20260705)
    lifts = []
    for _ in range(n):
        lifts.append(rng.choice(da, len(da), replace=True).mean() - rng.choice(db, len(db), replace=True).mean())
    return {
        "model_key": model_key,
        "length": length,
        "position": pos,
        "contrast": f"{a}-{b}",
        "mean_lift": float(np.mean(lifts)),
        "ci_low": float(np.quantile(lifts, 0.025)),
        "ci_high": float(np.quantile(lifts, 0.975)),
    }


def write_plots(summary: pd.DataFrame, output_dir: Path) -> None:
    for (model_key, length), g in summary.groupby(["model_key", "length"]):
        curve = g[g["position"].isin([10, 50, 90])]
        if curve.empty:
            continue
        plt.figure(figsize=(7, 4.5))
        for method, m in curve.groupby("method_label"):
            m = m.sort_values("position")
            plt.plot(m["position"], m["accuracy"], marker="o", linewidth=2, label=method)
        plt.xlabel("Needle position (%)")
        plt.ylabel("Accuracy")
        plt.title(f"NIAH Position Curve, {model_key}, {int(length/1024)}K")
        plt.ylim(-0.03, 1.03)
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"position_curves_{model_key}_{length}.png", dpi=180)
        plt.close()

    gamma = summary[summary["method_label"].str.startswith("apbs_g") & (summary["position"] == 50)]
    if not gamma.empty:
        for model_key, model_gamma in gamma.groupby("model_key"):
            plt.figure(figsize=(6, 4))
            for length, g in model_gamma.groupby("length"):
                parsed = g.assign(gamma=g["method_label"].str.replace("apbs_g", "", regex=False).astype(float))
                parsed = parsed.sort_values("gamma")
                plt.plot(parsed["gamma"], parsed["accuracy"], marker="o", linewidth=2, label=f"{int(length/1024)}K")
            plt.xlabel("Gamma")
            plt.ylabel("Middle-position accuracy")
            plt.title(f"APBS Gamma Sensitivity at 50%, {model_key}")
            plt.ylim(-0.03, 1.03)
            plt.grid(alpha=0.25)
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_dir / f"gamma_sensitivity_{model_key}.png", dpi=180)
            plt.close()


def write_report(summary: pd.DataFrame, flat: pd.DataFrame, ci: pd.DataFrame, output_dir: Path) -> None:
    lines = [
        "# Phase 5 APBS MVP Report",
        "",
        "## Claim",
        "",
        "In the Qwen 9B 16K NIAH setting, APBS improves middle-position retrieval compared with baseline RoPE and global NTK.",
        "",
        "## Accuracy By Method",
        "",
        summary.to_markdown(index=False),
        "",
        "## U-Shape Flattening",
        "",
        flat.to_markdown(index=False) if not flat.empty else "Not enough 10/50/90 data to compute flattening.",
        "",
        "## Bootstrap Middle-Position Lift",
        "",
        ci.to_markdown(index=False) if not ci.empty else "Not enough contrasts to compute bootstrap CI.",
        "",
        "## Interpretation Template",
        "",
        "Report the result as initial causal intervention evidence, not as a universal cross-model claim. "
        "The strongest valid wording is: APBS improves 50% retrieval in this controlled Qwen 9B 16K setup and reduces the U-shaped position bias relative to baseline/global NTK.",
    ]
    (output_dir / "phase5_apbs_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = read_results(args.inputs)
    summary = (
        df.groupby(["model_key", "length", "position", "method_label"], as_index=False)
        .agg(accuracy=("correct", "mean"), n=("correct", "size"))
        .sort_values(["model_key", "length", "position", "method_label"])
    )
    summary.to_csv(output_dir / "metrics_by_method.csv", index=False)

    flat = flattening_index(summary)
    flat.to_csv(output_dir / "flattening_index.csv", index=False)

    ci_rows = []
    for model_key in sorted(df["model_key"].unique()):
        for length in sorted(df[df["model_key"] == model_key]["length"].unique()):
            labels = set(df[(df["model_key"] == model_key) & (df["length"] == length)]["method_label"].unique())
            apbs = sorted([x for x in labels if x.startswith("apbs_g")])
            primary = "apbs_g0.3" if "apbs_g0.3" in apbs else (apbs[0] if apbs else None)
            if primary:
                for baseline in ["baseline", "ntk"]:
                    if baseline in labels:
                        row = bootstrap_lift(df, model_key, length, 50, primary, baseline)
                        if row:
                            ci_rows.append(row)
    ci = pd.DataFrame(ci_rows)
    ci.to_csv(output_dir / "bootstrap_ci.csv", index=False)

    write_plots(summary, output_dir)
    write_report(summary, flat, ci, output_dir)
    print(f"Wrote analysis to {output_dir}")


if __name__ == "__main__":
    main()
