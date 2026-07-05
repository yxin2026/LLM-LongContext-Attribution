from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lmaf.utils.io import ensure_parent


def main() -> None:
    args = parse_args()
    rows = read_csv(args.input)
    if args.plot in {"niah_position_curve", "pac_A_position_curve"}:
        plot_position_curve(rows, args.output, title=args.plot)
    elif args.plot == "pac_B_density_curve":
        plot_density_curve(rows, args.output)
    elif args.plot == "ruler_effective_context":
        plot_ruler_ecl(rows, args.output)
    elif args.plot == "pac_C_confusion_matrix":
        plot_error_bars(rows, args.output, "C_overlap")
    elif args.plot == "pac_D_multihop_decay":
        plot_multihop(rows, args.output)
    else:
        raise SystemExit(f"Unknown plot type: {args.plot}")
    print(f"Wrote plot to {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot aggregate experiment results.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--plot", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def read_csv(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def plot_position_curve(rows: list[dict[str, Any]], output: str | Path, title: str) -> None:
    plt = _plt()
    series: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        if row.get("position") in ("", None) or row.get("accuracy") in ("", None):
            continue
        series[(row.get("model") or "model", str(row.get("length") or ""))].append(
            (float(row["position"]), float(row["accuracy"]))
        )
    for (model, length), points in sorted(series.items()):
        points = sorted(points)
        plt.plot([p[0] for p in points], [p[1] for p in points], marker="o", label=f"{model} {length}")
    plt.xlabel("Position (%)")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.ylim(0, 1.05)
    plt.legend()
    _save(plt, output)


def plot_density_curve(rows: list[dict[str, Any]], output: str | Path) -> None:
    plt = _plt()
    series: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        if row.get("density") in ("", None) or row.get("accuracy") in ("", None):
            continue
        series[(row.get("model") or "model", row.get("interference_type") or "unknown")].append(
            (float(row["density"]), float(row["accuracy"]))
        )
    for (model, kind), points in sorted(series.items()):
        points = sorted(points)
        plt.plot([p[0] for p in points], [p[1] for p in points], marker="o", label=f"{model} {kind}")
    plt.xlabel("Interference density (%)")
    plt.ylabel("Accuracy")
    plt.title("PAC B density curve")
    plt.ylim(0, 1.05)
    plt.legend()
    _save(plt, output)


def plot_ruler_ecl(rows: list[dict[str, Any]], output: str | Path) -> None:
    plt = _plt()
    values: dict[str, float] = {}
    for row in rows:
        key = f"{row.get('model')}/{row.get('subtask')}"
        value = row.get("effective_context_length")
        if value not in ("", None):
            values[key] = max(values.get(key, 0), float(value))
    labels = list(values)
    heights = [values[label] for label in labels]
    plt.bar(range(len(labels)), heights)
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.ylabel("Effective context length")
    plt.title("RULER effective context")
    _save(plt, output)


def plot_error_bars(rows: list[dict[str, Any]], output: str | Path, subtask: str) -> None:
    plt = _plt()
    counts = defaultdict(int)
    for row in rows:
        if row.get("subtask") == subtask and row.get("top_error_type"):
            counts[row["top_error_type"]] += int(float(row.get("n_samples") or 0))
    labels = list(counts)
    heights = [counts[label] for label in labels]
    plt.bar(range(len(labels)), heights)
    plt.xticks(range(len(labels)), labels, rotation=35, ha="right")
    plt.ylabel("Grouped sample count")
    plt.title(f"{subtask} top error types")
    _save(plt, output)


def plot_multihop(rows: list[dict[str, Any]], output: str | Path) -> None:
    plt = _plt()
    series: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        if row.get("hops") in ("", None) or row.get("accuracy") in ("", None):
            continue
        series[row.get("model") or "model"].append((float(row["hops"]), float(row["accuracy"])))
    for model, points in sorted(series.items()):
        points = sorted(points)
        plt.plot([p[0] for p in points], [p[1] for p in points], marker="o", label=model)
    plt.xlabel("Hop count")
    plt.ylabel("Accuracy")
    plt.title("PAC D multihop decay")
    plt.ylim(0, 1.05)
    plt.legend()
    _save(plt, output)


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(9, 5))
    return plt


def _save(plt: Any, output: str | Path) -> None:
    ensure_parent(output)
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


if __name__ == "__main__":
    main()

