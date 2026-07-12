from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lmaf.eval.metrics import linear_slope, pearson
from lmaf.utils.io import collect_jsonl, ensure_parent
from lmaf.utils.io import TERMINAL_NONRETRY_ERRORS
from lmaf.utils.models import is_excluded_model


FIELDNAMES = [
    "experiment",
    "subtask",
    "model",
    "provider",
    "api_model",
    "length",
    "position",
    "density",
    "interference_type",
    "similarity",
    "distance",
    "hops",
    "hop_distance",
    "chain_type",
    "implementation",
    "n_samples",
    "n_success",
    "accuracy",
    "f1",
    "rouge_l",
    "mean_latency",
    "p50_latency",
    "p95_latency",
    "error_rate",
    "effective_context_length",
    "effective_context_length_abs80",
    "middle_drop",
    "relative_middle_drop",
    "density_accuracy_slope",
    "density_accuracy_pearson",
    "critical_density_threshold",
    "top_error_type",
]


def main() -> None:
    args = parse_args()
    rows = collect_jsonl(args.input)
    if args.experiment:
        rows = [row for row in rows if row.get("experiment") == args.experiment]
    if not args.include_excluded_models:
        rows = [
            row
            for row in rows
            if not is_excluded_model(row.get("model")) and not is_excluded_model(row.get("api_model"))
        ]
    if not args.include_skipped:
        rows = [row for row in rows if row.get("error") not in TERMINAL_NONRETRY_ERRORS]
    grouped = aggregate(rows)
    enrich_ruler_ecl(grouped)
    enrich_pac_stats(grouped, rows)
    write_csv(args.output, grouped)
    print(f"Wrote {len(grouped)} aggregate rows to {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate JSONL experiment results into CSV.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--include-excluded-models", action="store_true")
    parser.add_argument("--include-skipped", action="store_true")
    return parser.parse_args()


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row.get("experiment"),
            row.get("subtask") or row.get("task"),
            row.get("model"),
            row.get("provider"),
            row.get("api_model"),
            row.get("length_tokens_target") or row.get("length"),
            row.get("position_percent"),
            row.get("density"),
            row.get("interference_type"),
            row.get("similarity"),
            row.get("distance"),
            row.get("hops"),
            row.get("hop_distance"),
            row.get("chain_type"),
            row.get("implementation"),
        )
        buckets[key].append(row)

    out: list[dict[str, Any]] = []
    for key, items in buckets.items():
        scores = [_float(row.get("score")) for row in items if _float(row.get("score")) is not None]
        f1s = [_float(row.get("f1") or row.get("partial_f1")) for row in items if _float(row.get("f1") or row.get("partial_f1")) is not None]
        rouges = [_float(row.get("rouge_l")) for row in items if _float(row.get("rouge_l")) is not None]
        latencies = [_float(row.get("latency_sec")) for row in items if _float(row.get("latency_sec")) is not None]
        errors = [row for row in items if row.get("error") not in (None, "")]
        error_types = Counter(str(row.get("error_type")) for row in items if row.get("error_type"))
        out.append(
            {
                "experiment": key[0],
                "subtask": key[1],
                "model": key[2],
                "provider": key[3],
                "api_model": key[4],
                "length": key[5],
                "position": key[6],
                "density": key[7],
                "interference_type": key[8],
                "similarity": key[9],
                "distance": key[10],
                "hops": key[11],
                "hop_distance": key[12],
                "chain_type": key[13],
                "implementation": key[14],
                "n_samples": len(items),
                "n_success": len(items) - len(errors),
                "accuracy": _mean(scores),
                "f1": _mean(f1s),
                "rouge_l": _mean(rouges),
                "mean_latency": _mean(latencies),
                "p50_latency": _percentile(latencies, 50),
                "p95_latency": _percentile(latencies, 95),
                "error_rate": len(errors) / len(items) if items else 0,
                "top_error_type": error_types.most_common(1)[0][0] if error_types else "",
            }
        )
    return sorted(out, key=lambda row: tuple(str(row.get(field, "")) for field in FIELDNAMES[:12]))


def enrich_ruler_ecl(grouped: list[dict[str, Any]]) -> None:
    by_model_task: dict[tuple[Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for row in grouped:
        if row.get("experiment") == "ruler":
            by_model_task[(row.get("model"), row.get("subtask"))].append(row)
    for rows in by_model_task.values():
        with_lengths = [row for row in rows if _float(row.get("length")) is not None]
        if not with_lengths:
            continue
        min_len = min(int(row["length"]) for row in with_lengths)
        base_accs = [float(row["accuracy"]) for row in with_lengths if int(row["length"]) == min_len and row.get("accuracy") != ""]
        if not base_accs:
            continue
        threshold = 0.85 * mean(base_accs)
        ecl = _max_len_at_threshold(with_lengths, threshold)
        ecl_abs80 = _max_len_at_threshold(with_lengths, 0.80)
        for row in with_lengths:
            row["effective_context_length"] = ecl
            row["effective_context_length_abs80"] = ecl_abs80


def enrich_pac_stats(grouped: list[dict[str, Any]], raw_rows: list[dict[str, Any]]) -> None:
    enrich_pac_a(grouped)
    enrich_pac_b(grouped)
    enrich_pac_error_types(grouped, raw_rows)


def enrich_pac_a(grouped: list[dict[str, Any]]) -> None:
    by_model_len: dict[tuple[Any, Any], dict[int, float]] = defaultdict(dict)
    for row in grouped:
        if row.get("experiment") == "pac" and row.get("subtask") == "A_position" and row.get("position") not in ("", None):
            by_model_len[(row.get("model"), row.get("length"))][int(row["position"])] = float(row["accuracy"])
    for (model, length), pos_acc in by_model_len.items():
        if 50 not in pos_acc:
            continue
        edge_values = [pos_acc[p] for p in (10, 90) if p in pos_acc]
        if not edge_values:
            continue
        edge_mean = mean(edge_values)
        middle_drop = edge_mean - pos_acc[50]
        relative = middle_drop / edge_mean if edge_mean else math.nan
        for row in grouped:
            if row.get("experiment") == "pac" and row.get("subtask") == "A_position" and row.get("model") == model and row.get("length") == length:
                row["middle_drop"] = middle_drop
                row["relative_middle_drop"] = relative


def enrich_pac_b(grouped: list[dict[str, Any]]) -> None:
    by_model_kind: dict[tuple[Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for row in grouped:
        if row.get("experiment") == "pac" and row.get("subtask") == "B_interference" and row.get("density") not in ("", None):
            by_model_kind[(row.get("model"), row.get("interference_type"))].append(row)
    for rows in by_model_kind.values():
        xs = [float(row["density"]) for row in rows]
        ys = [float(row["accuracy"]) for row in rows]
        slope = linear_slope(xs, ys)
        corr = pearson(xs, ys)
        threshold = ""
        for density, acc in sorted(zip(xs, ys)):
            if acc < 0.80:
                threshold = density
                break
        for row in rows:
            row["density_accuracy_slope"] = slope
            row["density_accuracy_pearson"] = corr
            row["critical_density_threshold"] = threshold


def enrich_pac_error_types(grouped: list[dict[str, Any]], raw_rows: list[dict[str, Any]]) -> None:
    counts: dict[tuple[Any, ...], Counter[str]] = defaultdict(Counter)
    for row in raw_rows:
        if row.get("experiment") != "pac" or not row.get("error_type"):
            continue
        key = (
            row.get("subtask"),
            row.get("model"),
            row.get("similarity"),
            row.get("distance"),
            row.get("hops"),
        )
        counts[key][str(row["error_type"])] += 1
    for row in grouped:
        key = (
            row.get("subtask"),
            row.get("model"),
            row.get("similarity"),
            row.get("distance"),
            row.get("hops"),
        )
        if counts.get(key):
            row["top_error_type"] = counts[key].most_common(1)[0][0]


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out = ensure_parent(path)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDNAMES})


def _max_len_at_threshold(rows: list[dict[str, Any]], threshold: float) -> int | str:
    candidates = [int(row["length"]) for row in rows if row.get("accuracy") != "" and float(row["accuracy"]) >= threshold]
    return max(candidates) if candidates else ""


def _float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: list[float]) -> float | str:
    return mean(values) if values else ""


def _percentile(values: list[float], percentile: int) -> float | str:
    if not values:
        return ""
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * percentile / 100
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[int(index)]
    return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)


if __name__ == "__main__":
    main()
