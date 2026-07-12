from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "results" / "improve" / args.run_id
    raw_path = out_dir / "improve_raw_results.deduped.jsonl"
    if not raw_path.exists() or raw_path.stat().st_size == 0:
        raw_path = out_dir / "improve_raw_results.jsonl"
    if not raw_path.exists():
        raise SystemExit(f"Missing raw results: {raw_path}")
    rows = dedupe(read_jsonl(raw_path))
    plan_by_key = load_plan(out_dir / "improve_plan.jsonl")

    cell_rows = summarize(rows, plan_by_key, by_model=True)
    strategy_rows = summarize(rows, plan_by_key, by_model=False)
    decision = decide(args, strategy_rows, cell_rows)

    write_csv(out_dir / "c_strategy_probe_by_cell.csv", cell_rows)
    write_csv(out_dir / "c_strategy_probe_by_strategy.csv", strategy_rows)
    (out_dir / "c_strategy_probe_decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_report(out_dir, strategy_rows, cell_rows, decision)
    print_report(out_dir, strategy_rows, decision)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze PAC-C strategy probe results.")
    parser.add_argument("--run-id", default="pac_v21_improve_C_strategy_probe")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--min-aggregate-gain", type=float, default=0.05)
    parser.add_argument("--max-empty-value-rate", type=float, default=0.10)
    parser.add_argument("--max-parse-error-rate", type=float, default=0.10)
    parser.add_argument("--max-broken-rate", type=float, default=0.25)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_plan(path: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    if not path.exists():
        return {}
    return {key(row): row for row in read_jsonl(path)}


def key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row.get("sample_id")), str(row.get("intervention")), str(row.get("model_alias")))


def dedupe(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[tuple[str, str, str], tuple[int, dict[str, Any]]] = {}
    for idx, row in enumerate(rows):
        row_key = key(row)
        rank = (1 if row.get("error") in (None, "") else 0, idx)
        old = best.get(row_key)
        if old is None or rank >= (1 if old[1].get("error") in (None, "") else 0, old[0]):
            best[row_key] = (idx, row)
    return [row for _idx, row in best.values()]


def summarize(
    rows: list[dict[str, Any]],
    plan_by_key: dict[tuple[str, str, str], dict[str, Any]],
    by_model: bool,
) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        bucket_key = (
            str(row.get("intervention")),
            str(row.get("model_alias")) if by_model else "ALL_MODELS",
        )
        buckets[bucket_key].append(row)
    out = []
    for (strategy, model), items in sorted(buckets.items()):
        valid = [row for row in items if row.get("error") in (None, "")]
        baseline = [int(row.get("correct_baseline") or 0) for row in valid]
        intervention = [int(row.get("correct_intervention") or 0) for row in valid]
        transitions = Counter(transition(row) for row in valid)
        parse_errors = sum(1 for row in valid if str(row.get("intervention_error_type") or row.get("error_type")) == "parse_error")
        empty_rates = [empty_value_rate(row, plan_by_key.get(key(row), {})) for row in valid]
        out.append(
            {
                "intervention": strategy,
                "model_alias": model,
                "n": len(items),
                "n_valid": len(valid),
                "api_error_rate": round((len(items) - len(valid)) / len(items), 4) if items else "",
                "baseline_accuracy": round(mean_safe(baseline), 4),
                "intervention_accuracy": round(mean_safe(intervention), 4),
                "absolute_gain": round(mean_safe([i - b for b, i in zip(baseline, intervention)]), 4),
                "fixed": transitions["fixed"],
                "broken": transitions["broken"],
                "kept_correct": transitions["kept_correct"],
                "kept_wrong": transitions["kept_wrong"],
                "broken_rate": round(transitions["broken"] / len(valid), 4) if valid else "",
                "parse_error_rate": round(parse_errors / len(valid), 4) if valid else "",
                "empty_value_rate": round(mean_safe(empty_rates), 4),
                "binding_error_rate": round(
                    mean_safe([int(row.get("binding_error_intervention") or 0) for row in valid]),
                    4,
                ),
                "mean_completion_tokens": round(mean_safe([float(row.get("completion_tokens") or 0) for row in valid]), 2),
                "mean_latency_sec": round(mean_safe([float(row.get("latency_sec") or 0) for row in valid]), 2),
            }
        )
    return out


def transition(row: dict[str, Any]) -> str:
    base = int(row.get("correct_baseline") or 0)
    intervention = int(row.get("correct_intervention") or 0)
    if not base and intervention:
        return "fixed"
    if base and not intervention:
        return "broken"
    if base and intervention:
        return "kept_correct"
    return "kept_wrong"


def empty_value_rate(row: dict[str, Any], plan: dict[str, Any]) -> float:
    parsed = row.get("prediction_parsed")
    if not parsed:
        return 1.0
    try:
        parsed_obj = json.loads(parsed) if isinstance(parsed, str) else parsed
    except Exception:
        return 1.0
    if not isinstance(parsed_obj, dict):
        return 1.0
    aliases = plan.get("query_aliases") or []
    if not aliases:
        aliases = list(parsed_obj)
    if not aliases:
        return 1.0
    empty = 0
    for alias in aliases:
        value = parsed_obj.get(str(alias))
        if value is None or str(value).strip() == "":
            empty += 1
    return empty / len(aliases)


def decide(args: argparse.Namespace, strategy_rows: list[dict[str, Any]], cell_rows: list[dict[str, Any]]) -> dict[str, Any]:
    recommendations = []
    for row in strategy_rows:
        ok = (
            float(row["absolute_gain"]) >= args.min_aggregate_gain
            and float(row["empty_value_rate"]) <= args.max_empty_value_rate
            and float(row["parse_error_rate"]) <= args.max_parse_error_rate
            and float(row["broken_rate"]) <= args.max_broken_rate
        )
        recommendations.append(
            {
                "intervention": row["intervention"],
                "recommend_confirm": ok,
                "reason": reason(args, row, ok),
            }
        )
    best = sorted(strategy_rows, key=lambda row: (float(row["absolute_gain"]), -float(row["empty_value_rate"])), reverse=True)
    return {
        "recommendations": recommendations,
        "best_by_gain": best[0]["intervention"] if best else None,
        "thresholds": {
            "min_aggregate_gain": args.min_aggregate_gain,
            "max_empty_value_rate": args.max_empty_value_rate,
            "max_parse_error_rate": args.max_parse_error_rate,
            "max_broken_rate": args.max_broken_rate,
        },
    }


def reason(args: argparse.Namespace, row: dict[str, Any], ok: bool) -> str:
    if ok:
        return "passes gain, empty-value, parse-error, and broken-rate thresholds"
    reasons = []
    if float(row["absolute_gain"]) < args.min_aggregate_gain:
        reasons.append(f"gain {row['absolute_gain']} < {args.min_aggregate_gain}")
    if float(row["empty_value_rate"]) > args.max_empty_value_rate:
        reasons.append(f"empty_value_rate {row['empty_value_rate']} > {args.max_empty_value_rate}")
    if float(row["parse_error_rate"]) > args.max_parse_error_rate:
        reasons.append(f"parse_error_rate {row['parse_error_rate']} > {args.max_parse_error_rate}")
    if float(row["broken_rate"]) > args.max_broken_rate:
        reasons.append(f"broken_rate {row['broken_rate']} > {args.max_broken_rate}")
    return "; ".join(reasons)


def mean_safe(values: list[float | int]) -> float:
    vals = [float(value) for value in values if value is not None]
    return mean(vals) if vals else 0.0


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not fields:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_report(
    out_dir: Path,
    strategy_rows: list[dict[str, Any]],
    cell_rows: list[dict[str, Any]],
    decision: dict[str, Any],
) -> None:
    rec_by_strategy = {row["intervention"]: row for row in decision["recommendations"]}
    lines = [
        "# PAC-C Strategy Probe Report",
        "",
        "| strategy | n | baseline | intervention | gain | empty | parse | broken | recommend |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in strategy_rows:
        rec = rec_by_strategy[row["intervention"]]
        lines.append(
            f"| {row['intervention']} | {row['n_valid']} | {row['baseline_accuracy']} | "
            f"{row['intervention_accuracy']} | {row['absolute_gain']} | {row['empty_value_rate']} | "
            f"{row['parse_error_rate']} | {row['broken_rate']} | {rec['recommend_confirm']} |"
        )
    lines.extend(["", "## By Model", ""])
    for row in cell_rows:
        lines.append(
            f"- {row['intervention']} / {row['model_alias']}: {row['baseline_accuracy']} -> "
            f"{row['intervention_accuracy']} (gain {row['absolute_gain']}), fixed={row['fixed']}, "
            f"broken={row['broken']}, empty={row['empty_value_rate']}, parse={row['parse_error_rate']}."
        )
    out_dir.joinpath("c_strategy_probe_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_report(out_dir: Path, strategy_rows: list[dict[str, Any]], decision: dict[str, Any]) -> None:
    print(f"Wrote C strategy probe analysis to {out_dir}")
    rec_by_strategy = {row["intervention"]: row for row in decision["recommendations"]}
    for row in strategy_rows:
        rec = rec_by_strategy[row["intervention"]]
        print(
            f"[C-PROBE] {row['intervention']}: {row['baseline_accuracy']} -> "
            f"{row['intervention_accuracy']} gain={row['absolute_gain']} "
            f"empty={row['empty_value_rate']} parse={row['parse_error_rate']} "
            f"recommend={rec['recommend_confirm']} reason={rec['reason']}"
        )


if __name__ == "__main__":
    main()
