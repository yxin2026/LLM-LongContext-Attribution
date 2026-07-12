from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = "improve_raw_results.jsonl"


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "results" / "improve" / args.run_id
    raw_path = out_dir / RAW_FILE
    if not raw_path.exists():
        raise SystemExit(f"Raw result file not found: {raw_path}")

    rows = dedupe_rows(read_jsonl(raw_path))
    if not rows:
        raise SystemExit(f"No result rows in {raw_path}")

    cell_rows = build_cell_rows(rows)
    intervention_rows = build_intervention_rows(rows)
    decision = build_decision(args, rows, cell_rows, intervention_rows)

    write_csv(out_dir / "pilot_gate_by_cell.csv", cell_rows)
    write_csv(out_dir / "pilot_gate_by_intervention.csv", intervention_rows)
    (out_dir / "pilot_gate_decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_report(out_dir, args, cell_rows, intervention_rows, decision)
    print_report(out_dir, decision, intervention_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze a PAC improve pilot and decide whether to run full scale.")
    parser.add_argument("--run-id", default="pac_v21_improve_aggressive_pilot")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--min-aggregate-gain", type=float, default=0.05)
    parser.add_argument("--max-format-missing-rate", type=float, default=0.20)
    parser.add_argument("--max-hit-max-token-rate", type=float, default=0.35)
    parser.add_argument("--max-api-error-rate", type=float, default=0.05)
    parser.add_argument("--full-run-id", default=None)
    parser.add_argument("--full-samples-per-model", type=int, default=60)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def task_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row.get("sample_id")), str(row.get("intervention")), str(row.get("model_alias")))


def dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[tuple[str, str, str], tuple[int, dict[str, Any]]] = {}
    for idx, row in enumerate(rows):
        key = task_key(row)
        rank = (1 if row.get("error") in (None, "") else 0, idx)
        old = best.get(key)
        if old is None or rank >= (1 if old[1].get("error") in (None, "") else 0, old[0]):
            best[key] = (idx, row)
    return [row for _idx, row in best.values()]


def build_cell_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[(str(row.get("pac_subset")), str(row.get("intervention")), str(row.get("model_alias")))].append(row)
    return [summarize_bucket(subset, intervention, model, items) for (subset, intervention, model), items in sorted(buckets.items())]


def build_intervention_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[(str(row.get("pac_subset")), str(row.get("intervention")))].append(row)
    return [
        summarize_bucket(subset, intervention, "ALL_MODELS", items)
        for (subset, intervention), items in sorted(buckets.items())
    ]


def summarize_bucket(subset: str, intervention: str, model: str, items: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(items)
    valid = [row for row in items if row.get("error") in (None, "")]
    baseline = [int(row.get("correct_baseline") or 0) for row in valid]
    intervention_scores = [int(row.get("correct_intervention") or 0) for row in valid]
    transitions = Counter(classify_transition(row) for row in valid)
    format_missing = sum(1 for row in valid if is_format_missing(row))
    hit_max = sum(1 for row in valid if hit_max_tokens(row))
    return {
        "pac_subset": subset,
        "intervention": intervention,
        "model_alias": model,
        "n": n,
        "n_valid": len(valid),
        "api_error_rate": round((n - len(valid)) / n, 4) if n else "",
        "baseline_accuracy": round(mean_safe(baseline), 4),
        "intervention_accuracy": round(mean_safe(intervention_scores), 4),
        "absolute_gain": round(mean_safe([i - b for b, i in zip(baseline, intervention_scores)]), 4),
        "fixed": transitions["fixed"],
        "broken": transitions["broken"],
        "kept_correct": transitions["kept_correct"],
        "kept_wrong": transitions["kept_wrong"],
        "format_missing_rate": round(format_missing / len(valid), 4) if valid else "",
        "hit_max_token_rate": round(hit_max / len(valid), 4) if valid else "",
        "mean_completion_tokens": round(mean_safe([float(row.get("completion_tokens") or 0) for row in valid]), 2),
        "mean_latency_sec": round(mean_safe([float(row.get("latency_sec") or 0) for row in valid]), 2),
    }


def classify_transition(row: dict[str, Any]) -> str:
    base = int(row.get("correct_baseline") or 0)
    intervention = int(row.get("correct_intervention") or 0)
    if not base and intervention:
        return "fixed"
    if base and not intervention:
        return "broken"
    if base and intervention:
        return "kept_correct"
    return "kept_wrong"


def is_format_missing(row: dict[str, Any]) -> bool:
    parsed = str(row.get("prediction_parsed") or "").strip()
    if str(row.get("pac_subset")) == "C":
        return not parsed or str(row.get("intervention_error_type") or row.get("error_type")) == "parse_error"
    return not parsed or str(row.get("intervention_error_type") or row.get("error_type")) == "empty_output"


def hit_max_tokens(row: dict[str, Any]) -> bool:
    completion = row.get("completion_tokens")
    max_tokens = row.get("max_tokens")
    if completion is None or max_tokens is None:
        return False
    return float(completion) >= float(max_tokens)


def build_decision(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    cell_rows: list[dict[str, Any]],
    intervention_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    full_run_id = args.full_run_id or default_full_run_id(args.run_id)
    build_command = full_build_command(args, rows, full_run_id)
    recommendations = []
    for row in intervention_rows:
        pass_gate = (
            float(row["absolute_gain"]) >= args.min_aggregate_gain
            and float(row["api_error_rate"] or 0) <= args.max_api_error_rate
            and float(row["format_missing_rate"] or 0) <= args.max_format_missing_rate
            and float(row["hit_max_token_rate"] or 0) <= args.max_hit_max_token_rate
        )
        recommendations.append(
            {
                "pac_subset": row["pac_subset"],
                "intervention": row["intervention"],
                "recommend_full": pass_gate,
                "reason": gate_reason(args, row, pass_gate),
            }
        )
    return {
        "run_id": args.run_id,
        "n_rows": len(rows),
        "thresholds": {
            "min_aggregate_gain": args.min_aggregate_gain,
            "max_format_missing_rate": args.max_format_missing_rate,
            "max_hit_max_token_rate": args.max_hit_max_token_rate,
            "max_api_error_rate": args.max_api_error_rate,
        },
        "recommendations": recommendations,
        "full_build_command_cmd": build_command,
        "full_run_command_cmd": (
            "python scripts\\run_pac_improve_queue.py "
            f"--run-id {full_run_id} --slots-per-key 1 --per-key-delay-sec 20 --queue-max-attempts 4"
        ),
    }


def gate_reason(args: argparse.Namespace, row: dict[str, Any], pass_gate: bool) -> str:
    if pass_gate:
        return "aggregate gain and format/runtime health pass thresholds"
    reasons = []
    if float(row["absolute_gain"]) < args.min_aggregate_gain:
        reasons.append(f"gain {row['absolute_gain']} < {args.min_aggregate_gain}")
    if float(row["api_error_rate"] or 0) > args.max_api_error_rate:
        reasons.append(f"api_error_rate {row['api_error_rate']} > {args.max_api_error_rate}")
    if float(row["format_missing_rate"] or 0) > args.max_format_missing_rate:
        reasons.append(f"format_missing_rate {row['format_missing_rate']} > {args.max_format_missing_rate}")
    if float(row["hit_max_token_rate"] or 0) > args.max_hit_max_token_rate:
        reasons.append(f"hit_max_token_rate {row['hit_max_token_rate']} > {args.max_hit_max_token_rate}")
    return "; ".join(reasons)


def default_full_run_id(run_id: str) -> str:
    if "pilot" in run_id:
        return run_id.replace("pilot", "full")
    return f"{run_id}_full"


def full_build_command(args: argparse.Namespace, rows: list[dict[str, Any]], full_run_id: str) -> str:
    subsets = sorted({str(row.get("pac_subset")) for row in rows if row.get("pac_subset")})
    interventions = sorted({str(row.get("intervention")) for row in rows if row.get("intervention")})
    command = (
        "python scripts\\build_pac_improve_aggressive_dataset.py "
        f"--run-id {full_run_id} --samples-per-model {args.full_samples_per_model}"
    )
    if subsets:
        command += f" --subsets {','.join(subsets)}"
    if "prompt_only_slots" in interventions:
        command += " --b-intervention prompt_only_slots"
    return command


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
    args: argparse.Namespace,
    cell_rows: list[dict[str, Any]],
    intervention_rows: list[dict[str, Any]],
    decision: dict[str, Any],
) -> None:
    lines = [
        "# PAC Improve Aggressive Pilot Gate",
        "",
        "This gate checks small-sample validity before a full B/C aggressive run.",
        "",
        "## Intervention Summary",
        "",
        "| subset | intervention | n | baseline | intervention | gain | format_missing | hit_max | recommend_full |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    rec_by_key = {
        (row["pac_subset"], row["intervention"]): row
        for row in decision["recommendations"]
    }
    for row in intervention_rows:
        rec = rec_by_key[(row["pac_subset"], row["intervention"])]
        lines.append(
            f"| {row['pac_subset']} | {row['intervention']} | {row['n_valid']} | "
            f"{row['baseline_accuracy']} | {row['intervention_accuracy']} | {row['absolute_gain']} | "
            f"{row['format_missing_rate']} | {row['hit_max_token_rate']} | {rec['recommend_full']} |"
        )
    lines.extend(["", "## Cell Summary", ""])
    for row in cell_rows:
        lines.append(
            f"- PAC-{row['pac_subset']} / {row['intervention']} / {row['model_alias']}: "
            f"{row['baseline_accuracy']} -> {row['intervention_accuracy']} "
            f"(gain {row['absolute_gain']}), fixed={row['fixed']}, broken={row['broken']}, "
            f"format_missing={row['format_missing_rate']}, hit_max={row['hit_max_token_rate']}."
        )
    lines.extend(
        [
            "",
            "## Full Commands",
            "",
            "```cmd",
            decision["full_build_command_cmd"],
            decision["full_run_command_cmd"],
            "```",
        ]
    )
    out_dir.joinpath("pilot_gate_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_report(out_dir: Path, decision: dict[str, Any], intervention_rows: list[dict[str, Any]]) -> None:
    print(f"Wrote pilot gate outputs to {out_dir}")
    for row in intervention_rows:
        rec = next(
            item for item in decision["recommendations"]
            if item["pac_subset"] == row["pac_subset"] and item["intervention"] == row["intervention"]
        )
        print(
            f"[GATE] PAC-{row['pac_subset']} {row['intervention']}: "
            f"{row['baseline_accuracy']} -> {row['intervention_accuracy']} "
            f"gain={row['absolute_gain']} recommend_full={rec['recommend_full']} "
            f"reason={rec['reason']}"
        )


if __name__ == "__main__":
    main()
