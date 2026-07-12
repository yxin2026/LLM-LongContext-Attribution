from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import build_pac_improve_dataset as base


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "PAC" / "data" / "PAC-C_binding_capacity" / "samples.jsonl"
BASELINE_SUBDIR = "PAC-C_binding_capacity"
DEFAULT_MODELS = ["qwen35_9b", "qwen35_122b_a10b"]
DEFAULT_STRATEGIES = [
    "c_alias_card_json",
    "c_guarded_alias_json",
    "c_compact_bind_json",
]

STRATEGY_MAX_TOKENS = {
    "c_alias_card_json": 384,
    "c_guarded_alias_json": 384,
    "c_compact_bind_json": 512,
}


def main() -> None:
    args = parse_args()
    strategies = base.parse_csv(args.strategies)
    bad = sorted(set(strategies) - set(DEFAULT_STRATEGIES))
    if bad:
        raise SystemExit(f"Unsupported C strategies: {bad}")
    models = base.parse_csv(args.models)
    baseline_root = Path(args.baseline_root)
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "results" / "improve" / args.run_id
    base.guard_output_dir(out_dir)
    base.INTERVENTION_MAX_TOKENS.update(STRATEGY_MAX_TOKENS)

    source_samples = base.load_source_samples(SOURCE_PATH)
    baselines = {
        model: base.load_baseline_file(baseline_root / BASELINE_SUBDIR / f"{model}.jsonl")
        if (baseline_root / BASELINE_SUBDIR / f"{model}.jsonl").exists()
        else {}
        for model in models
    }
    available_by_model = {model: set(rows).intersection(source_samples) for model, rows in baselines.items()}
    nonempty_sets = [ids for ids in available_by_model.values() if ids]
    common_ids = set.intersection(*nonempty_sets) if nonempty_sets else set()
    common_selected = base.stratified_select(
        sorted(common_ids),
        source_samples,
        args.samples_per_model,
        base.stratum_fn_for("C"),
        seed=args.seed,
        prefer_fn=base.prefer_fn_for("C"),
    )
    common_selected_set = set(common_selected)

    plan_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    for strategy in strategies:
        for model in models:
            baseline_rows = baselines.get(model, {})
            available_ids = sorted(set(baseline_rows).intersection(source_samples))
            selected = list(common_selected)
            if len(selected) < args.samples_per_model:
                selected_set = set(selected)
                selected.extend(
                    base.stratified_select(
                        [sample_id for sample_id in available_ids if sample_id not in selected_set],
                        source_samples,
                        args.samples_per_model - len(selected),
                        base.stratum_fn_for("C"),
                        seed=args.seed + len(model) * 103 + len(strategy) * 17,
                        prefer_fn=base.prefer_fn_for("C"),
                    )
                )
            selected = [sample_id for sample_id in selected if sample_id in baseline_rows]
            coverage_rows.append(
                {
                    "pac_subset": "C",
                    "intervention": strategy,
                    "model_alias": model,
                    "requested": args.samples_per_model,
                    "source_samples": len(source_samples),
                    "baseline_available": len(available_ids),
                    "common_across_available_models": len(common_ids),
                    "common_selected": sum(1 for sample_id in selected if sample_id in common_selected_set),
                    "selected": len(selected),
                    "shortfall": max(0, args.samples_per_model - len(selected)),
                }
            )
            if not selected:
                missing_rows.append(
                    {
                        "pac_subset": "C",
                        "intervention": strategy,
                        "model_alias": model,
                        "reason": "no_paired_baseline_for_model_subset",
                        "baseline_dir": str(baseline_root / BASELINE_SUBDIR),
                    }
                )
                continue

            for sample_id in selected:
                sample = source_samples[sample_id]
                prompt = build_c_strategy_prompt(sample, strategy)
                row = base.build_plan_row(
                    sample=sample,
                    baseline=baseline_rows[sample_id],
                    subset_key="C",
                    intervention=strategy,
                    model_alias=model,
                    intervention_prompt=prompt,
                    common_sample=sample_id in common_selected_set,
                )
                row["prompt_version"] = f"{strategy}_v1_probe"
                row["strategy_family"] = "structured_binding_prompt_engineering"
                row["strategy_design_note"] = design_note(strategy)
                plan_rows.append(row)

    plan_rows.sort(key=lambda r: (r["intervention"], r["model_alias"], r["sample_id"]))
    config = {
        "run_id": args.run_id,
        "created_at": base.utc_now(),
        "purpose": "PAC-C structured binding prompt-engineering strategy probe. Baseline is read-only and not rerun.",
        "samples_per_model_requested": args.samples_per_model,
        "models": models,
        "strategies": strategies,
        "baseline_root": str(baseline_root),
        "source_path": str(SOURCE_PATH),
        "output_dir": str(out_dir),
        "boundaries": [
            "source sample_id is reused",
            "gold answer is unchanged",
            "answer values are not inserted into added prompt text",
            "SRC is explicitly not used as a shortcut, priority signal, or new filter",
            "source PAC/data files are not modified",
            "baseline raw files are not modified",
        ],
    }

    if args.dry_run:
        print_report(plan_rows, coverage_rows, missing_rows, out_dir, dry_run=True)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "run_config.json", config)
    write_jsonl(out_dir / "improve_plan.jsonl", plan_rows)
    write_jsonl(out_dir / "intervention_prompts.jsonl", base.intervention_prompt_rows(plan_rows))
    write_csv(out_dir / "coverage_report.csv", coverage_rows)
    write_csv(out_dir / "missing_baseline.csv", missing_rows)
    write_readme(out_dir, config, plan_rows, missing_rows)
    write_runner_hint(out_dir, plan_rows)
    print_report(plan_rows, coverage_rows, missing_rows, out_dir, dry_run=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a PAC-C structured binding strategy probe.")
    parser.add_argument("--run-id", default="pac_v21_improve_C_strategy_probe")
    parser.add_argument("--samples-per-model", type=int, default=12)
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--strategies", default=",".join(DEFAULT_STRATEGIES))
    parser.add_argument(
        "--baseline-root",
        default=str(ROOT / "results" / "raw" / "pac_v21_queue" / "pac_v21_full_queue"),
    )
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_c_strategy_prompt(sample: dict[str, Any], strategy: str) -> str:
    parts = base.split_prompt(str(sample.get("prompt") or ""))
    if not parts:
        return str(sample.get("prompt") or "")
    task, context, question = parts["task"], parts["context"], parts["question"]
    if strategy == "c_alias_card_json":
        return base.compose_prompt(
            task=c_alias_card_task(task, sample, guarded=False, compact_public=False),
            context=context,
            question=c_json_question(question, sample, compact_public=False),
        )
    if strategy == "c_guarded_alias_json":
        return base.compose_prompt(
            task=c_alias_card_task(task, sample, guarded=True, compact_public=False),
            context=context,
            question=c_json_question(question, sample, compact_public=False),
        )
    if strategy == "c_compact_bind_json":
        return base.compose_prompt(
            task=c_alias_card_task(task, sample, guarded=True, compact_public=True),
            context=context,
            question=c_json_question(question, sample, compact_public=True),
        )
    raise ValueError(strategy)


def c_alias_card_task(original_task: str, sample: dict[str, Any], guarded: bool, compact_public: bool) -> str:
    aliases = [str(item) for item in sample.get("query_aliases") or []]
    alias_lines = "\n".join(f"- {idx}. {alias}" for idx, alias in enumerate(aliases, start=1))
    guard_text = (
        "\nCompleteness guard: every requested alias must receive a non-empty V value. "
        "If a value appears missing, silently re-check the active alias line and the matching evidence before answering. "
        "Do not output empty strings unless the original task truly has no valid record."
        if guarded
        else ""
    )
    public_text = (
        "\nYou may output a compact BIND line for each requested alias before the JSON, but each BIND line must contain "
        "only alias=value. Do not include evidence lists or rejected candidates."
        if compact_public
        else "\nThink silently. Do not write tables, explanations, evidence lists, or rejected candidates."
    )
    return (
        f"{original_task}\n\n"
        "Prompt-only structured binding intervention for PAC-C.\n"
        "This is only a reasoning and output-format aid. Do not add new data rules beyond the original task. "
        "Do not use SRC as a shortcut, priority signal, tie breaker, or filter.\n"
        "For each requested alias independently:\n"
        "1. Resolve exactly that alias using AS=active.\n"
        "2. Keep the resolved entity bound to that alias.\n"
        "3. Use only the requested approved PID/profile, requested field, requested batch, valid status, "
        "approved review, primary channel, and strict before-cutoff time.\n"
        "4. Reject near aliases, near entities, wrong batch, wrong field, inactive aliases, revised facts, "
        "secondary channel, pending review, draft, revoked, expired, and after-cutoff records.\n"
        f"{guard_text}"
        f"{public_text}\n\n"
        "Locked alias order:\n"
        f"{alias_lines}"
    )


def c_json_question(question: str, sample: dict[str, Any], compact_public: bool) -> str:
    aliases = [str(item) for item in sample.get("query_aliases") or []]
    shape = "{" + ", ".join(f'"{alias}":""' for alias in aliases) + "}"
    prefix = (
        "Optionally write compact lines as BIND alias=value, one per requested alias. "
        "Then write the final JSON line.\n"
        if compact_public
        else "Return exactly one single line and nothing else.\n"
    )
    return (
        f"{question}\n\n"
        f"{prefix}"
        f"ANSWER_JSON: {shape}"
    )


def design_note(strategy: str) -> str:
    if strategy == "c_alias_card_json":
        return "Minimal alias-order card plus silent per-alias binding checklist; no SRC shortcut."
    if strategy == "c_guarded_alias_json":
        return "Alias-order card plus non-empty completeness guard to target omitted alias values; no SRC shortcut."
    if strategy == "c_compact_bind_json":
        return "Compact public alias=value binding lines before JSON, bounded by query count; no SRC shortcut."
    return ""


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


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


def write_readme(
    out_dir: Path,
    config: dict[str, Any],
    plan_rows: list[dict[str, Any]],
    missing_rows: list[dict[str, Any]],
) -> None:
    counts = Counter((row["intervention"], row["model_alias"]) for row in plan_rows)
    lines = [
        "# PAC-C Strategy Probe",
        "",
        "Derived C-only prompt-engineering strategy probe. Source PAC data and baseline raw results are read only.",
        "",
        f"- run_id: `{config['run_id']}`",
        f"- created_at: `{config['created_at']}`",
        f"- planned calls: `{len(plan_rows)}`",
        "",
        "| strategy | model | selected |",
        "|---|---|---:|",
    ]
    for (strategy, model), count in sorted(counts.items()):
        lines.append(f"| {strategy} | {model} | {count} |")
    lines.extend(["", "## Missing Baseline", ""])
    if missing_rows:
        for row in missing_rows:
            lines.append(f"- {row['intervention']} / {row['model_alias']}: {row['reason']}")
    else:
        lines.append("- None.")
    out_dir.joinpath("README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_runner_hint(out_dir: Path, plan_rows: list[dict[str, Any]]) -> None:
    text = (
        "# Runner Hint\n\n"
        "Run the C strategy probe, then analyze which strategy should receive a larger confirmatory run.\n\n"
        "```cmd\n"
        "cd /d C:\\Users\\GET-DATA402\\Documents\\Codex\\2026-06-26\\files-mentioned-by-the-user-lmaf\\lmaf_experiments\n"
        "set SILICONFLOW_API_KEYS=sk-1,sk-2\n"
        f"python scripts\\run_pac_improve_queue.py --run-id {out_dir.name} --slots-per-key 1 --per-key-delay-sec 20 --queue-max-attempts 4\n"
        f"python scripts\\analyze_pac_c_strategy_probe.py --run-id {out_dir.name}\n"
        "```\n"
    )
    out_dir.joinpath("runner_hint.md").write_text(text, encoding="utf-8")


def print_report(
    plan_rows: list[dict[str, Any]],
    coverage_rows: list[dict[str, Any]],
    missing_rows: list[dict[str, Any]],
    out_dir: Path,
    dry_run: bool,
) -> None:
    label = "DRY-RUN" if dry_run else "WROTE"
    print(f"[{label}] C strategy probe output dir: {out_dir}")
    print(f"[{label}] planned calls: {len(plan_rows)}")
    for row in coverage_rows:
        print(
            "[COVERAGE] "
            f"C {row['intervention']} {row['model_alias']} requested={row['requested']} "
            f"available={row['baseline_available']} selected={row['selected']} shortfall={row['shortfall']}"
        )
    for row in missing_rows:
        print(f"[MISSING] C {row['intervention']} {row['model_alias']}: {row['reason']}")


if __name__ == "__main__":
    main()
