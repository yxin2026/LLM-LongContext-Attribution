from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import build_pac_improve_dataset as base


ROOT = Path(__file__).resolve().parents[1]

SOURCE_SUBSETS = {
    "B": {
        "subset_dir": "PAC-B_interference",
        "source_path": ROOT / "PAC" / "data" / "PAC-B_interference" / "samples.jsonl",
        "intervention": "source_priority_slots",
        "pac_subset": "B",
    },
    "C": {
        "subset_dir": "PAC-C_binding_capacity",
        "source_path": ROOT / "PAC" / "data" / "PAC-C_binding_capacity" / "samples.jsonl",
        "intervention": "json_only_binding",
        "pac_subset": "C",
    },
}

INTERVENTION_MAX_TOKENS = {
    "source_priority_slots": 128,
    "prompt_only_slots": 128,
    "json_only_binding": 384,
}


def main() -> None:
    args = parse_args()
    models = base.parse_csv(args.models)
    subsets = base.parse_csv(args.subsets)
    bad_subsets = sorted(set(subsets) - set(SOURCE_SUBSETS))
    if bad_subsets:
        raise SystemExit(f"Unsupported aggressive subsets: {bad_subsets}. Use B,C.")

    baseline_root = Path(args.baseline_root)
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "results" / "improve" / args.run_id
    base.guard_output_dir(out_dir)
    base.INTERVENTION_MAX_TOKENS.update(INTERVENTION_MAX_TOKENS)

    source_by_subset = {
        key: base.load_source_samples(SOURCE_SUBSETS[key]["source_path"])
        for key in subsets
    }
    baseline_by_subset_model = load_aggressive_baselines(baseline_root, models, subsets)

    plan_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []

    for subset_key in subsets:
        config = SOURCE_SUBSETS[subset_key]
        source_samples = source_by_subset[subset_key]
        available_by_model = {
            model: set(baseline_by_subset_model[subset_key].get(model, {}).keys())
            for model in models
        }
        nonempty_model_sets = [ids for ids in available_by_model.values() if ids]
        common_ids = set.intersection(*nonempty_model_sets) if nonempty_model_sets else set()
        common_selected = base.stratified_select(
            sorted(common_ids),
            source_samples,
            args.samples_per_model,
            base.stratum_fn_for(subset_key),
            seed=args.seed + ord(subset_key),
            prefer_fn=base.prefer_fn_for(subset_key),
        )
        common_selected_set = set(common_selected)

        for model in models:
            baseline_rows = baseline_by_subset_model[subset_key].get(model, {})
            available_ids = sorted(set(baseline_rows).intersection(source_samples))
            selected = list(common_selected)
            if len(selected) < args.samples_per_model:
                selected_set = set(selected)
                selected.extend(
                    base.stratified_select(
                        [sample_id for sample_id in available_ids if sample_id not in selected_set],
                        source_samples,
                        args.samples_per_model - len(selected),
                        base.stratum_fn_for(subset_key),
                        seed=args.seed + len(model) * 101 + ord(subset_key),
                        prefer_fn=base.prefer_fn_for(subset_key),
                    )
                )
            selected = [sample_id for sample_id in selected if sample_id in baseline_rows]

            intervention = intervention_for_subset(subset_key, args)
            if not selected:
                missing_rows.append(
                    {
                        "pac_subset": subset_key,
                        "intervention": intervention,
                        "model_alias": model,
                        "reason": "no_paired_baseline_for_model_subset",
                        "baseline_dir": str(baseline_root / config["subset_dir"]),
                    }
                )

            coverage_rows.append(
                {
                    "pac_subset": subset_key,
                    "intervention": intervention,
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

            for sample_id in selected:
                sample = source_samples[sample_id]
                baseline = baseline_rows[sample_id]
                intervention_prompt = build_aggressive_prompt(sample, intervention)
                row = base.build_plan_row(
                    sample=sample,
                    baseline=baseline,
                    subset_key=subset_key,
                    intervention=intervention,
                    model_alias=model,
                    intervention_prompt=intervention_prompt,
                    common_sample=sample_id in common_selected_set,
                )
                row["prompt_version"] = f"{intervention}_v2_aggressive"
                row["aggressive_design_note"] = design_note_for(intervention)
                plan_rows.append(row)

    plan_rows.sort(key=lambda r: (r["pac_subset"], r["intervention"], r["model_alias"], r["sample_id"]))
    config = {
        "run_id": args.run_id,
        "created_at": base.utc_now(),
        "purpose": "PAC v2.1 B/C aggressive improve pilot/full plan. Baseline is read-only and not rerun.",
        "samples_per_model_requested": args.samples_per_model,
        "models": models,
        "subsets": subsets,
        "baseline_root": str(baseline_root),
        "source_paths": {key: str(SOURCE_SUBSETS[key]["source_path"]) for key in subsets},
        "output_dir": str(out_dir),
        "semantic_boundaries": [
            "source sample_id is reused",
            "gold answer is unchanged",
            "target answer values are not inserted into the prompt",
            "B source-priority uses only the SRC field already present in context",
            "C JSON-only uses only query aliases already present in the question",
            "source PAC/data files are not modified",
            "baseline raw files are not modified",
        ],
        "interventions": {
            "source_priority_slots": (
                "PAC-B: lock query keys and profile slots; require SRC=route-ledger before standard filters; "
                "final answer only."
            ),
            "prompt_only_slots": (
                "PAC-B: prompt-only query-key and profile-slot checklist; no SRC priority or extra data rule; "
                "final answer only."
            ),
            "json_only_binding": (
                "PAC-C: one-line ANSWER_JSON only; require SRC=capacity-ledger and exact alias/entity/field/batch filters."
            ),
        },
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
    write_readme(out_dir, config, plan_rows, coverage_rows, missing_rows)
    write_runner_hint(out_dir, plan_rows)
    print_report(plan_rows, coverage_rows, missing_rows, out_dir, dry_run=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build B/C aggressive PAC improve plans without modifying source data or baseline results."
    )
    parser.add_argument("--run-id", default="pac_v21_improve_aggressive_pilot")
    parser.add_argument("--samples-per-model", type=int, default=12)
    parser.add_argument("--models", default=",".join(base.DEFAULT_MODELS))
    parser.add_argument("--subsets", default="B,C")
    parser.add_argument(
        "--b-intervention",
        choices=["source_priority_slots", "prompt_only_slots"],
        default="source_priority_slots",
    )
    parser.add_argument(
        "--baseline-root",
        default=str(ROOT / "results" / "raw" / "pac_v21_queue" / "pac_v21_full_queue"),
    )
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def intervention_for_subset(subset_key: str, args: argparse.Namespace) -> str:
    if subset_key == "B":
        return args.b_intervention
    return SOURCE_SUBSETS[subset_key]["intervention"]


def load_aggressive_baselines(
    baseline_root: Path,
    models: list[str],
    subsets: list[str],
) -> dict[str, dict[str, dict[str, dict[str, Any]]]]:
    loaded: dict[str, dict[str, dict[str, dict[str, Any]]]] = defaultdict(dict)
    for subset_key in subsets:
        subset_dir = baseline_root / SOURCE_SUBSETS[subset_key]["subset_dir"]
        for model in models:
            path = subset_dir / f"{model}.jsonl"
            loaded[subset_key][model] = base.load_baseline_file(path) if path.exists() else {}
    return loaded


def build_aggressive_prompt(sample: dict[str, Any], intervention: str) -> str:
    prompt = str(sample.get("prompt") or "")
    parts = base.split_prompt(prompt)
    if not parts:
        return prompt
    task, context, question = parts["task"], parts["context"], parts["question"]
    if intervention == "source_priority_slots":
        return base.compose_prompt(
            task=source_priority_task(task, sample),
            context=context,
            question=source_priority_question(question, sample),
        )
    if intervention == "prompt_only_slots":
        return base.compose_prompt(
            task=prompt_only_slots_task(task, sample),
            context=context,
            question=prompt_only_slots_question(question, sample),
        )
    if intervention == "json_only_binding":
        return base.compose_prompt(
            task=json_only_task(task, sample),
            context=context,
            question=json_only_question(question, sample),
        )
    raise ValueError(intervention)


def source_priority_task(original_task: str, sample: dict[str, Any]) -> str:
    fields = [str(item) for item in sample.get("target_fields") or base.split_answer(str(sample.get("target_attribute") or ""))]
    slot_lines = "\n".join(f"- SLOT {idx}: F={field}" for idx, field in enumerate(fields, start=1))
    return (
        f"{original_task}\n\n"
        "Aggressive intervention: Source-priority slot lock.\n"
        "Use this locked query card before scanning evidence. It restates only non-answer keys already "
        "present in the task/context.\n"
        f"- A={sample.get('target_alias')}\n"
        f"- resolved active E={sample.get('target_entity')}\n"
        f"- Q={sample.get('target_qualifier')}\n"
        f"- PID={sample.get('profile_id')}\n"
        f"- cutoff T<{sample.get('cutoff_time') or '23:50'}\n"
        f"{slot_lines}\n\n"
        "Hard source rule for this route-triplet task: after the keys above match, accept DOC records "
        "with SRC=route-ledger. Reject SRC=decoy-ledger records even if they look newer, valid, approved, "
        "primary, or near-matching. Still reject inactive aliases, wrong entity, wrong field, wrong batch, "
        "after-cutoff, pending-review, secondary-channel, draft, revoked, and expired records.\n"
        "Think silently. Do not list evidence or candidates."
    )


def source_priority_question(question: str, sample: dict[str, Any]) -> str:
    fields = [str(item) for item in sample.get("target_fields") or base.split_answer(str(sample.get("target_attribute") or ""))]
    slots = ", ".join(f"SLOT{idx}={field}" for idx, field in enumerate(fields, start=1))
    return (
        f"{question}\n\n"
        f"Internal slot order: {slots}. "
        "Return exactly one line and nothing else:\n"
        "FINAL_ANSWER: V1|V2|V3"
    )


def prompt_only_slots_task(original_task: str, sample: dict[str, Any]) -> str:
    fields = [str(item) for item in sample.get("target_fields") or base.split_answer(str(sample.get("target_attribute") or ""))]
    slot_lines = "\n".join(f"- SLOT {idx}: F={field}" for idx, field in enumerate(fields, start=1))
    return (
        f"{original_task}\n\n"
        "Prompt-only intervention: Query-card slot checklist.\n"
        "This is only a reasoning/format aid. Do not add any new filtering rule beyond the original task. "
        "In particular, do not use SRC as a shortcut, source-priority signal, or tie breaker.\n"
        "Silently follow this order:\n"
        "1. Resolve the requested alias using exactly AS=active.\n"
        "2. Use only the requested approved PID and its approved STEP order.\n"
        "3. For each slot, select the latest memo strictly before cutoff that simultaneously matches "
        "E, F, Q, S=valid, REVIEW=approved, and CHANNEL=primary.\n"
        "4. Reject near entity, near batch, wrong field, wrong profile, inactive alias, pending-review, "
        "secondary-channel, draft, revoked, expired, and after-cutoff records.\n\n"
        "Locked non-answer query card:\n"
        f"- A={sample.get('target_alias')}\n"
        f"- Q={sample.get('target_qualifier')}\n"
        f"- PID={sample.get('profile_id')}\n"
        f"- cutoff T<{sample.get('cutoff_time') or '23:50'}\n"
        f"{slot_lines}\n\n"
        "Think silently. Do not list evidence or candidates."
    )


def prompt_only_slots_question(question: str, sample: dict[str, Any]) -> str:
    fields = [str(item) for item in sample.get("target_fields") or base.split_answer(str(sample.get("target_attribute") or ""))]
    slots = ", ".join(f"SLOT{idx}={field}" for idx, field in enumerate(fields, start=1))
    return (
        f"{question}\n\n"
        f"Use the internal slot order: {slots}. "
        "Return exactly one line and nothing else:\n"
        "FINAL_ANSWER: V1|V2|V3"
    )


def json_only_task(original_task: str, sample: dict[str, Any]) -> str:
    aliases = [str(item) for item in sample.get("query_aliases") or []]
    alias_lines = "\n".join(f"- {alias}" for alias in aliases)
    return (
        f"{original_task}\n\n"
        "Aggressive intervention: JSON-only binding lock.\n"
        "Think silently. Do not write a table, explanation, bullets, markdown, or derivation.\n"
        "For each requested alias independently: resolve the active alias to its exact entity, then select "
        "the DOC whose PID/profile is approved, F equals the requested field, Q equals the requested batch, "
        "S=valid, REVIEW=approved, CHANNEL=primary, T is strictly before cutoff, and SRC=capacity-ledger. "
        "Reject SRC=capacity-decoy and all near entity/batch/field aliases.\n"
        "Requested alias order:\n"
        f"{alias_lines}"
    )


def json_only_question(question: str, sample: dict[str, Any]) -> str:
    aliases = [str(item) for item in sample.get("query_aliases") or []]
    shape = "{" + ", ".join(f'"{alias}":""' for alias in aliases) + "}"
    return (
        f"{question}\n\n"
        "Return exactly one single line and nothing else. No prose before or after.\n"
        f"ANSWER_JSON: {shape}"
    )


def design_note_for(intervention: str) -> str:
    if intervention == "source_priority_slots":
        return "Uses non-answer query-key locks plus explicit route-ledger over decoy-ledger source priority."
    if intervention == "prompt_only_slots":
        return "Uses non-answer query-key locks and slot checklist only; explicitly forbids SRC priority shortcuts."
    if intervention == "json_only_binding":
        return "Suppresses table reasoning; forces one-line JSON and capacity-ledger over capacity-decoy source priority."
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
    coverage_rows: list[dict[str, Any]],
    missing_rows: list[dict[str, Any]],
) -> None:
    counts = Counter((row["pac_subset"], row["intervention"], row["model_alias"]) for row in plan_rows)
    lines = [
        "# PAC v2.1 Improve Aggressive B/C Plan",
        "",
        "This directory contains derived prompts for a B/C aggressive improve pilot or full run.",
        "Original PAC data and baseline raw results were read only; they were not modified.",
        "",
        f"- run_id: `{config['run_id']}`",
        f"- created_at: `{config['created_at']}`",
        f"- planned calls: `{len(plan_rows)}`",
        f"- requested samples per model/intervention: `{config['samples_per_model_requested']}`",
        "",
        "## Counts",
        "",
        "| subset | intervention | model | selected |",
        "|---|---|---|---:|",
    ]
    for (subset, intervention, model), count in sorted(counts.items()):
        lines.append(f"| {subset} | {intervention} | {model} | {count} |")
    lines.extend(["", "## Missing Baseline Notes", ""])
    if missing_rows:
        for row in missing_rows:
            lines.append(f"- {row['pac_subset']} / {row['intervention']} / {row['model_alias']}: {row['reason']}")
    else:
        lines.append("- None.")
    out_dir.joinpath("README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_runner_hint(out_dir: Path, plan_rows: list[dict[str, Any]]) -> None:
    counts = Counter(row["intervention"] for row in plan_rows)
    text = (
        "# Runner Hint\n\n"
        "Run this pilot first. The runner resumes from successful rows in `improve_raw_results.jsonl`.\n\n"
        "```cmd\n"
        "cd /d C:\\Users\\GET-DATA402\\Documents\\Codex\\2026-06-26\\files-mentioned-by-the-user-lmaf\\lmaf_experiments\n"
        "set SILICONFLOW_API_KEYS=sk-1,sk-2\n"
        f"python scripts\\run_pac_improve_queue.py --run-id {out_dir.name} --slots-per-key 1 --per-key-delay-sec 20 --queue-max-attempts 4\n"
        f"python scripts\\analyze_pac_improve_pilot_gate.py --run-id {out_dir.name}\n"
        "```\n\n"
        "Planned calls by intervention:\n\n"
        + "\n".join(f"- {name}: {count}" for name, count in sorted(counts.items()))
        + "\n"
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
    print(f"[{label}] aggressive output dir: {out_dir}")
    print(f"[{label}] planned calls: {len(plan_rows)}")
    for row in coverage_rows:
        print(
            "[COVERAGE] "
            f"{row['pac_subset']} {row['intervention']} {row['model_alias']} "
            f"requested={row['requested']} available={row['baseline_available']} "
            f"selected={row['selected']} shortfall={row['shortfall']}"
        )
    for row in missing_rows:
        print(f"[MISSING] {row['pac_subset']} {row['intervention']} {row['model_alias']}: {row['reason']}")


if __name__ == "__main__":
    main()
