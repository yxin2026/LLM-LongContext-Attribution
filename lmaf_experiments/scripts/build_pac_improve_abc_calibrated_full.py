from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import build_pac_c_strategy_probe as c_probe
import build_pac_improve_aggressive_dataset as aggressive
import build_pac_improve_dataset as base


ROOT = Path(__file__).resolve().parents[1]

CALIBRATED_SUBSETS = {
    "A": {
        "subset_dir": "PAC-A_position",
        "source_path": ROOT / "PAC" / "data" / "PAC-A_position" / "samples.jsonl",
        "intervention": "segment_anchor",
        "design_note": "Segment Anchor calibrated positive on PAC-A position/structure salience.",
    },
    "B": {
        "subset_dir": "PAC-B_interference",
        "source_path": ROOT / "PAC" / "data" / "PAC-B_interference" / "samples.jsonl",
        "intervention": "prompt_only_slots",
        "design_note": "Prompt-only query-card and slot checklist calibrated positive on PAC-B; no SRC shortcut.",
    },
    "C": {
        "subset_dir": "PAC-C_binding_capacity",
        "source_path": ROOT / "PAC" / "data" / "PAC-C_binding_capacity" / "samples.jsonl",
        "intervention": "c_compact_bind_json",
        "design_note": "Compact alias=value binding lines plus ANSWER_JSON calibrated positive on PAC-C; no SRC shortcut.",
    },
}

CALIBRATED_MAX_TOKENS = {
    "segment_anchor": 256,
    "prompt_only_slots": 128,
    "c_compact_bind_json": 512,
}


def main() -> None:
    args = parse_args()
    subsets = base.parse_csv(args.subsets)
    bad_subsets = sorted(set(subsets) - set(CALIBRATED_SUBSETS))
    if bad_subsets:
        raise SystemExit(f"Unsupported calibrated subsets: {bad_subsets}. Use A,B,C.")

    models = base.parse_csv(args.models)
    baseline_root = Path(args.baseline_root)
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "results" / "improve" / args.run_id
    base.guard_output_dir(out_dir)
    base.INTERVENTION_MAX_TOKENS.update(CALIBRATED_MAX_TOKENS)

    source_by_subset = {
        subset: base.load_source_samples(CALIBRATED_SUBSETS[subset]["source_path"])
        for subset in subsets
    }
    baseline_by_subset_model = load_baselines(baseline_root, models, subsets)

    plan_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []

    for subset in subsets:
        source_samples = source_by_subset[subset]
        available_by_model = {
            model: set(baseline_by_subset_model[subset].get(model, {})).intersection(source_samples)
            for model in models
        }
        nonempty_sets = [ids for ids in available_by_model.values() if ids]
        common_ids = set.intersection(*nonempty_sets) if nonempty_sets else set()
        common_selected = base.stratified_select(
            sorted(common_ids),
            source_samples,
            args.samples_per_model,
            base.stratum_fn_for(subset),
            seed=args.seed + ord(subset),
            prefer_fn=base.prefer_fn_for(subset),
        )
        common_selected_set = set(common_selected)

        for model in models:
            baseline_rows = baseline_by_subset_model[subset].get(model, {})
            available_ids = sorted(set(baseline_rows).intersection(source_samples))
            selected = list(common_selected)
            if len(selected) < args.samples_per_model:
                selected_set = set(selected)
                selected.extend(
                    base.stratified_select(
                        [sample_id for sample_id in available_ids if sample_id not in selected_set],
                        source_samples,
                        args.samples_per_model - len(selected),
                        base.stratum_fn_for(subset),
                        seed=args.seed + ord(subset) + len(model) * 109,
                        prefer_fn=base.prefer_fn_for(subset),
                    )
                )
            selected = [sample_id for sample_id in selected if sample_id in baseline_rows]
            paired_selected_available = len(selected)
            drop_reason = ""
            if selected and len(selected) < args.samples_per_model and not args.allow_partial_cells:
                drop_reason = "insufficient_paired_baseline_for_full_cell"
                selected = []

            intervention = CALIBRATED_SUBSETS[subset]["intervention"]
            coverage_rows.append(
                {
                    "pac_subset": subset,
                    "intervention": intervention,
                    "model_alias": model,
                    "requested": args.samples_per_model,
                    "source_samples": len(source_samples),
                    "baseline_available": len(available_ids),
                    "common_across_available_models": len(common_ids),
                    "common_selected": sum(1 for sample_id in selected if sample_id in common_selected_set),
                    "paired_selected_available": paired_selected_available,
                    "selected": len(selected),
                    "shortfall": max(0, args.samples_per_model - len(selected)),
                }
            )
            if not selected:
                reason = drop_reason or "no_paired_baseline_for_model_subset"
                missing_rows.append(
                    {
                        "pac_subset": subset,
                        "intervention": intervention,
                        "model_alias": model,
                        "reason": reason,
                        "paired_selected_available": paired_selected_available,
                        "requested": args.samples_per_model,
                        "baseline_dir": str(baseline_root / CALIBRATED_SUBSETS[subset]["subset_dir"]),
                    }
                )
                continue

            for sample_id in selected:
                sample = source_samples[sample_id]
                baseline = baseline_rows[sample_id]
                prompt = build_calibrated_prompt(sample, subset, intervention)
                row = base.build_plan_row(
                    sample=sample,
                    baseline=baseline,
                    subset_key=subset,
                    intervention=intervention,
                    model_alias=model,
                    intervention_prompt=prompt,
                    common_sample=sample_id in common_selected_set,
                )
                row["prompt_version"] = f"{intervention}_calibrated_full_v1"
                row["calibrated_full_design_note"] = CALIBRATED_SUBSETS[subset]["design_note"]
                row["calibration_status"] = "selected_from_pilot"
                plan_rows.append(row)

    plan_rows.sort(key=lambda r: (r["pac_subset"], r["intervention"], r["model_alias"], r["sample_id"]))
    config = {
        "run_id": args.run_id,
        "created_at": base.utc_now(),
        "purpose": "PAC v2.1 ABC calibrated full improve test plan. Baseline is read-only and not rerun.",
        "samples_per_model_requested": args.samples_per_model,
        "models": models,
        "subsets": subsets,
        "baseline_root": str(baseline_root),
        "output_dir": str(out_dir),
        "source_paths": {subset: str(CALIBRATED_SUBSETS[subset]["source_path"]) for subset in subsets},
        "calibrated_interventions": {
            subset: {
                "intervention": CALIBRATED_SUBSETS[subset]["intervention"],
                "design_note": CALIBRATED_SUBSETS[subset]["design_note"],
                "max_tokens_recommended": CALIBRATED_MAX_TOKENS[CALIBRATED_SUBSETS[subset]["intervention"]],
            }
            for subset in subsets
        },
        "semantic_boundaries": [
            "source sample_id is reused",
            "gold answer is unchanged",
            "baseline rows are paired by sample_id and model_alias",
            "source PAC/data files are not modified",
            "baseline raw files are not modified",
            "B and C calibrated prompts do not introduce SRC priority shortcuts",
        ],
        "calibration_evidence_dirs": [
            str(ROOT / "results" / "improve" / "pac_v21_improve_540"),
            str(ROOT / "results" / "improve" / "pac_v21_improve_B_promptonly_pilot"),
            str(ROOT / "results" / "improve" / "pac_v21_improve_C_strategy_probe"),
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
    parser = argparse.ArgumentParser(description="Build the PAC ABC calibrated full improve test plan.")
    parser.add_argument("--run-id", default="pac_v21_improve_ABC_calibrated_full")
    parser.add_argument("--samples-per-model", type=int, default=60)
    parser.add_argument("--models", default=",".join(base.DEFAULT_MODELS))
    parser.add_argument("--subsets", default="A,B,C")
    parser.add_argument(
        "--baseline-root",
        default=str(ROOT / "results" / "raw" / "pac_v21_queue" / "pac_v21_full_queue"),
    )
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument(
        "--allow-partial-cells",
        action="store_true",
        help="Include model/subset cells with fewer than --samples-per-model paired baseline rows.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_baselines(
    baseline_root: Path,
    models: list[str],
    subsets: list[str],
) -> dict[str, dict[str, dict[str, dict[str, Any]]]]:
    loaded: dict[str, dict[str, dict[str, dict[str, Any]]]] = defaultdict(dict)
    for subset in subsets:
        subset_dir = baseline_root / CALIBRATED_SUBSETS[subset]["subset_dir"]
        for model in models:
            path = subset_dir / f"{model}.jsonl"
            loaded[subset][model] = base.load_baseline_file(path) if path.exists() else {}
    return loaded


def build_calibrated_prompt(sample: dict[str, Any], subset: str, intervention: str) -> str:
    if subset == "A" and intervention == "segment_anchor":
        return base.build_intervention_prompt(sample, "segment_anchor")
    if subset == "B" and intervention == "prompt_only_slots":
        return aggressive.build_aggressive_prompt(sample, "prompt_only_slots")
    if subset == "C" and intervention == "c_compact_bind_json":
        return c_probe.build_c_strategy_prompt(sample, "c_compact_bind_json")
    raise ValueError(f"Unsupported calibrated prompt: PAC-{subset} / {intervention}")


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
    counts = Counter((row["pac_subset"], row["intervention"], row["model_alias"]) for row in plan_rows)
    lines = [
        "# PAC v2.1 ABC Calibrated Full Improve Plan",
        "",
        "This directory contains the full ABC improve test plan selected from calibration pilots.",
        "Original PAC data and baseline raw results were read only; they were not modified.",
        "",
        f"- run_id: `{config['run_id']}`",
        f"- created_at: `{config['created_at']}`",
        f"- planned calls: `{len(plan_rows)}`",
        f"- requested samples per model/intervention: `{config['samples_per_model_requested']}`",
        "",
        "## Calibrated Interventions",
        "",
    ]
    for subset, payload in config["calibrated_interventions"].items():
        lines.append(f"- PAC-{subset}: `{payload['intervention']}` - {payload['design_note']}")
    lines.extend(
        [
            "",
            "## Counts",
            "",
            "| subset | intervention | model | selected |",
            "|---|---|---|---:|",
        ]
    )
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
        "Run this ABC calibrated full plan with the existing resumable API queue.\n\n"
        "```cmd\n"
        "cd /d C:\\Users\\GET-DATA402\\Documents\\Codex\\2026-06-26\\files-mentioned-by-the-user-lmaf\\lmaf_experiments\n"
        "set SILICONFLOW_API_KEYS=sk-1,sk-2\n"
        f"python scripts\\run_pac_improve_queue.py --run-id {out_dir.name} --slots-per-key 1 --per-key-delay-sec 20 --queue-max-attempts 4\n"
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
    print(f"[{label}] ABC calibrated output dir: {out_dir}")
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
