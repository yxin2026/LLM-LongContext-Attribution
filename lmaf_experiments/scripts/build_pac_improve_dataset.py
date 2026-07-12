from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]

SOURCE_SUBSETS = {
    "A": {
        "subset_dir": "PAC-A_position",
        "source_path": ROOT / "PAC" / "data" / "PAC-A_position" / "samples.jsonl",
        "intervention": "segment_anchor",
        "pac_subset": "A",
    },
    "B": {
        "subset_dir": "PAC-B_interference",
        "source_path": ROOT / "PAC" / "data" / "PAC-B_interference" / "samples.jsonl",
        "intervention": "evidence_first",
        "pac_subset": "B",
    },
    "C": {
        "subset_dir": "PAC-C_binding_capacity",
        "source_path": ROOT / "PAC" / "data" / "PAC-C_binding_capacity" / "samples.jsonl",
        "intervention": "binding_table",
        "pac_subset": "C",
    },
}

DEFAULT_MODELS = ["qwen35_9b", "qwen35_27b", "qwen35_122b_a10b"]

MODEL_TO_API = {
    "qwen35_9b": "Qwen/Qwen3.5-9B",
    "qwen35_27b": "Qwen/Qwen3.5-27B",
    "qwen35_35b_a3b": "Qwen/Qwen3.5-35B-A3B",
    "qwen35_122b_a10b": "Qwen/Qwen3.5-122B-A10B",
}

INTERVENTION_MAX_TOKENS = {
    "segment_anchor": 256,
    "evidence_first": 512,
    "binding_table": 1024,
}


def main() -> None:
    args = parse_args()
    models = parse_csv(args.models)
    baseline_root = Path(args.baseline_root)
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "results" / "improve" / args.run_id
    guard_output_dir(out_dir)

    source_by_subset = {
        key: load_source_samples(config["source_path"])
        for key, config in SOURCE_SUBSETS.items()
    }
    baseline_by_subset_model = load_all_baselines(baseline_root, models)

    plan_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []

    for subset_key, config in SOURCE_SUBSETS.items():
        source_samples = source_by_subset[subset_key]
        available_by_model = {
            model: set(baseline_by_subset_model[subset_key].get(model, {}).keys())
            for model in models
        }
        common_ids = set.intersection(
            *[ids for ids in available_by_model.values() if ids]
        ) if any(available_by_model.values()) else set()
        common_selected = stratified_select(
            sorted(common_ids),
            source_samples,
            args.samples_per_model,
            stratum_fn_for(subset_key),
            seed=args.seed + ord(subset_key),
            prefer_fn=prefer_fn_for(subset_key),
        )
        common_selected_set = set(common_selected)

        for model in models:
            baseline_rows = baseline_by_subset_model[subset_key].get(model, {})
            available_ids = sorted(set(baseline_rows).intersection(source_samples))
            selected = list(common_selected)
            if len(selected) < args.samples_per_model:
                selected_set = set(selected)
                selected.extend(
                    stratified_select(
                        [sample_id for sample_id in available_ids if sample_id not in selected_set],
                        source_samples,
                        args.samples_per_model - len(selected),
                        stratum_fn_for(subset_key),
                        seed=args.seed + len(model) * 97 + ord(subset_key),
                        prefer_fn=prefer_fn_for(subset_key),
                    )
                )
            selected = [sample_id for sample_id in selected if sample_id in baseline_rows]

            if not selected:
                missing_rows.append(
                    {
                        "pac_subset": subset_key,
                        "intervention": config["intervention"],
                        "model_alias": model,
                        "reason": "no_paired_baseline_for_model_subset",
                        "baseline_dir": str(baseline_root / config["subset_dir"]),
                    }
                )

            coverage_rows.append(
                {
                    "pac_subset": subset_key,
                    "intervention": config["intervention"],
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
                intervention = config["intervention"]
                intervention_prompt = build_intervention_prompt(sample, intervention)
                plan_rows.append(
                    build_plan_row(
                        sample=sample,
                        baseline=baseline,
                        subset_key=subset_key,
                        intervention=intervention,
                        model_alias=model,
                        intervention_prompt=intervention_prompt,
                        common_sample=sample_id in common_selected_set,
                    )
                )

    plan_rows.sort(key=lambda r: (r["pac_subset"], r["intervention"], r["model_alias"], r["sample_id"]))
    config = {
        "run_id": args.run_id,
        "created_at": utc_now(),
        "purpose": "PAC v2.1 improve intervention dataset. Baseline is read-only and not rerun.",
        "samples_per_model_requested": args.samples_per_model,
        "models": models,
        "baseline_root": str(baseline_root),
        "source_paths": {key: str(value["source_path"]) for key, value in SOURCE_SUBSETS.items()},
        "output_dir": str(out_dir),
        "semantic_invariants": [
            "source sample_id is reused",
            "gold answer is unchanged",
            "target facts are unchanged",
            "decoy values and baseline metadata are copied read-only",
            "source PAC/data files are not modified",
            "baseline raw files are not modified",
        ],
        "interventions": {
            "segment_anchor": "PAC-A: add segment anchors and boundary markers around target active evidence records in a derived prompt.",
            "evidence_first": "PAC-B: require evidence-first strict filtering, then final answer line only is scored.",
            "binding_table": "PAC-C: require candidate binding table reconstruction, then ANSWER_JSON line is scored.",
        },
    }

    if args.dry_run:
        print_report(plan_rows, coverage_rows, missing_rows, out_dir, dry_run=True)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "run_config.json", config)
    write_jsonl(out_dir / "improve_plan.jsonl", plan_rows)
    write_jsonl(out_dir / "intervention_prompts.jsonl", intervention_prompt_rows(plan_rows))
    write_csv(out_dir / "coverage_report.csv", coverage_rows)
    write_csv(out_dir / "missing_baseline.csv", missing_rows)
    write_readme(out_dir, config, plan_rows, coverage_rows, missing_rows)
    write_runner_hint(out_dir, plan_rows)
    print_report(plan_rows, coverage_rows, missing_rows, out_dir, dry_run=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a read-only PAC v2.1 Improve intervention dataset. This script does not call APIs "
            "and does not modify PAC/data or baseline raw results."
        )
    )
    parser.add_argument("--run-id", default="pac_v21_improve_540")
    parser.add_argument("--samples-per-model", type=int, default=60)
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument(
        "--baseline-root",
        default=str(ROOT / "results" / "raw" / "pac_v21_queue" / "pac_v21_full_queue"),
    )
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def guard_output_dir(out_dir: Path) -> None:
    resolved = out_dir.resolve()
    improve_root = (ROOT / "results" / "improve").resolve()
    if improve_root not in [resolved, *resolved.parents]:
        raise SystemExit(f"Refusing to write outside results/improve: {resolved}")


def load_source_samples(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"Missing source sample file: {path}")
    rows = {}
    for row in read_jsonl(path):
        sample_id = row.get("sample_id")
        if not sample_id:
            raise SystemExit(f"Source row without sample_id in {path}")
        rows[sample_id] = row
    return rows


def load_all_baselines(
    baseline_root: Path, models: list[str]
) -> dict[str, dict[str, dict[str, dict[str, Any]]]]:
    loaded: dict[str, dict[str, dict[str, dict[str, Any]]]] = defaultdict(dict)
    for subset_key, config in SOURCE_SUBSETS.items():
        subset_dir = baseline_root / config["subset_dir"]
        for model in models:
            path = subset_dir / f"{model}.jsonl"
            loaded[subset_key][model] = load_baseline_file(path) if path.exists() else {}
    return loaded


def load_baseline_file(path: Path) -> dict[str, dict[str, Any]]:
    rows_by_id: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        sample_id = row.get("sample_id")
        if not sample_id:
            continue
        if row.get("score") is None:
            continue
        # Resume runs can append duplicate sample_ids; the last valid row is the freshest one.
        rows_by_id[sample_id] = row
    return rows_by_id


def stratum_fn_for(subset_key: str) -> Callable[[dict[str, Any]], str]:
    if subset_key == "A":
        return lambda sample: f"pos={sample.get('position_percent') or sample.get('position_bin')}"
    if subset_key == "B":
        return lambda sample: f"decoy={sample.get('decoy_count')}"
    if subset_key == "C":
        return lambda sample: f"K={sample.get('binding_k')}_Q={sample.get('query_count')}"
    raise ValueError(subset_key)


def prefer_fn_for(subset_key: str) -> Callable[[dict[str, Any]], int] | None:
    if subset_key == "B":
        preferred = {64: 3, 128: 4, 192: 5, 32: 2, 16: 1, 0: 0}
        return lambda sample: preferred.get(int(sample.get("decoy_count") or 0), 0)
    if subset_key == "C":
        return lambda sample: int(sample.get("binding_k") or 0) * 10 + int(sample.get("query_count") or 0)
    return None


def stratified_select(
    sample_ids: list[str],
    source_samples: dict[str, dict[str, Any]],
    target: int,
    stratum_fn: Callable[[dict[str, Any]], str],
    seed: int,
    prefer_fn: Callable[[dict[str, Any]], int] | None = None,
) -> list[str]:
    rng = random.Random(seed)
    buckets: dict[str, list[str]] = defaultdict(list)
    for sample_id in sample_ids:
        if sample_id in source_samples:
            buckets[stratum_fn(source_samples[sample_id])].append(sample_id)
    for ids in buckets.values():
        if prefer_fn:
            ids.sort(key=lambda sid: (-prefer_fn(source_samples[sid]), sid))
        else:
            ids.sort()
            rng.shuffle(ids)

    selected: list[str] = []
    strata = sorted(buckets)
    while len(selected) < target and any(buckets.values()):
        for stratum in strata:
            if len(selected) >= target:
                break
            if buckets[stratum]:
                selected.append(buckets[stratum].pop(0))
    return selected


def build_plan_row(
    sample: dict[str, Any],
    baseline: dict[str, Any],
    subset_key: str,
    intervention: str,
    model_alias: str,
    intervention_prompt: str,
    common_sample: bool,
) -> dict[str, Any]:
    source_prompt = sample.get("prompt", "")
    baseline_prediction = str(baseline.get("prediction") or "")
    baseline_error_type = str(baseline.get("error_type") or "unknown")
    answer = str(sample.get("answer") or "")
    expected_answers = [str(item) for item in sample.get("expected_answers") or split_answer(answer)]
    return {
        "sample_id": sample["sample_id"],
        "pac_subset": subset_key,
        "formal_subset": sample.get("formal_subset"),
        "intervention": intervention,
        "model_alias": model_alias,
        "model_id": MODEL_TO_API.get(model_alias, baseline.get("api_model", model_alias)),
        "prompt_version": f"{intervention}_v1",
        "common_across_available_models": bool(common_sample),
        "source_prompt_sha256": sha256_text(source_prompt),
        "intervention_prompt_sha256": sha256_text(intervention_prompt),
        "source_prompt_chars": len(source_prompt),
        "intervention_prompt_chars": len(intervention_prompt),
        "prompt_char_delta": len(intervention_prompt) - len(source_prompt),
        "source_length_tokens_actual": sample.get("length_tokens_actual"),
        "position": sample.get("position_percent") or sample.get("position_bin"),
        "position_actual": sample.get("position_percent_actual"),
        "decoy_count": sample.get("decoy_count"),
        "decoy_values": sample.get("decoy_values") or sample.get("distractor_answers") or [],
        "binding_load_K": sample.get("binding_k"),
        "query_count_Q": sample.get("query_count"),
        "target_alias": sample.get("target_alias"),
        "target_entity": sample.get("target_entity"),
        "target_attribute": sample.get("target_attribute") or sample.get("target_field"),
        "target_qualifier": sample.get("target_qualifier"),
        "profile_id": sample.get("profile_id"),
        "query_aliases": sample.get("query_aliases"),
        "gold_answer": answer,
        "gold_expected_answers": expected_answers,
        "gold_answer_json": build_gold_answer_json(sample, expected_answers),
        "baseline_answer": baseline_prediction,
        "correct_baseline": int(float(baseline.get("score") or 0.0) > 0.0),
        "field_accuracy_baseline": float(baseline.get("field_accuracy") or 0.0),
        "decoy_capture_baseline": int(baseline.get("decoy_captured") or baseline_error_type == "decoy_value_capture"),
        "binding_error_baseline": int(is_binding_error(baseline_error_type)),
        "baseline_error_type": baseline_error_type,
        "baseline_latency_sec": baseline.get("latency_sec"),
        "baseline_timestamp": baseline.get("timestamp"),
        "max_tokens_recommended": INTERVENTION_MAX_TOKENS[intervention],
        "temperature": 0.0,
        "top_p": 1.0,
        "intervention_prompt": intervention_prompt,
        "source_invariants_ok": check_invariants(sample, intervention_prompt),
    }


def build_intervention_prompt(sample: dict[str, Any], intervention: str) -> str:
    prompt = str(sample.get("prompt") or "")
    parts = split_prompt(prompt)
    if not parts:
        return prompt
    task, context, question = parts["task"], parts["context"], parts["question"]
    if intervention == "segment_anchor":
        return compose_prompt(
            task=segment_anchor_task(task),
            context=segment_anchor_context(context, sample),
            question=segment_anchor_question(question),
        )
    if intervention == "evidence_first":
        return compose_prompt(
            task=evidence_first_task(task),
            context=context,
            question=evidence_first_question(question),
        )
    if intervention == "binding_table":
        return compose_prompt(
            task=binding_table_task(task),
            context=context,
            question=binding_table_question(question, sample),
        )
    raise ValueError(intervention)


def split_prompt(prompt: str) -> dict[str, str] | None:
    pattern = re.compile(
        r"\[Task\]\s*(?P<task>.*?)\s*\[Long Context\]\s*(?P<context>.*?)\s*\[Question\]\s*(?P<question>.*?)\s*\[Answer\]\s*$",
        re.S,
    )
    match = pattern.search(prompt)
    if not match:
        return None
    return {
        "task": match.group("task").strip(),
        "context": match.group("context").strip(),
        "question": match.group("question").strip(),
    }


def compose_prompt(task: str, context: str, question: str) -> str:
    return f"[Task]\n\n{task}\n\n[Long Context]\n\n{context}\n\n[Question]\n{question}\n[Answer]"


def segment_anchor_task(original_task: str) -> str:
    return (
        f"{original_task}\n\n"
        "Intervention: Segment Anchor. Treat explicit segment markers as navigation anchors. "
        "Active evidence records may be enclosed by boundary markers; use those boundaries only to "
        "separate record spans, not to change any fact. Final output must still be only V1|V2|V3."
    )


def segment_anchor_context(context: str, sample: dict[str, Any]) -> str:
    anchored = context
    for fact in sample.get("target_facts") or [sample.get("target_fact")]:
        if not fact:
            continue
        marker = (
            "[[ACTIVE_RECORD_BOUNDARY_START]]\n"
            f"{fact}\n"
            "[[ACTIVE_RECORD_BOUNDARY_END]]"
        )
        anchored = anchored.replace(str(fact), marker, 1)
    return add_segment_markers(anchored, segment_count=8)


def add_segment_markers(context: str, segment_count: int = 8) -> str:
    lines = context.splitlines()
    if not lines:
        return context
    chunk = max(1, len(lines) // segment_count)
    output: list[str] = []
    segment = 1
    for idx, line in enumerate(lines):
        if idx % chunk == 0:
            if idx != 0:
                output.append(f"[[SEGMENT_{segment:02d}_END]]")
                segment += 1
            output.append(f"[[SEGMENT_{segment:02d}_START]]")
        output.append(line)
    output.append(f"[[SEGMENT_{segment:02d}_END]]")
    return "\n".join(output)


def segment_anchor_question(question: str) -> str:
    return (
        f"{question}\n\n"
        "Use the segment anchors to locate the relevant active evidence. "
        "Return only the final triplet in the required order."
    )


def evidence_first_task(original_task: str) -> str:
    return (
        f"{original_task}\n\n"
        "Intervention: Evidence-first strict filter. Before giving the final answer, identify the evidence "
        "records that simultaneously satisfy: S=valid, REVIEW=approved, CHANNEL=primary, active alias entity "
        "matches exactly, requested batch matches exactly, requested profile field matches exactly, and T is "
        "strictly before the cutoff. Then derive the final answer only from those evidence records."
    )


def evidence_first_question(question: str) -> str:
    return (
        f"{question}\n\n"
        "First write a compact EVIDENCE section listing the selected DOC ids and V values. "
        "Then write a final line exactly as FINAL_ANSWER: V1|V2|V3. "
        "Only the FINAL_ANSWER line will be scored."
    )


def binding_table_task(original_task: str) -> str:
    return (
        f"{original_task}\n\n"
        "Intervention: Binding Table. Reconstruct a compact candidate table before answering. "
        "The table must include columns: project_or_alias | attribute | value | status | exact_project_match | "
        "exact_attribute_match | valid_for_answer. The final answer must be derived only from rows where "
        "valid_for_answer=yes."
    )


def binding_table_question(question: str, sample: dict[str, Any]) -> str:
    aliases = sample.get("query_aliases") or []
    alias_hint = ", ".join(aliases) if aliases else "the requested aliases"
    return (
        f"{question}\n\n"
        f"Build the binding table for {alias_hint}. Then write a final line exactly as "
        "ANSWER_JSON: {\"alias\": \"value\", ...}. "
        "Only the ANSWER_JSON line will be scored."
    )


def build_gold_answer_json(sample: dict[str, Any], expected_answers: list[str]) -> dict[str, str] | None:
    aliases = sample.get("query_aliases")
    if not aliases:
        return None
    return {str(alias): expected_answers[idx] for idx, alias in enumerate(aliases) if idx < len(expected_answers)}


def check_invariants(sample: dict[str, Any], intervention_prompt: str) -> bool:
    answer = str(sample.get("answer") or "")
    expected = [str(item) for item in sample.get("expected_answers") or split_answer(answer)]
    # The full joined answer often does not appear contiguously in PAC prompts;
    # the individual evidence values should remain present.
    return all(item in intervention_prompt for item in expected)


def is_binding_error(error_type: str) -> bool:
    return error_type in {"partial_triplet", "near_miss_value", "decoy_value_capture"}


def split_answer(answer: str) -> list[str]:
    return [part.strip() for part in answer.split("|") if part.strip()]


def intervention_prompt_rows(plan_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keep = [
        "sample_id",
        "pac_subset",
        "intervention",
        "model_alias",
        "model_id",
        "prompt_version",
        "gold_answer",
        "gold_answer_json",
        "max_tokens_recommended",
        "temperature",
        "top_p",
        "intervention_prompt",
    ]
    return [{key: row.get(key) for key in keep} for row in plan_rows]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
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
        "# PAC v2.1 Improve Dataset",
        "",
        "This directory contains derived intervention prompts for a paired PAC improve experiment.",
        "The original PAC data and baseline raw results were read only; they were not modified.",
        "",
        f"- run_id: `{config['run_id']}`",
        f"- created_at: `{config['created_at']}`",
        f"- planned intervention calls: `{len(plan_rows)}`",
        f"- requested samples per model/intervention: `{config['samples_per_model_requested']}`",
        "",
        "## Counts",
        "",
        "| subset | intervention | model | selected |",
        "|---|---|---|---:|",
    ]
    for (subset, intervention, model), count in sorted(counts.items()):
        lines.append(f"| {subset} | {intervention} | {model} | {count} |")
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `run_config.json`: immutable build configuration.",
            "- `improve_plan.jsonl`: full paired plan with baseline metrics and derived prompts.",
            "- `intervention_prompts.jsonl`: lean prompt file for an API runner.",
            "- `coverage_report.csv`: requested/available/selected counts.",
            "- `missing_baseline.csv`: explicit missing paired-baseline cases.",
            "- `runner_hint.md`: suggested commands for `scripts/run_pac_improve_queue.py`.",
            "",
            "## Missing Baseline Notes",
            "",
        ]
    )
    if missing_rows:
        for row in missing_rows:
            lines.append(
                f"- {row['pac_subset']} / {row['intervention']} / {row['model_alias']}: {row['reason']}"
            )
    else:
        lines.append("- None.")
    out_dir.joinpath("README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_runner_hint(out_dir: Path, plan_rows: list[dict[str, Any]]) -> None:
    counts = Counter(row["intervention"] for row in plan_rows)
    text = (
        "# Runner Hint\n\n"
        "This build step intentionally did not call the SiliconFlow API. Use "
        "`scripts/run_pac_improve_queue.py` from the project root to execute the queue. "
        "The runner reads `improve_plan.jsonl`, resumes from successful rows already present "
        "in `improve_raw_results.jsonl`, and writes only into this improve output directory.\n\n"
        "Example:\n\n"
        "```powershell\n"
        "$env:SILICONFLOW_API_KEYS=\"sk-1,sk-2\"\n"
        "python scripts\\run_pac_improve_queue.py --run-id pac_v21_improve_540 --slots-per-key 1 --per-key-delay-sec 20 --queue-max-attempts 4\n"
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
    print(f"[{label}] improve output dir: {out_dir}")
    print(f"[{label}] planned calls: {len(plan_rows)}")
    for row in coverage_rows:
        print(
            "[COVERAGE] "
            f"{row['pac_subset']} {row['intervention']} {row['model_alias']} "
            f"requested={row['requested']} available={row['baseline_available']} "
            f"selected={row['selected']} shortfall={row['shortfall']}"
        )
    if missing_rows:
        for row in missing_rows:
            print(
                "[MISSING] "
                f"{row['pac_subset']} {row['intervention']} {row['model_alias']}: {row['reason']}"
            )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    main()
