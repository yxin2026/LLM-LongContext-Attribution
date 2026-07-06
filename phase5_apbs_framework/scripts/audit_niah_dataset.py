from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


REQUIRED = ["sample_id", "length", "position", "answer", "prompt", "question"]


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON at line {i}: {exc}") from exc
    return rows


def extract_key(row: dict) -> str | None:
    question = row.get("question", "")
    match = re.search(r"(?:lookup_key|retrieval key)\s+([A-Za-z0-9_-]+)", question)
    return match.group(1) if match else None


def target_position_pct(row: dict) -> float | None:
    prompt = row.get("prompt", "")
    answer = row.get("answer", "")
    idx = prompt.find(answer)
    if idx < 0 or not prompt:
        return None
    return idx / len(prompt) * 100


def summarize(values: list[float]) -> dict:
    if not values:
        return {"min": None, "p50": None, "max": None}
    values = sorted(values)
    return {
        "min": values[0],
        "p50": values[len(values) // 2],
        "max": values[-1],
    }


def audit(rows: list[dict]) -> tuple[list[str], list[dict]]:
    issues = []
    table = []
    ids = [row.get("sample_id") for row in rows]
    dup_ids = [k for k, v in Counter(ids).items() if k and v > 1]
    if dup_ids:
        issues.append(f"duplicate sample_id count={len(dup_ids)}")

    for idx, row in enumerate(rows):
        missing = [key for key in REQUIRED if key not in row]
        if missing:
            issues.append(f"row {idx} missing fields: {missing}")
            continue
        prompt = row["prompt"]
        answer = row["answer"]
        answer_count = prompt.count(answer)
        key = extract_key(row)
        key_count = prompt.count(key) if key else 0
        decoy_count = prompt.count("DECOY-")
        target_count = prompt.count("TARGET")
        pos_actual = target_position_pct(row)
        declared_pos = float(row["position"])
        pos_error = abs(pos_actual - declared_pos) if pos_actual is not None else None
        if answer_count != 1:
            issues.append(f"{row['sample_id']}: answer_count={answer_count}, expected 1")
        if key and key_count != 2:
            issues.append(f"{row['sample_id']}: key_count={key_count}, expected 2 (context + question)")
        if target_count != 1:
            issues.append(f"{row['sample_id']}: target_count={target_count}, expected 1")
        if pos_error is None or pos_error > 8:
            issues.append(f"{row['sample_id']}: target position error={pos_error}")
        table.append(
            {
                "sample_id": row["sample_id"],
                "length": row["length"],
                "position": row["position"],
                "chars": len(prompt),
                "words_approx": len(prompt.split()),
                "answer_count": answer_count,
                "key_count": key_count,
                "target_count": target_count,
                "decoy_count": decoy_count,
                "target_pos_actual": pos_actual,
                "target_pos_error": pos_error,
                "difficulty": row.get("difficulty", "unknown"),
            }
        )
    return issues, table


def write_markdown(path: Path, dataset: Path, rows: list[dict], issues: list[str], table: list[dict]) -> None:
    by_cell: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for row in table:
        by_cell[(int(row["length"]), int(row["position"]))].append(row)

    lines = [
        "# NIAH Dataset Audit",
        "",
        f"dataset: `{dataset}`",
        f"rows: {len(rows)}",
        f"issues: {len(issues)}",
        "",
        "## Cell Summary",
        "",
        "| length | position | n | words_p50 | decoys_p50 | answer_count_bad | pos_error_p50 | pos_error_max |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for (length, position), items in sorted(by_cell.items()):
        word_stats = summarize([float(x["words_approx"]) for x in items])
        decoy_stats = summarize([float(x["decoy_count"]) for x in items])
        err_stats = summarize([float(x["target_pos_error"]) for x in items if x["target_pos_error"] is not None])
        answer_bad = sum(1 for x in items if x["answer_count"] != 1)
        lines.append(
            f"| {length} | {position} | {len(items)} | {word_stats['p50']:.0f} | "
            f"{decoy_stats['p50']:.0f} | {answer_bad} | {err_stats['p50']:.2f} | {err_stats['max']:.2f} |"
        )

    lines.extend(["", "## Issues", ""])
    if issues:
        for issue in issues[:100]:
            lines.append(f"- {issue}")
        if len(issues) > 100:
            lines.append(f"- ... truncated {len(issues) - 100} more issues")
    else:
        lines.append("No structural issues found.")

    lines.extend(
        [
            "",
            "## Pass Criteria",
            "",
            "- `answer_count_bad` should be 0.",
            "- median decoys should be comfortably above 30 for the 16K hard set.",
            "- median target position error should be small, preferably under 3 percentage points.",
            "- position 50 should not be trivially easier than 10/90 by structure.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_samples(output_dir: Path, rows: list[dict], n: int, seed: int) -> None:
    rng = random.Random(seed)
    picks = rng.sample(rows, min(n, len(rows)))
    sample_dir = output_dir / "audit_samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    for row in picks:
        text = [
            f"sample_id: {row.get('sample_id')}",
            f"length: {row.get('length')}",
            f"position: {row.get('position')}",
            f"answer: {row.get('answer')}",
            f"question: {row.get('question')}",
            "",
            row.get("prompt", ""),
        ]
        (sample_dir / f"{row.get('sample_id', 'sample')}.txt").write_text("\n".join(text), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", default="results/audit")
    parser.add_argument("--sample-count", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260706)
    args = parser.parse_args()

    dataset = Path(args.dataset)
    rows = load_jsonl(dataset)
    issues, table = audit(rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_markdown(output_dir / "audit_report.md", dataset, rows, issues, table)
    write_samples(output_dir, rows, args.sample_count, args.seed)
    print(f"rows={len(rows)} issues={len(issues)}")
    print(f"wrote={output_dir / 'audit_report.md'}")
    print(f"samples={output_dir / 'audit_samples'}")


if __name__ == "__main__":
    main()

