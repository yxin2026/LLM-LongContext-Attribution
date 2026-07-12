from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lmaf.utils.io import write_jsonl


RULER_TASK_MAP = {
    "niah_single_1": "niah",
    "niah_single_2": "niah",
    "niah_single_3": "niah",
    "niah_multikey_1": "niah",
    "niah_multikey_2": "niah",
    "niah_multikey_3": "niah",
    "niah_multivalue": "niah",
    "niah_multiquery": "niah",
    "vt": "variable_tracking",
    "cwe": "common_words_extraction",
    "fwe": "freq_words_extraction",
    "qa_1": "qa_squad",
    "qa_2": "qa_hotpotqa",
}


def main() -> None:
    args = parse_args()
    source = Path(args.source_root)
    if not source.exists():
        raise SystemExit(f"source root not found: {source}")

    rows = []
    for path in find_source_files(source, args.subset):
        task_name = infer_task_name(source, path)
        if args.tasks and task_name not in set(parse_csv(args.tasks)):
            continue
        converted = convert_file(path, task_name, args)
        rows.extend(converted)

    if not rows:
        raise SystemExit("No rows imported. Check --source-root, --subset, and --tasks.")
    write_jsonl(ROOT / args.output, rows)
    print(f"Wrote {len(rows)} imported official samples to {ROOT / args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import official RULER or official-style NIAH JSONL into this project's prompt/answer schema."
    )
    parser.add_argument("--source-root", required=True, help="Root containing official generated JSONL files.")
    parser.add_argument("--output", required=True, help="Output JSONL or directory under the project root.")
    parser.add_argument("--kind", choices=["ruler", "niah"], required=True)
    parser.add_argument("--subset", default="validation", help="Prefer files named SUBSET.jsonl. Use '*' for all JSONL.")
    parser.add_argument("--tasks", default=None, help="Optional comma list of official task folder names.")
    parser.add_argument("--limit-per-file", type=int, default=None)
    parser.add_argument(
        "--append-answer-prefix",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For official RULER JSONL, append answer_prefix to input to match the official generation prefix.",
    )
    return parser.parse_args()


def find_source_files(source: Path, subset: str) -> list[Path]:
    if source.is_file():
        return [source]
    if subset == "*":
        return sorted(source.rglob("*.jsonl"))
    exact = sorted(source.rglob(f"{subset}.jsonl"))
    if exact:
        return exact
    return sorted(source.rglob("*.jsonl"))


def infer_task_name(source_root: Path, path: Path) -> str:
    try:
        relative = path.relative_to(source_root)
    except ValueError:
        relative = path
    if len(relative.parts) >= 2:
        return relative.parts[-2]
    return path.stem


def convert_file(path: Path, task_name: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = []
    for idx, row in enumerate(read_jsonl(path)):
        if args.limit_per_file is not None and idx >= args.limit_per_file:
            break
        rows.append(convert_row(row, task_name, idx, path, args))
    return rows


def convert_row(row: dict[str, Any], task_name: str, idx: int, path: Path, args: argparse.Namespace) -> dict[str, Any]:
    prompt = extract_prompt(row, args.append_answer_prefix)
    answer = extract_answer(row)
    if not prompt:
        raise ValueError(
            f"{path}:{idx + 1} does not contain a reusable prompt/input. "
            "For needlehaystack v2 result rows, reconstruct prompts first with `niah reconstruct`."
        )
    if answer in (None, "", []):
        raise ValueError(f"{path}:{idx + 1} does not contain an answer/outputs field.")

    subtask = RULER_TASK_MAP.get(task_name, task_name)
    length = row.get("length") or row.get("length_tokens_target") or row.get("context_length")
    sample_id = str(row.get("sample_id") or row.get("id") or "")
    if not sample_id:
        source_slug = source_slug_for_sample(path)
        sample_id = f"official_{args.kind}_{source_slug}_{idx:06d}"
    converted = {
        "experiment": args.kind,
        "subtask": subtask,
        "official_task": task_name,
        "source_schema": f"official_{args.kind}",
        "source_file": str(path),
        "source_index": idx,
        "sample_id": sample_id,
        "prompt": prompt,
        "answer": answer,
        "length_tokens_target": length,
        "length_tokens_actual": row.get("length_w_model_temp") or length,
        "error": None,
    }
    if args.kind == "ruler":
        converted["task"] = subtask
        length_dir = infer_length_dir(path)
        if length_dir is not None:
            converted["official_length_dir"] = length_dir
    if "token_position_answer" in row:
        converted["token_position_answer"] = row.get("token_position_answer")
    if "answer_prefix" in row:
        converted["answer_prefix"] = row.get("answer_prefix")
    if "outputs" in row:
        converted["outputs"] = row.get("outputs")
    return converted


def source_slug_for_sample(path: Path) -> str:
    parts = path.with_suffix("").parts[-3:]
    return "_".join(_clean_id_part(part) for part in parts)


def infer_length_dir(path: Path) -> int | None:
    parent = path.parent.parent.name
    try:
        return int(parent)
    except ValueError:
        return None


def _clean_id_part(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value).strip("_")


def extract_prompt(row: dict[str, Any], append_answer_prefix: bool) -> str:
    if row.get("prompt"):
        return str(row["prompt"])
    if row.get("input"):
        prompt = str(row["input"])
        prefix = str(row.get("answer_prefix") or "")
        if append_answer_prefix and prefix and not prompt.rstrip().endswith(prefix.strip()):
            prompt = prompt.rstrip() + "\n" + prefix.lstrip()
        return prompt
    if row.get("context") and row.get("question"):
        return f"{row['context']}\n\nQuestion: {row['question']}\nAnswer:"
    return ""


def extract_answer(row: dict[str, Any]) -> Any:
    for key in ("answer", "answers", "outputs", "expected_answer"):
        if key in row:
            return row[key]
    score = row.get("score")
    if isinstance(score, dict) and "expected" in score:
        return score["expected"]
    return None


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}") from exc


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    main()
