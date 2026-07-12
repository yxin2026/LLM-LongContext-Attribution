from __future__ import annotations

import gzip
import json
import zipfile
from pathlib import Path
from typing import Any, Iterator

from lmaf.utils.io import write_jsonl
from lmaf.utils.token_count import TokenCounter


LONGBENCH_CATEGORIES = {
    "narrativeqa": "single_doc_qa",
    "qasper": "single_doc_qa",
    "multifieldqa_en": "single_doc_qa",
    "multifieldqa_zh": "single_doc_qa",
    "hotpotqa": "multi_doc_qa",
    "2wikimqa": "multi_doc_qa",
    "musique": "multi_doc_qa",
    "dureader": "multi_doc_qa",
    "gov_report": "summarization",
    "qmsum": "summarization",
    "multi_news": "summarization",
    "vcsum": "summarization",
}


FALLBACK_PROMPTS = {
    "zh": "\u8bf7\u9605\u8bfb\u4ee5\u4e0b\u957f\u6587\u672c\uff0c\u5e76\u6839\u636e\u6587\u672c\u5185\u5bb9\u56de\u7b54\u95ee\u9898\u3002\u53ea\u8f93\u51fa\u6700\u7ec8\u7b54\u6848\uff0c\u4e0d\u8981\u89e3\u91ca\u3002\n\n[\u957f\u6587\u672c]\n{context}\n\n[\u95ee\u9898]\n{question}\n\n[\u7b54\u6848]",
    "en": "Read the following long context and answer the question based only on the context. Output only the final answer.\n\n[Context]\n{context}\n\n[Question]\n{question}\n\n[Answer]",
}


def prepare_longbench(
    tasks: list[str],
    output_dir: str | Path,
    repo_dir: str | Path = "external/LongBench",
    sample_limit: int | None = None,
    tokenizer_name: str | None = None,
) -> list[Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    counter = TokenCounter(tokenizer_name)
    prompt_map = load_prompt_map(repo_dir)
    written: list[Path] = []
    for task in tasks:
        rows: list[dict[str, Any]] = []
        for idx, raw in enumerate(iter_longbench_rows(task, repo_dir)):
            if sample_limit is not None and idx >= sample_limit:
                break
            prompt = build_prompt(task, dict(raw), prompt_map)
            answers = raw.get("answers") or raw.get("answer") or raw.get("outputs") or []
            if isinstance(answers, str):
                answers = [answers]
            row = {
                "experiment": "longbench",
                "subtask": task,
                "sample_id": f"longbench_{task}_{idx:06d}",
                "task": task,
                "category": LONGBENCH_CATEGORIES.get(task, "unknown"),
                "context": raw.get("context") or raw.get("document") or "",
                "question": raw.get("input") or raw.get("question") or raw.get("query") or "",
                "answers": list(answers),
                "prompt": prompt,
                "length_tokens_actual": counter.count(prompt),
                "metadata": {
                    key: raw.get(key)
                    for key in ("all_classes", "length", "language")
                    if key in raw
                },
                "error": None,
            }
            rows.append(row)
        if not rows:
            raise RuntimeError(f"No LongBench rows loaded for task '{task}'. Check the task name and data source.")
        out_path = output / f"{task}.jsonl"
        write_jsonl(out_path, rows)
        written.append(out_path)
    return written


def iter_longbench_rows(task: str, repo_dir: str | Path = "external/LongBench") -> Iterator[dict[str, Any]]:
    local_file = find_local_task_file(task, repo_dir)
    if local_file:
        yield from read_json_records(local_file)
        return

    zip_path = find_local_data_zip(repo_dir)
    if zip_path is None:
        zip_path = download_longbench_data_zip()
    yield from read_zip_task_records(zip_path, task)


def find_local_task_file(task: str, repo_dir: str | Path) -> Path | None:
    root = Path(repo_dir)
    candidates = [
        root / "data" / f"{task}.jsonl",
        root / "LongBench" / "data" / f"{task}.jsonl",
        root / f"{task}.jsonl",
        root / "data" / f"{task}.jsonl.gz",
        root / "LongBench" / "data" / f"{task}.jsonl.gz",
        root / f"{task}.jsonl.gz",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if root.exists() and root.is_dir():
        for pattern in (f"{task}.jsonl", f"{task}.jsonl.gz"):
            matches = sorted(root.rglob(pattern))
            if matches:
                return matches[0]
    return None


def find_local_data_zip(repo_dir: str | Path) -> Path | None:
    root = Path(repo_dir)
    candidates = [
        root,
        root / "data.zip",
        root / "LongBench" / "data.zip",
    ]
    for candidate in candidates:
        if candidate.is_file() and candidate.suffix.lower() == ".zip":
            return candidate
    if root.exists() and root.is_dir():
        matches = sorted(root.rglob("data.zip"))
        if matches:
            return matches[0]
    return None


def download_longbench_data_zip() -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:
        raise RuntimeError(
            "huggingface_hub is required to download LongBench data.zip. "
            "Install it or place data.zip under external/LongBench."
        ) from exc
    try:
        return Path(hf_hub_download(repo_id="THUDM/LongBench", filename="data.zip", repo_type="dataset"))
    except Exception as exc:
        raise RuntimeError(
            "Failed to download THUDM/LongBench data.zip. "
            "Check network access, set HF_TOKEN if rate limited, or manually place data.zip under external/LongBench."
        ) from exc


def read_json_records(path: str | Path) -> Iterator[dict[str, Any]]:
    p = Path(path)
    opener = gzip.open if p.suffix.lower() == ".gz" else open
    with opener(p, "rt", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid LongBench JSONL at {p}:{line_no}") from exc


def read_zip_task_records(zip_path: str | Path, task: str) -> Iterator[dict[str, Any]]:
    with zipfile.ZipFile(zip_path) as zf:
        member = find_task_member(zf.namelist(), task)
        if member is None:
            raise RuntimeError(f"Task '{task}' was not found in LongBench archive {zip_path}.")
        with zf.open(member) as f:
            for line_no, raw_line in enumerate(f, start=1):
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid LongBench JSONL at {zip_path}!{member}:{line_no}") from exc


def find_task_member(members: list[str], task: str) -> str | None:
    normalized = [member.replace("\\", "/") for member in members]
    preferred = f"data/{task}.jsonl"
    for original, member in zip(members, normalized):
        if member == preferred or member.endswith(f"/{preferred}"):
            return original
    suffix = f"/{task}.jsonl"
    for original, member in zip(members, normalized):
        if member == f"{task}.jsonl" or member.endswith(suffix):
            return original
    return None


def load_prompt_map(repo_dir: str | Path) -> dict[str, str]:
    candidates = [
        Path(repo_dir) / "config" / "dataset2prompt.json",
        Path(repo_dir) / "LongBench" / "config" / "dataset2prompt.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            with candidate.open("r", encoding="utf-8") as f:
                return json.load(f)
    return {}


def build_prompt(task: str, row: dict[str, Any], prompt_map: dict[str, str] | None = None) -> str:
    prompt_map = prompt_map or {}
    template = prompt_map.get(task)
    if template:
        try:
            return template.format_map(_SafeDict(row))
        except Exception:
            pass
    context = row.get("context") or row.get("document") or row.get("article") or ""
    question = row.get("input") or row.get("question") or row.get("query") or ""
    language = "zh" if task.endswith("_zh") or task in {"dureader", "vcsum"} else "en"
    return FALLBACK_PROMPTS[language].format(context=context, question=question)


def longbench_metric_kind(task: str) -> str:
    category = LONGBENCH_CATEGORIES.get(task, "")
    if category == "summarization":
        return "summarization"
    return "qa"


class _SafeDict(dict):
    def __missing__(self, key: str) -> str:
        return ""
