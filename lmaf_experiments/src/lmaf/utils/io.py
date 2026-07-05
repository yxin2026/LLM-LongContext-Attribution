from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator


def ensure_parent(path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return
    with p.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {p}:{line_no}") from exc


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    out = ensure_parent(path)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    out = ensure_parent(path)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def iter_jsonl_paths(path: str | Path) -> Iterator[Path]:
    p = Path(path)
    if p.is_file():
        yield p
        return
    if not p.exists():
        return
    yield from sorted(p.rglob("*.jsonl"))


def load_success_ids(path: str | Path) -> set[str]:
    ids: set[str] = set()
    for p in iter_jsonl_paths(path):
        for row in read_jsonl(p):
            if row.get("sample_id") and row.get("error") in (None, ""):
                ids.add(str(row["sample_id"]))
    return ids


def collect_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for p in iter_jsonl_paths(path):
        rows.extend(read_jsonl(p))
    return rows

